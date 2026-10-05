"""Convert the public Moonlake espresso USDZ to an inspectable MJCF asset.

Requires usd-core, trimesh, vhacdx, scipy and mujoco. Source joint frames and
limits are retained. Collision hulls are regenerated from the source meshes;
materials are approximated by their diffuse colors (no texture conversion).
The source's zero masses are replaced with explicit diagnostic estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import vhacdx
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

SOURCE = 'https://media.githubusercontent.com/media/MoonlakeAI/sim-env-builder/main/assets/library/espresso_machine.usdz'
LICENSE = 'https://raw.githubusercontent.com/MoonlakeAI/sim-env-builder/main/LICENSE'
SOURCE_SHA256 = '835e9070409e8db964e6934f4a6dcec0f549b80a63078a27c27cf06778272ba5'


def numbers(values):
    return ' '.join(f'{float(v):.10g}' for v in values)


def transform(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def triangles(mesh):
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    faces, owners, offset = [], [], 0
    for face_id, n in enumerate(mesh.GetFaceVertexCountsAttr().Get()):
        polygon = indices[offset:offset+n]
        faces.extend([polygon[0], polygon[i], polygon[i+1]] for i in range(1, n-1))
        owners.extend([face_id] * (n-2))
        offset += n
    return np.asarray(faces), np.asarray(owners)


def portafilter_socket():
    """Collision-only socket repair for the source's overlapping seated parts.

    Keep the original group-head exterior and visual mesh. The lower 12.4 mm
    becomes an annular socket with 83 mm diameter clearance for the basket's
    locking tabs; a cap above the rim closes the head. This is authored geometry,
    not a claim that the source includes a working locking mechanism.
    """
    center = np.array([-.0173, -.0193])
    cap = trimesh.creation.cylinder(radius=.048, height=.227-.2075, sections=64)
    cap.apply_translation([*center, (.227+.2075)/2])
    hulls = [(cap.vertices, cap.faces)]
    for i in range(32):
        angles = 2*np.pi*np.array([i, i+1])/32
        vertices = [[center[0]+radius*np.cos(a), center[1]+radius*np.sin(a), z]
                    for z in (.1951, .2075) for radius in (.0415, .048) for a in angles]
        hull = trimesh.Trimesh(vertices=np.array(vertices), faces=[]).convex_hull
        hulls.append((hull.vertices, hull.faces))
    return hulls


def convert(source, output, repair_socket=False):
    output.mkdir(parents=True, exist_ok=False)
    mesh_dir = output / 'meshes'
    mesh_dir.mkdir()
    if source is None:
        source = output / 'source.usdz'
        urllib.request.urlretrieve(SOURCE, source)
    else:
        shutil.copyfile(source, output / 'source.usdz')
        source = output / 'source.usdz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Source changed: inspect the new asset before updating the pinned hash')
    urllib.request.urlretrieve(LICENSE, output / 'LICENSE')
    (output / 'NOTICE.txt').write_text(
        'MoonlakeAI SimEnvBuilder espresso_machine, Apache-2.0.\n'
        f'Source: {SOURCE}\nSHA256: {digest}\n'
        'Modified: USD to MJCF conversion, collision decomposition, material '
        'approximations and estimated masses. No brewing or granular behavior '
        'is provided by the original asset.\n'
        + ('Group-head collision socket additionally authored for portafilter clearance.\n' if repair_socket else ''))
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Expected meter-scale, Z-up source')
    root = ET.Element('mujoco', model='moonlake_espresso_machine')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep='.001', gravity='0 0 -9.81', integrator='implicitfast')
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    body_prims = {str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    body_poses = {p: transform(prim) for p, prim in body_prims.items()}
    joints = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            child = str(joint.GetBody1Rel().GetTargets()[0])
            parents = joint.GetBody0Rel().GetTargets()
            joints[child] = (str(parents[0]) if parents else None, prim)
    nodes = {}
    joint_report = []

    def add_body(path):
        if path in nodes:
            return nodes[path]
        parent, joint_prim = joints[path]
        parent_node = add_body(parent) if parent else world
        pose = np.linalg.inv(body_poses[parent]) @ body_poses[path] if parent else body_poses[path]
        node = ET.SubElement(parent_node, 'body', name=body_prims[path].GetName(),
                             pos=numbers(pose[:3, 3]),
                             quat=numbers(Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)))
        nodes[path] = node
        # USD source reports mass=0 for every body, including the portafilter.
        mass = .001 if 'twist_pivot' in path else (.35 if 'portafilter' in path else .1)
        ET.SubElement(node, 'inertial', pos='0 0 0', mass=str(mass), diaginertia=numbers([mass*.002]*3))
        if not joint_prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(joint_prim)
            q = joint.GetLocalRot1Attr().Get()
            rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()])
            axis = rotation.apply(np.eye(3)['XYZ'.index(joint_prim.GetAttribute('physics:axis').Get())])
            lower, upper = [joint_prim.GetAttribute('physics:'+a).Get() for a in ('lowerLimit', 'upperLimit')]
            kind = 'hinge' if joint_prim.IsA(UsdPhysics.RevoluteJoint) else 'slide'
            limits = np.deg2rad([lower, upper]) if kind == 'hinge' else [lower, upper]
            ET.SubElement(node, 'joint', name=joint_prim.GetName(), type=kind,
                          pos=numbers(joint.GetLocalPos1Attr().Get()), axis=numbers(axis),
                          range=numbers(limits), damping='2', armature='.001')
            joint_report.append(dict(name=joint_prim.GetName(), type=kind, range=list(limits),
                                     axis=axis.tolist(), body=node.get('name')))
        return node

    for path in body_prims:
        add_body(path)
    mesh_report = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        parent = prim
        while str(parent.GetPath()) not in body_prims:
            parent = parent.GetParent()
        path = str(parent.GetPath())
        node = nodes[path]
        mesh = UsdGeom.Mesh(prim)
        pose = np.linalg.inv(body_poses[path]) @ transform(prim)
        vertices = np.array(mesh.GetPointsAttr().Get())
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ pose.T)[:, :3]
        faces, owners = triangles(mesh)
        name = prim.GetName()
        geometry = trimesh.Trimesh(vertices, faces, process=True)
        subsets = [p for p in prim.GetChildren() if p.IsA(UsdGeom.Subset)]
        face_material = np.full(len(faces), -1)
        for i, sub in enumerate(subsets):
            face_material[np.isin(owners, sub.GetAttribute('indices').Get())] = i
        for i in np.unique(face_material):
            part = trimesh.Trimesh(vertices, faces[face_material == i], process=True)
            part.remove_unreferenced_vertices()
            label = f'{name}_visual_{i+1}'
            filename = mesh_dir / f'{label}.obj'
            part.export(filename)
            binding = subsets[i] if i >= 0 else prim
            material, _ = UsdShade.MaterialBindingAPI(binding).ComputeBoundMaterial()
            color, metallic, roughness = [.55, .57, .60], .3, .4
            if material:
                shader = UsdShade.Shader(material.GetPrim().GetChild('Principled_BSDF'))
                for key in ('diffuseColor', 'metallic', 'roughness'):
                    value = shader.GetInput(key).Get() if shader else None
                    if value is not None:
                        if key == 'diffuseColor': color = list(value)
                        elif key == 'metallic': metallic = float(value)
                        else: roughness = float(value)
            ET.SubElement(assets, 'mesh', name=label, file=str(filename.resolve()))
            ET.SubElement(assets, 'material', name=label, rgba=numbers([*color, 1]),
                          specular=str(.2+.5*metallic), shininess=str(1-roughness))
            ET.SubElement(node, 'geom', name=label, type='mesh', mesh=label, material=label,
                          group='1', contype='0', conaffinity='0', mass='0')
        # Each convex piece is a separate geom, so the open basket stays open.
        # vhacdx takes VTK polygons: [3, i, j, k, 3, ...], NOT an Nx3
        # triangle array. Passing triangles silently produces incomplete hulls.
        vtk_faces = np.column_stack([np.full(len(geometry.faces), 3), geometry.faces]).astype(np.uint32).ravel()
        if repair_socket and name == 'group_head_cage_001':
            if not np.allclose(body_poses[path], np.eye(4), atol=1e-6):
                raise ValueError('Socket repair assumes the source group-head body frame is world aligned')
            hulls = portafilter_socket()
        else:
            hulls = vhacdx.compute_vhacd(geometry.vertices, vtk_faces,
                                        maxConvexHulls=64, resolution=400000,
                                        minimumVolumePercentErrorAllowed=.1, maxRecursionDepth=12)
        for i, (v, f) in enumerate(hulls):
            hull = trimesh.Trimesh(v, f, process=True)
            if abs(hull.volume) < 1e-12:
                continue
            label = f'{name}_collision_{i}'
            filename = mesh_dir / f'{label}.obj'
            hull.export(filename)
            ET.SubElement(assets, 'mesh', name=label, file=str(filename.resolve()))
            ET.SubElement(node, 'geom', name=label, type='mesh', mesh=label, group='0',
                          friction='.6 .005 .0001', solref='.004 1', mass='0')
        hull_vertices = np.concatenate([v for v, _ in hulls])
        bounds_error = float(np.max(np.abs(np.array([hull_vertices.min(0), hull_vertices.max(0)])-geometry.bounds)))
        if bounds_error > .002:
            raise ValueError(f'{name}: collision decomposition lost geometry ({bounds_error:.4f} m bounds error)')
        mesh_report.append(dict(name=name, source_vertices=len(vertices), convex_pieces=len(hulls),
                                collision_bounds_error_m=bounds_error))
        print(f'{name}: {len(vertices)} vertices, {len(hulls)} collision pieces', flush=True)
    ET.indent(root)
    scene = output / 'machine.xml'
    ET.ElementTree(root).write(scene)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    report = dict(source_url=SOURCE, source_sha256=digest, license='Apache-2.0',
                  material_conversion='Diffuse-color approximation; source textures not converted',
                  mass_conversion='Source zero masses replaced by diagnostic estimates',
                  socket_repair=repair_socket,
                  joints=joint_report, meshes=mesh_report,
                  body_world_poses={body_prims[p].GetName(): pose.tolist() for p, pose in body_poses.items()},
                  compiled=dict(nbody=model.nbody, njnt=model.njnt, ngeom=model.ngeom))
    (output / 'conversion.json').write_text(json.dumps(report, indent=2)+'\n')
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair-socket', action='store_true', help='Author a collision socket at the overlapping portafilter/group-head interface')
    args = parser.parse_args()
    print(convert(args.source, args.output.resolve(), args.repair_socket))


if __name__ == '__main__':
    main()
