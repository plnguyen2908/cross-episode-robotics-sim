"""Path-trace a frame exported by mujoco_frame.py with Blender Cycles (bpy module).

    .venv-render/bin/python scripts/render_quality/blender_render.py runs/render_test --samples 256

Setup (separate venv; bpy pins its own Python 3.11 build):
    uv venv --python 3.11 .venv-render && uv pip install --python .venv-render/bin/python bpy==4.2.0

Presentation-only: geometry and poses come straight from the exported MuJoCo state; only the
look (lights, materials, path tracing) differs. Writes <dir>/blender_cycles.png (and .blend with --save-blend).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asset_swap  # noqa: E402

RGBA_SRGB = True
BOX, CAPSULE, CYLINDER, ELLIPSOID, SPHERE, MESH = 6, 3, 5, 4, 2, 7


def srgb_to_linear(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles(args):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    device = 'CPU'
    for kind in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                device = kind
                break
        except TypeError:
            continue
    scene.cycles.device = 'GPU' if device != 'CPU' else 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = args.exposure
    return device


def make_material(name, rgba, tex_path, repeat, uniform, shininess, reflectance, specular):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes['Principled BSDF']
    # MuJoCo rgba values are display (sRGB) colours; decode them to linear albedo. --rgba-linear skips the
    # decode, which mimics MuJoCo's over-bright summed lighting but washes the image out.
    rgb = srgb_to_linear(rgba[:3]) if RGBA_SRGB else np.clip(rgba[:3], 0, 1)
    color = list(rgb) + [1.0]
    bsdf.inputs['Base Color'].default_value = color
    # MuJoCo's Phong shininess/reflectance -> a plausible roughness; nothing here is physical ground truth.
    bsdf.inputs['Roughness'].default_value = float(np.clip(0.85 - 0.6 * shininess - 0.5 * reflectance, 0.08, 0.9))
    bsdf.inputs['Specular IOR Level'].default_value = float(np.clip(0.3 + 0.4 * specular, 0.2, 0.7))
    if rgba[3] < 0.999:
        bsdf.inputs['Alpha'].default_value = float(rgba[3])
    if tex_path is not None:
        img = bpy.data.images.load(str(tex_path), check_existing=True)
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = img
        tex.interpolation = 'Smart'
        coord = nodes.new('ShaderNodeTexCoord')
        mapping = nodes.new('ShaderNodeMapping')
        if name.startswith('uvmat'):
            links.new(coord.outputs['UV'], mapping.inputs['Vector'])
        else:
            # Primitive geoms: MuJoCo derives 2D texture coordinates from the geom's own frame.
            tex.projection = 'BOX'
            tex.projection_blend = 0.15
            links.new(coord.outputs['Object' if uniform else 'Generated'], mapping.inputs['Vector'])
        mapping.inputs['Scale'].default_value = (repeat[0], repeat[1], repeat[0])
        links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
        mix = nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Factor'].default_value = 1.0
        mix.inputs[6].default_value = color
        links.new(tex.outputs['Color'], mix.inputs[7])
        links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    return mat


def primitive_mesh(kind, size, name):
    if kind == BOX:
        bpy.ops.mesh.primitive_cube_add(size=2)
        obj = bpy.context.object
        obj.data.transform(Matrix.Diagonal((*size, 1)))
    elif kind in (SPHERE, ELLIPSOID):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=48, ring_count=24)
        obj = bpy.context.object
        obj.data.transform(Matrix.Diagonal((size[0], size[0], size[0], 1) if kind == SPHERE else (*size, 1)))
        obj.data.shade_smooth()
    elif kind == CYLINDER:
        bpy.ops.mesh.primitive_cylinder_add(radius=size[0], depth=2 * size[1], vertices=48)
        obj = bpy.context.object
        obj.data.shade_smooth()
        _sharp(obj.data, 40)
    elif kind == CAPSULE:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size[0], segments=32, ring_count=16)
        obj = bpy.context.object
        for v in obj.data.vertices:
            v.co.z += size[1] if v.co.z > 0 else -size[1]
        obj.data.shade_smooth()
    else:
        return None
    obj.name = name
    return obj


def _sharp(mesh, degrees):
    try:
        mesh.set_sharp_from_angle(angle=math.radians(degrees))
    except AttributeError:
        pass


def mujoco_mesh(z, mid, name, cache):
    if mid in cache:
        return cache[mid]
    vert, face = z[f'mesh{mid}_vert'], z[f'mesh{mid}_face']
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vert.tolist(), [], face.tolist())
    if f'mesh{mid}_uv' in z.files:
        uv, face_uv = z[f'mesh{mid}_uv'], z[f'mesh{mid}_face_uv']
        layer = mesh.uv_layers.new(name='UVMap')
        loops = uv[face_uv.reshape(-1)]
        loops[:, 1] = 1.0 - loops[:, 1]  # MuJoCo stores v flipped relative to Blender
        layer.data.foreach_set('uv', loops.reshape(-1))
    mesh.validate()
    mesh.shade_smooth()
    _sharp(mesh, 35)
    cache[mid] = mesh
    return mesh


def build(z, tex_dir):
    mats = {int(m): i for i, m in enumerate(z['mat_ids'])}
    material_cache, mesh_cache = {}, {}
    default_rgba = np.array([0.5, 0.5, 0.5, 1.0])
    for k in range(len(z['geom_ids'])):
        kind, size = int(z['geom_type'][k]), z['geom_size'][k]
        matid, rgba = int(z['geom_matid'][k]), z['geom_rgba'][k]
        name = f"g{int(z['geom_ids'][k])}_{z['geom_names'][k]}"[:60]
        if kind == MESH:
            mid = int(z['geom_dataid'][k])
            obj = bpy.data.objects.new(name, mujoco_mesh(z, mid, f'mesh{mid}', mesh_cache))
            bpy.context.collection.objects.link(obj)
            has_uv = f'mesh{mid}_uv' in z.files
        else:
            obj = primitive_mesh(kind, size, name)
            has_uv = False
            if obj is None:
                continue
        # Material colour follows MuJoCo: a non-default geom rgba overrides the material rgba.
        if matid >= 0:
            j = mats[matid]
            color = rgba if not np.allclose(rgba, default_rgba) else z['mat_rgba'][j]
            texid = int(z['mat_texid'][j])
            tex_path = tex_dir / f'tex_{texid}.png' if texid >= 0 else None
            key = (matid, tuple(np.round(color, 4)), has_uv)
            if key not in material_cache:
                material_cache[key] = make_material(
                    f"{'uvmat' if has_uv else 'mat'}{matid}", color, tex_path, z['mat_texrepeat'][j],
                    bool(z['mat_texuniform'][j]), float(z['mat_shininess'][j]),
                    float(z['mat_reflectance'][j]), float(z['mat_specular'][j]))
            mat = material_cache[key]
        else:
            key = ('rgba', tuple(np.round(rgba, 4)))
            if key not in material_cache:
                material_cache[key] = make_material(f'rgba{len(material_cache)}', rgba, None, (1, 1), False, 0.5, 0.0, 0.5)
            mat = material_cache[key]
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        rot = Matrix(z['geom_xmat'][k].reshape(3, 3).tolist())
        obj.matrix_world = Matrix.Translation(Vector(z['geom_xpos'][k].tolist())) @ rot.to_4x4()
    return len(z['geom_ids'])


def camera(z, args):
    pos, forward, up = (Vector(z[k].tolist()) for k in ('cam_pos', 'cam_forward', 'cam_up'))
    cam = bpy.data.cameras.new('cam')
    cam.sensor_fit = 'VERTICAL'
    cam.angle_y = math.radians(float(z['cam_fovy']))
    cam.clip_start, cam.clip_end = 0.02, 200
    obj = bpy.data.objects.new('cam', cam)
    bpy.context.collection.objects.link(obj)
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    obj.matrix_world = Matrix((
        (right.x, up.x, -forward.x, pos.x),
        (right.y, up.y, -forward.y, pos.y),
        (right.z, up.z, -forward.z, pos.z),
        (0, 0, 0, 1)))
    bpy.context.scene.camera = obj
    return obj


def lighting(z, args):
    world = bpy.data.worlds.new('world')
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    sky = nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'NISHITA'
    sky.sun_elevation = math.radians(args.sun_elevation)
    sky.sun_rotation = math.radians(args.sun_azimuth)
    sky.sun_disc = False
    world.node_tree.links.new(sky.outputs['Color'], nodes['Background'].inputs['Color'])
    nodes['Background'].inputs['Strength'].default_value = args.sky_strength
    sun = bpy.data.lights.new('sun', 'SUN')
    sun.energy = args.sun_strength
    sun.angle = math.radians(6)  # soft shadow edges
    sun_obj = bpy.data.objects.new('sun', sun)
    sun_obj.rotation_euler = (math.radians(90 - args.sun_elevation), 0, math.radians(args.sun_azimuth + 90))
    bpy.context.collection.objects.link(sun_obj)
    # Large soft key light above the work area, like an interior ceiling panel.
    target = Vector(z['cam_pos'].tolist()) + 3.2 * Vector(z['cam_forward'].tolist())
    area = bpy.data.lights.new('ceiling', 'AREA')
    area.shape, area.size, area.size_y = 'RECTANGLE', 3.0, 2.0
    area.energy = args.area_power
    area_obj = bpy.data.objects.new('ceiling', area)
    area_obj.location = (target.x, target.y, 2.9)
    bpy.context.collection.objects.link(area_obj)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dir', type=Path)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--exposure', type=float, default=0.3)
    p.add_argument('--rgba-linear', action='store_true', help='use MuJoCo rgba as linear albedo (brighter, washed out)')
    p.add_argument('--sun-elevation', type=float, default=40.0)
    p.add_argument('--sun-azimuth', type=float, default=-35.0)
    p.add_argument('--sun-strength', type=float, default=3.0)
    p.add_argument('--sky-strength', type=float, default=0.6)
    p.add_argument('--area-power', type=float, default=600.0)
    p.add_argument('--save-blend', action='store_true')
    p.add_argument('--out', default='blender_cycles.png')
    p.add_argument('--swap', action='append', metavar='PREFIX[,PREFIX]=ASSET_DIR',
                   help='render-only: replace geoms named PREFIX* with a SimReady or glTF asset; comma-separate prefixes to merge them into one object (repeatable)')
    p.add_argument('--swap-scale', choices=('fit', 'uniform'), default='fit')
    args = p.parse_args()

    global RGBA_SRGB
    RGBA_SRGB = not args.rgba_linear
    z = np.load(args.dir / 'scene.npz', allow_pickle=False)
    reset()
    device = setup_cycles(args)
    n = build(z, (args.dir / 'textures').resolve())
    swaps = asset_swap.apply(z, asset_swap.parse_rules(args.swap), args.swap_scale)
    if swaps:
        (args.dir / (Path(args.out).stem + '_swaps.json')).write_text(json.dumps(swaps, indent=2))
    camera(z, args)
    lighting(z, args)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str((args.dir / 'scene.blend').resolve()), relative_remap=False)
    bpy.context.scene.render.filepath = str((args.dir / args.out).resolve())
    import time
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f'rendered {n} geoms on {device} with {args.samples} samples in {time.time() - t:.1f}s -> {args.dir / args.out}')


if __name__ == '__main__':
    main()
