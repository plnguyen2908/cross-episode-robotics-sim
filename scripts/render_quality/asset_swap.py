"""Render-only replacement of simulated objects with higher-quality meshes (runs inside bpy).

Asset sources that work (each needs attribution per its license when a render is published):
  * NVIDIA SimReady, https://huggingface.co/datasets/nvidia/simready-assets (CC BY 4.0): pass the asset
    folder (it contains simready_usd/sm_*.usd[a]); materials come from the USD preview surfaces.
  * Poly Haven models, https://polyhaven.com/models (CC0): pass the folder holding the *.gltf.

A rule is ``PREFIX=ASSET_DIR``. With one prefix, each numbered instance (``bowl_01``, ``chair_2``) is
replaced separately. With several comma-separated prefixes (``dining_table_,table_leg_=...``) all
matching geoms form ONE object, e.g. a procedurally built table made of a top and four leg boxes.

The asset is posed in the instance's MuJoCo body frame. Up axis and yaw come from a chamfer search
against points sampled on the simulated visual geoms (mesh vertices, primitive surfaces), so handles
and chair backs face the right way. ``fit`` scaling (default) stretches each body axis to the
simulated bounding box so the contacts physics produced still read correctly; ``uniform`` keeps the
asset's proportions. This changes pixels only: physics, grasps and scoring ran on the original
objects, and a demo rendered this way must say so.
"""

import glob
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

MESH, BOX = 7, 6


def parse_rules(values):
    rules = []
    for value in values or []:
        prefixes, _, folder = value.partition('=')
        if not folder:
            raise SystemExit(f'--swap expects PREFIX[,PREFIX...]=ASSET_DIR, got {value!r}')
        rules.append((tuple(p for p in prefixes.split(',') if p), Path(folder)))
    return rules


def instance_key(name, prefixes):
    if len(prefixes) > 1:
        return prefixes[0].rstrip('_')
    prefix = prefixes[0]
    token = name[len(prefix):].split('_')[0]
    return prefix + token if token.isdigit() else prefix.rstrip('_')


def collect_instances(z, rules):
    """{key: (asset folder, [geom row indices])} for every geom matching a rule."""
    found = {}
    for k, name in enumerate(z['geom_names']):
        name = str(name)
        for prefixes, folder in rules:
            if any(name.startswith(p) for p in prefixes):
                found.setdefault(instance_key(name, prefixes), (folder, []))[1].append(k)
                break
    return found


def world_points(z, rows, rng):
    """Points on the simulated visual surface: mesh vertices, or samples on primitive (box-like) faces."""
    pts = []
    for k in rows:
        R, p = z['geom_xmat'][k].reshape(3, 3), z['geom_xpos'][k]
        if int(z['geom_type'][k]) == MESH:
            local = z[f"mesh{int(z['geom_dataid'][k])}_vert"]
        else:
            half = z['geom_size'][k].copy()
            if int(z['geom_type'][k]) != BOX:  # cylinder/capsule/sphere: radius, radius, half-length
                half = np.array([half[0], half[0], half[1] if half[1] > 0 else half[0]])
            n = int(np.clip(4e4 * np.prod(np.sort(half)[1:]), 60, 600))
            local = rng.uniform(-1, 1, size=(n, 3))
            axis = rng.integers(0, 3, size=n)
            local[np.arange(n), axis] = np.sign(local[np.arange(n), axis] + 1e-9)
            local *= half
        pts.append(local @ R.T + p)
    return np.concatenate(pts) if pts else None


def _sample(points, n, rng):
    return points[rng.choice(len(points), size=min(n, len(points)), replace=False)]


def _chamfer(a, b):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d.min(1)).mean() + np.sqrt(d.min(0)).mean())


def import_asset(folder):
    """Import a SimReady USD or a glTF, keep its materials, and bake it into one asset-frame mesh."""
    usd = sorted(glob.glob(str(folder / 'simready_usd' / 'sm_*.usd*')))
    gltf = sorted(glob.glob(str(folder / '*.gltf')) + glob.glob(str(folder / '*.glb')))
    before = set(bpy.data.objects)
    if usd:
        bpy.ops.wm.usd_import(filepath=usd[0])
    elif gltf:
        bpy.ops.import_scene.gltf(filepath=gltf[0])
    else:
        raise SystemExit(f'{folder}: no simready_usd/sm_*.usd* or *.gltf/*.glb')
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == 'MESH']
    if not meshes:
        raise SystemExit(f'{folder}: import produced no meshes')
    for o in meshes:  # bake hierarchy transforms into vertices, so the mesh is in the asset frame
        o.data = o.data.copy()
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh = meshes[0].data
    mesh.name = f'asset_{folder.name}'
    for o in new:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    return mesh


def _axis_rotations():
    """Rotations taking the asset's +Z (up) to each of the six body axes; body frames differ per source."""
    rx = lambda a: np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = lambda a: np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    h = math.pi / 2
    return {'+z': np.eye(3), '-z': rx(math.pi), '+x': ry(h), '-x': ry(-h), '+y': rx(-h), '-y': rx(h)}


def _place(asset_pts, rot, sim_lo, sim_hi, scale_mode):
    p = asset_pts @ rot.T
    lo, hi = p.min(0), p.max(0)
    scale = (sim_hi - sim_lo) / np.maximum(hi - lo, 1e-6)
    if scale_mode == 'uniform':
        scale = np.full(3, float(np.cbrt(np.prod(scale))))
    offset = 0.5 * (sim_lo + sim_hi) - 0.5 * (lo + hi) * scale
    return p, scale, offset


def fit_pose(asset_pts, sim_local, scale_mode, rng):
    """Best up-axis, yaw and per-axis scale/offset mapping asset points onto the sim points (body frame).

    Scale is applied in the body frame after rotation, so a stretch never shears the asset."""
    sim_lo, sim_hi = sim_local.min(0), sim_local.max(0)
    sim_s = _sample(sim_local, 800, rng)
    asset_s = _sample(asset_pts, 800, rng)
    best = None
    for up, base in _axis_rotations().items():
        for deg in range(0, 360, 5):
            c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            rot = base @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            p, scale, offset = _place(asset_s, rot, sim_lo, sim_hi, scale_mode)
            # Penalise strong stretching: with per-axis fitting a wrong up axis can otherwise fit anything.
            cost = _chamfer(p * scale + offset, sim_s) * (1 + abs(np.log(scale)).max())
            if best is None or cost < best[0]:
                best = (cost, f'{up} yaw {deg}', rot)
    cost, label, rot = best
    _, scale, offset = _place(asset_pts, rot, sim_lo, sim_hi, scale_mode)
    return cost, label, rot, scale, offset


def apply(z, rules, scale_mode='fit', log=print):
    """Replace matching instances; returns {key: info}. Call after the scene's geoms were built."""
    if not rules:
        return {}
    rng = np.random.default_rng(0)
    meshes, report = {}, {}
    for key, (folder, rows) in sorted(collect_instances(z, rules).items()):
        sim_world = world_points(z, rows, rng)
        body_R = z['geom_body_xmat'][rows[0]].reshape(3, 3)
        body_p = z['geom_body_xpos'][rows[0]]
        sim_local = (sim_world - body_p) @ body_R
        if folder not in meshes:
            meshes[folder] = import_asset(folder)
        mesh = meshes[folder]
        asset_pts = np.array([v.co[:] for v in mesh.vertices])
        cost, label, rot, scale, offset = fit_pose(asset_pts, sim_local, scale_mode, rng)
        local = np.eye(4)
        local[:3, :3] = np.diag(scale) @ rot
        local[:3, 3] = offset
        body = np.eye(4)
        body[:3, :3], body[:3, 3] = body_R, body_p
        obj = bpy.data.objects.new(f'swap_{key}', mesh)
        bpy.context.collection.objects.link(obj)
        obj.matrix_world = Matrix((body @ local).tolist())
        for k in rows:
            original = bpy.data.objects.get(_geom_object_name(z, k))
            if original is not None:
                original.hide_render = original.hide_viewport = True
        report[key] = dict(asset=folder.name, geoms=len(rows), orientation=label,
                           scale=scale.round(3).tolist(), weighted_chamfer_m=round(cost, 4))
        log(f'swap {key} ({len(rows)} geoms) -> {folder.name}: {label} deg, '
            f'scale {np.round(scale, 2).tolist()}, weighted chamfer {cost * 1000:.1f} mm')
    return report


def _geom_object_name(z, k):
    return f"g{int(z['geom_ids'][k])}_{z['geom_names'][k]}"[:60]
