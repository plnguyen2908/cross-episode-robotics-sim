"""Render one recorded frame of a saved episode two ways in MuJoCo and export it for Blender.

Presentation-only experiment: it replays recorded qpos kinematically (mj_forward, no stepping),
never changes the saved episode, and is not a robot observation (see AGENTS.md camera rules).

    .venv-sift/bin/python scripts/render_quality/mujoco_frame.py <episode dir> --frame 37500 --out runs/render_test

Writes into --out:
  mujoco_default.png   what the datagen renderer looks like (headlight ambient 0.5, default groups, 640x480)
  mujoco_enhanced.png  same frame, MuJoCo OpenGL with extra shadowed lights, MSAA, 1920x1080
  scene.npz + textures/  geometry, materials, textures and camera for blender_render.py
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

VISUAL_GROUPS = (1, 2)  # 0 = collision proxies, 3 = robot collision meshes


def load_frame(episode, frame):
    rec = episode / 'action_recording'
    model = mujoco.MjModel.from_binary_path(str(rec / 'model.mjb' if (rec / 'model.mjb').exists() else episode / 'model.mjb'))
    data = mujoco.MjData(model)
    actions = np.load(rec / 'actions.npz', allow_pickle=True)
    frame = int(np.clip(frame, 0, len(actions['qpos']) - 1))
    data.qpos[:] = actions['qpos'][frame]
    data.time = float(actions['time'][frame])
    mujoco.mj_forward(model, data)
    stages = json.loads((rec / 'metadata.json').read_text()).get('stages', [])
    stage = actions['stage'][frame]
    try:
        stage = stages[int(stage)]
    except (ValueError, IndexError, TypeError):
        stage = str(stage)
    return model, data, frame, stage


def free_camera(model, data, args):
    cam = mujoco.MjvCamera()
    if args.camera:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = model.camera(args.camera).id
        return cam
    base = data.body('robot_0/base').xpos
    hand = data.site_xpos[model.site('robot_0/gripper/grasp_site').id] if _has_site(model, 'robot_0/gripper/grasp_site') else base + [0, 0, 1]
    cam.lookat[:] = args.lookat if args.lookat else 0.5 * (base + hand) + [0, 0, 0.15]
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    return cam


def _has_site(model, name):
    try:
        model.site(name)
        return True
    except KeyError:
        return False


def render(model, data, cam, width, height, enhanced):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    option = mujoco.MjvOption()
    if enhanced:
        model.vis.quality.offsamples = 8
        model.vis.quality.shadowsize = 8192
        model.vis.headlight.ambient[:] = 0.22
        model.vis.headlight.diffuse[:] = 0.25
        model.vis.headlight.specular[:] = 0.0
        model.vis.map.shadowclip = 2.5
        option.geomgroup[:] = 0
        for g in VISUAL_GROUPS:
            option.geomgroup[g] = 1
        option.sitegroup[:] = 0  # debug markers (object frames, grasp sites)
    else:
        model.vis.headlight.ambient[:] = 0.5  # as controller/base.py sets for datagen videos
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=option)
        scn = renderer.scene
        if enhanced:
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
            lookat = np.array(cam.lookat) if cam.type == mujoco.mjtCamera.mjCAMERA_FREE else data.body('robot_0/base').xpos
            _add_light(scn, lookat + [2.5, -1.5, 4.0], directional=True, diffuse=0.55, shadow=True)
            _add_light(scn, lookat + [-2.0, 2.0, 3.0], directional=False, diffuse=0.35, shadow=True, target=lookat)
            _add_light(scn, lookat + [0.0, 0.0, 2.6], directional=False, diffuse=0.25, shadow=False, target=lookat)
        return renderer.render().copy()


def _add_light(scn, pos, directional, diffuse, shadow, target=None):
    if scn.nlight >= len(scn.lights):
        return
    light = scn.lights[scn.nlight]
    scn.nlight += 1
    target = pos - [0.3, -0.2, 1.0] if target is None else target
    direction = np.asarray(target, float) - pos
    light.pos[:] = pos
    light.dir[:] = direction / np.linalg.norm(direction)
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if directional else mujoco.mjtLightType.mjLIGHT_SPOT
    light.castshadow = int(shadow)
    light.headlight = 0
    light.ambient[:] = 0.02
    light.diffuse[:] = diffuse
    light.specular[:] = 0.15
    light.attenuation[:] = [1, 0, 0]
    light.cutoff, light.exponent = 70.0, 2.0


def gl_camera(model, data, cam, width, height):
    """Camera pose MuJoCo actually used (average of the two eyes) and its vertical FOV."""
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=cam)
        eyes = renderer.scene.camera
        pos = 0.5 * (np.array(eyes[0].pos) + np.array(eyes[1].pos))
        forward, up = np.array(eyes[0].forward), np.array(eyes[0].up)
        top, bottom, near = eyes[0].frustum_top, eyes[0].frustum_bottom, eyes[0].frustum_near
    fovy = 2 * np.degrees(np.arctan(0.5 * (top - bottom) / near))
    return dict(pos=pos, forward=forward, up=up, fovy=fovy)


def export(model, data, cam, out, width, height):
    tex_dir = out / 'textures'
    tex_dir.mkdir(parents=True, exist_ok=True)
    keep = np.isin(model.geom_group, VISUAL_GROUPS) & (model.geom_rgba[:, 3] > 0) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    geoms = np.where(keep)[0]
    used_meshes = sorted({int(model.geom_dataid[g]) for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH})
    used_mats = sorted({int(model.geom_matid[g]) for g in geoms if model.geom_matid[g] >= 0})
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    used_tex = sorted({int(model.mat_texid[m, rgb_role]) for m in used_mats if model.mat_texid[m, rgb_role] >= 0})
    for t in used_tex:
        w, h, c = model.tex_width[t], model.tex_height[t], model.tex_nchannel[t]
        img = model.tex_data[model.tex_adr[t]:model.tex_adr[t] + w * h * c].reshape(h, w, c)
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_CUBE and h == 6 * w:
            img = img[:w]  # one face is enough for box projection
        iio.imwrite(tex_dir / f'tex_{t}.png', img[..., :3] if c >= 3 else np.repeat(img, 3, -1))
    meshes = {}
    for mid in used_meshes:
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        entry = dict(vert=model.mesh_vert[va:va + vn].astype(np.float32), face=model.mesh_face[fa:fa + fn].astype(np.int32))
        tn = model.mesh_texcoordnum[mid]
        if tn > 0:
            ta = model.mesh_texcoordadr[mid]
            entry['uv'] = model.mesh_texcoord[ta:ta + tn].astype(np.float32)
            entry['face_uv'] = model.mesh_facetexcoord[fa:fa + fn].astype(np.int32)
        meshes[mid] = entry
    flat = {}
    for mid, entry in meshes.items():
        for k, v in entry.items():
            flat[f'mesh{mid}_{k}'] = v
    cam_info = gl_camera(model, data, cam, width, height)
    np.savez_compressed(
        out / 'scene.npz',
        geom_ids=geoms, geom_type=model.geom_type[geoms], geom_size=model.geom_size[geoms],
        geom_xpos=data.geom_xpos[geoms], geom_xmat=data.geom_xmat[geoms], geom_rgba=model.geom_rgba[geoms],
        geom_matid=model.geom_matid[geoms], geom_dataid=model.geom_dataid[geoms],
        geom_body_xpos=data.xpos[model.geom_bodyid[geoms]], geom_body_xmat=data.xmat[model.geom_bodyid[geoms]],
        geom_names=np.array([model.geom(g).name for g in geoms]),
        mat_ids=np.array(used_mats, int), mat_rgba=model.mat_rgba[used_mats], mat_texid=model.mat_texid[used_mats, rgb_role],
        mat_texrepeat=model.mat_texrepeat[used_mats], mat_texuniform=model.mat_texuniform[used_mats],
        mat_specular=model.mat_specular[used_mats], mat_shininess=model.mat_shininess[used_mats],
        mat_reflectance=model.mat_reflectance[used_mats],
        cam_pos=cam_info['pos'], cam_forward=cam_info['forward'], cam_up=cam_info['up'], cam_fovy=cam_info['fovy'],
        **flat)
    return len(geoms), len(used_meshes), len(used_tex), cam_info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episode', type=Path)
    p.add_argument('--frame', type=int, default=37500, help='index into action_recording/actions.npz')
    p.add_argument('--out', type=Path, default=Path('runs/render_test'))
    p.add_argument('--camera', help='named model camera instead of the free camera')
    p.add_argument('--lookat', type=float, nargs=3)
    p.add_argument('--azimuth', type=float, default=135.0)
    p.add_argument('--elevation', type=float, default=-28.0)
    p.add_argument('--distance', type=float, default=3.2)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, data, frame, stage = load_frame(args.episode, args.frame)
    cam = free_camera(model, data, args)
    # Datagen resolution is 640x480; keep 640 wide but match the output aspect so the views line up.
    iio.imwrite(args.out / 'mujoco_default.png',
                render(model, data, cam, 640, round(640 * args.height / args.width), enhanced=False))
    model, data, frame, stage = load_frame(args.episode, args.frame)  # fresh vis settings
    cam = free_camera(model, data, args)
    iio.imwrite(args.out / 'mujoco_enhanced.png', render(model, data, cam, args.width, args.height, enhanced=True))
    n_geoms, n_meshes, n_tex, cam_info = export(model, data, cam, args.out, args.width, args.height)
    info = dict(episode=str(args.episode.resolve()), frame=frame, time=float(data.time), stage=stage,
                width=args.width, height=args.height, geoms=n_geoms, meshes=n_meshes, textures=n_tex,
                camera={k: np.asarray(v).tolist() for k, v in cam_info.items()},
                free_camera=None if args.camera else dict(lookat=list(cam.lookat), azimuth=cam.azimuth,
                                                          elevation=cam.elevation, distance=cam.distance))
    (args.out / 'frame.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
