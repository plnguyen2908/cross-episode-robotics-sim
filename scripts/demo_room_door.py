"""Render the existing non-latched room doors under diagnostic joint torques."""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from cross_episode_sim.tasks.room_doors import install_room_doors


def main():
    Path('runs').mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='non_latched_room_door_', dir='runs'))
    print(f'OUTPUT={out.resolve()}', flush=True)
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = .002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 720
    for pos in ([0, -3, 4], [1, 3, 4]):
        spec.worldbody.add_light(pos=pos, dir=[0, 0, -1], diffuse=[.8, .8, .8])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[4, 4, .1], rgba=[.35, .38, .4, 1])
    for side, x in [('left', 0), ('right', 2)]:
        post = spec.worldbody.add_body(name=f'doorframe_0_{side}_demo_main',
                                      pos=[x, 0, 1.08])
        post.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.05, .08, 1.08],
                      rgba=[.8, .8, .78, 1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[1, 0, 2.2], size=[1.05, .08, .04],
                           rgba=[.8, .8, .78, 1])
    metadata = install_room_doors(spec, {'room_door_0': 'closed'})
    model = spec.compile()
    spec.to_file(str(out / 'scene.xml'))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1, .3, 1.05]
    camera.distance = 4.5
    camera.azimuth = 110
    camera.elevation = -18
    phases = [('Closed - fixed handles, no latch', 1., [0., 0.]),
              ('Open both leaves', 4., [5., -5.]),
              ('Open at 90 degrees', 1., [0., 0.]),
              ('Close both leaves', 4., [-5., 5.]),
              ('Closed', 1., [0., 0.])]
    rows, trace = [], []
    frame_index = 0
    with mujoco.Renderer(model, 720, 960) as renderer, imageio.get_writer(
            str(out / 'mechanism_test.mp4'), fps=25, codec='libx264', quality=8) as writer:
        for label, seconds, torque in tqdm(phases, desc='Non-latched door'):
            data.qfrc_applied[:] = torque
            for _ in range(round(seconds / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if data.time + 1e-9 < frame_index / 25:
                    continue
                frame_index += 1
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 48), fill='black')
                draw.text((10, 7), 'NON-LATCHED ROOM DOORS - mechanism test, external torques, NO ROBOT', fill='white')
                draw.text((10, 28), label, fill='white')
                writer.append_data(np.asarray(frame))
                trace.append(dict(time=float(data.time), phase=label,
                                  qpos=data.qpos.tolist(), external_torque_nm=torque))
            rows.append(dict(phase=label, angles_deg=np.degrees(data.qpos).tolist()))
    passed = (np.allclose(np.abs(rows[1]['angles_deg']), 90, atol=1)
              and np.allclose(rows[-1]['angles_deg'], 0, atol=1))
    report = dict(mechanism_test_passed=bool(passed), robot_skill_validated=False,
                  playback_speed=1., metadata=metadata, phases=rows)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'trace.json').write_text(json.dumps(trace))
    print(json.dumps(report), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
