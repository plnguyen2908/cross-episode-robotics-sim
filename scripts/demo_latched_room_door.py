"""Render a force-driven latch mechanism test (not a robot skill rollout)."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from cross_episode_sim.tasks.latched_room_door import install_latched_leaf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output is None:Path('runs').mkdir(exist_ok=True)
    out=args.output or Path(tempfile.mkdtemp(prefix='latched_room_door_',dir='runs'))
    if args.output:out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT='+str(out.resolve()),flush=True)
    spec=mujoco.MjSpec();spec.compiler.degree=False;spec.option.timestep=.002
    spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth=640;spec.visual.global_.offheight=480
    spec.worldbody.add_light(pos=[1,-2,4],dir=[0,0,-1],diffuse=[.8,.8,.8])
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3,3,.1],rgba=[.22,.25,.28,1])
    for x in (-.055,.975):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x,0,1.08],size=[.02,.08,1.08],rgba=[.7,.72,.74,1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[.46,0,2.15],size=[.535,.08,.02],rgba=[.7,.72,.74,1])
    metadata=install_latched_leaf(spec,'door',[0,0,.02],0,.9,2.1)
    model=spec.compile();data=mujoco.MjData(model)
    spec.to_file(str(out/'scene.xml'))
    cameras=[]
    for look,distance,azimuth,elevation in [([.45,0,1.05],3.4,120,-15),([.88,.01,.97],.65,65,-12)]:
        camera=mujoco.MjvCamera();camera.lookat[:]=look
        camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
        cameras.append(camera)
    phases=[('Pull without turning handle',2.,5.,0.),
            ('Release pressure on latch',.5,-5.,0.),
            ('Depress lever handle',1.,-5.,2.5),
            ('Open with handle depressed',3.,5.,2.5),
            ('Close with handle depressed',4.,-5.,2.5),
            ('Release handle to latch',2.,-2.,0.),
            ('Pull again: latch holds',2.,5.,0.)]
    rows=[];samples=[];next_frame=0.;fps=25
    with mujoco.Renderer(model,480,640) as renderer,imageio.get_writer(
            str(out/'mechanism_test.mp4'),fps=fps,codec='libx264',quality=8) as writer:
        for label,seconds,hinge,handle in tqdm(phases,desc='Latch mechanism'):
            data.qfrc_applied[:]=[hinge,handle]
            for _ in range(round(seconds/model.opt.timestep)):
                mujoco.mj_step(model,data)
                if data.time+1e-9 < next_frame:continue
                next_frame+=1/fps
                samples.append(dict(time=float(data.time),phase=label,qpos=data.qpos.tolist(),
                                    external_torque_nm=[hinge,handle]))
                frames=[]
                for camera in cameras:
                    renderer.update_scene(data,camera=camera)
                    frames.append(renderer.render().copy())
                frame=Image.fromarray(np.concatenate(frames,axis=1));draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,1280,46),fill='black')
                draw.text((8,6),'MECHANISM TEST - external torques, NO ROBOT',fill='white')
                draw.text((8,27),f'{label} | door {np.degrees(data.qpos[0]):.1f} deg | handle {np.degrees(data.qpos[1]):.1f} deg',fill='white')
                writer.append_data(np.asarray(frame))
            rows.append(dict(phase=label,door_deg=float(np.degrees(data.qpos[0])),
                             handle_deg=float(np.degrees(data.qpos[1]))))
    passed=(abs(rows[0]['door_deg'])<2 and rows[2]['handle_deg']>50
            and rows[3]['door_deg']>85 and abs(rows[4]['door_deg'])<1
            and abs(rows[5]['handle_deg'])<1 and abs(rows[6]['door_deg'])<2)
    report=dict(mechanism_test_passed=passed,robot_skill_validated=False,
                playback_speed=1.,metadata=metadata,phases=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'trace.json').write_text(json.dumps(samples))
    print(json.dumps(report),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
