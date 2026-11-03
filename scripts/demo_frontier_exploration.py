"""Closed-loop exploration qualification. Simulator state stays in the bridge.

Scene metadata is used for setup only; neither SLAM nor exploration receives it.
No saved trajectory, environment poses, room graph or collision map drives policy.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation import ObservedNavigationMap, ExplorationSkill, ExplorationConfig
from cross_episode_sim.robot.observations import depth_rgb


from cross_episode_sim.control.onboard_base import OnboardBaseActuator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    model=mujoco.MjModel.from_binary_path(str(a.source/'action_recording/model.mjb'));data=mujoco.MjData(model)
    with np.load(a.source/'action_recording/initial_state.npz') as x:mujoco.mj_setState(model,data,x['state'],int(x['state_type']))
    # Requested all-open condition is set before evaluation, never passed to policy.
    doors=[]
    for jid in range(model.njnt):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid) or ''
        if name.startswith('room_door_') and '_hinge_' in name:
            angle=float(model.jnt_range[jid,1] if name.endswith('left') else model.jnt_range[jid,0])
            data.qpos[model.jnt_qposadr[jid]]=angle;data.qvel[model.jnt_dofadr[jid]]=0
            doors.append(dict(name=name,angle=angle))
    mujoco.mj_forward(model,data)
    bridge=OnboardBaseActuator(model,data)
    mapper=RGBDMapper();grid=ObservedNavigationMap();skill=ExplorationSkill(ExplorationConfig(max_seconds=a.seconds))
    rows=[];poses=[];start=float(data.time);wall=time.monotonic();decision=None;layers=None
    with OnboardSensorInterface(model,data) as sensors, imageio.get_writer(str(a.output/'exploration.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
        for tick in tqdm(range(round(a.seconds*50)),desc='Onboard exploration'):
            if tick%5==0:
                packet=sensors.capture();result=mapper.update(packet.cameras['navigation'],packet.timestamp)
                grid.update(packet.cameras['navigation'],result);layers=grid.layers()
            decision=skill.step(layers,result,float(data.time))
            if tick%5==0:
                row=dict(time=float(data.time-start),status=decision.status,tracking=result.status,accepted=result.accepted,forward_mps=decision.forward_mps,yaw_rps=decision.yaw_rps,strict_area_m2=float(layers.strict_free.sum()*.01))
                rows.append(row);poses.append(result.T_map_base if result.accepted else np.full((4,4),np.nan))
                panels=[np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()]
                panel=Image.fromarray(np.concatenate(panels,axis=0));draw=ImageDraw.Draw(panel)
                draw.rectangle((0,0,1280,30),fill='white');draw.text((8,8),f'{row["time"]:.1f}s | {decision.status} | {result.status}',fill='black')
                video.append_data(np.asarray(panel))
                if tick==0:panel.save(a.output/'initial_onboard.png')
                if tick%50==0:tqdm.write(json.dumps(row))
            bridge.step(decision.forward_mps,decision.yaw_rps)
    report=grid.save(a.output);mapper.save(a.output/'slam')
    report.update(source=str(a.source),rooms=len(json.loads((a.source/'base_layout.json').read_text())['rooms']),doors_initialized_open=doors,
        wall_seconds=time.monotonic()-wall,simulated_seconds=float(data.time-start),last_status=decision.status,plans=skill.plans,
        nonzero_command_frames=sum(r['forward_mps']!=0 or r['yaw_rps']!=0 for r in rows),
        exploration_success=False,note='A budget stop or safe startup stop is not successful exploration. No scene metadata supplied to policy.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'decisions.json').write_text(json.dumps(rows,indent=2));np.save(a.output/'estimated_poses.npy',poses)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
