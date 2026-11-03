"""Physical local-follower regression from a saved TARE failure.

Uses recorded sensor-derived map/goal, not scene geometry, as policy input.
This tests a local recovery only; it is not a fresh whole-house exploration.
"""
import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm import tqdm

from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.house_exploration import HouseExplorer, HouseExplorationConfig
from cross_episode_sim.navigation.observed_map import MapConfig, ObservedNavigationMap
from cross_episode_sim.navigation.tare_exploration import TareExplorer
from cross_episode_sim.robot.observations import depth_rgb
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from scripts.demo_local_scan import setup, ContactAudit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('failed-run','source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--scan-first',action='store_true',help='Rescan when the old clipped map discarded needed evidence')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((a.failed_run/'report.json').read_text())
    rows=[json.loads(line) for line in (a.failed_run/'progress.jsonl').read_text().splitlines()]
    goal=next(row['goal'] for row in reversed(rows) if row.get('goal') is not None)
    state=np.load(a.failed_run/'terminal_state.npz')
    observation=np.load(a.failed_run/'terminal_observation.npz')
    offset=observation['estimated_pose']
    if not np.isfinite(offset).all():raise ValueError('Checkpoint has no accepted visual pose')
    responses=a.failed_run/'tare/planner_responses.jsonl'
    if responses.exists():
        with responses.open() as stream:upstream=json.loads(deque(stream,maxlen=1)[0])
        sensor_offset=(offset@observation['T_base_optical'])[:2,3]-offset[:2,3]
        goal=(np.asarray(upstream['waypoint'][:2])-sensor_offset).tolist()
    class RecordedGoal:
        def update(self,camera,tracking):
            pose=tracking.T_map_base
            sensor_offset=(pose@camera.T_base_optical)[:2,3]-pose[:2,3]
            return dict(waypoint=(np.asarray(goal)+sensor_offset).tolist()+[1.7],sequence=1,finished=False)

    model,data,_=setup(a.source)
    spec=int(state['state_spec'])
    if len(state['state'])!=mujoco.mj_stateSize(model,spec):raise ValueError('Checkpoint/model mismatch')
    mujoco.mj_setState(model,data,state['state'],spec);mujoco.mj_forward(model,data)
    bridge=OnboardBaseActuator(model,data);audit=ContactAudit(model)
    grid=ObservedNavigationMap(MapConfig(**previous['config']))
    saved=np.load(a.failed_run/'navigation_map.npz')
    grid.evidence[:]=saved['evidence'];grid.floor_views[:]=saved['floor_views']
    policy=TareExplorer(grid,planner=RecordedGoal(),config=HouseExplorationConfig(max_seconds=a.seconds))
    policy.support[:]=np.load(a.failed_run/'ego_support_prior.npy')
    policy.phase='scan' if a.scan_first else 'travel';policy.response=dict(waypoint=list(goal)+[0.],sequence=1)
    # The goal is already a recorded base goal; no sensor-offset conversion.
    policy.sensor_offset[:]=0.
    slam=RGBDMapper();start=float(data.time);wall=time.monotonic();records=[];reached=False
    (a.output/'inputs.json').write_text(json.dumps(dict(kind='local_follower_regression_not_whole_house',
        source=str(a.source),failed_run=str(a.failed_run),recorded_base_goal=goal,scan_first=a.scan_first),indent=2))
    with OnboardSensorInterface(model,data) as sensors, imageio.get_writer(str(a.output/'recovery.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video, open(a.output/'progress.jsonl','w',buffering=1) as progress:
        for tick in tqdm(range(round((a.seconds+1)*50)),desc='Local recovery physics'):
            if tick%5==0:
                packet=sensors.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
                if tracking.accepted:tracking=replace(tracking,T_map_base=offset@tracking.T_map_base)
                grid.update(packet.cameras['navigation'],tracking)
            command=policy.step(packet,tracking,float(data.time))
            if command['status'] in ('arrived_frontier','tare_waypoint_reached'):
                reached=True;policy.terminal='recorded_goal_reached'
            if tick%5==0:
                records.append(dict(time=float(data.time-start),**command,estimated_position=tracking.T_map_base[:2,3].tolist() if tracking.accepted else None,map_extent_m=grid.config.extent_m))
                progress.write(json.dumps(records[-1])+'\n')
                if len(records)==1 or records[-1]['status']!=records[-2]['status']:tqdm.write(json.dumps(records[-1]))
                video.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0))
            bridge.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
            if policy.terminal and bridge.stationary():break
    report=dict(kind='local_follower_regression_not_whole_house',goal_reached=reached,
        physical_success=reached and not audit.events,status=policy.terminal,
        simulated_seconds=float(data.time-start),wall_seconds=time.monotonic()-wall,
        external_contact_events=list(audit.events.values()),events=policy.events,
        requested_base_goal=goal,planned_base_goal=policy.goal.tolist() if policy.goal is not None else None,
        final_estimated_position=tracking.T_map_base[:2,3].tolist() if tracking.accepted else None,
        waypoint_tolerance=policy.config.waypoint_tolerance)
    grid.save(a.output/'sensor_map')
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    (a.output/'progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
