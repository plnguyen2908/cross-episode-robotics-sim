"""Debug a saved grasp with its recorded sensor-map support, not fixture truth.

This is a component replay, not a fresh exploration qualification. The complete
saved observed map is memory supplied to the replay, including later observations.
Use demo_room_coverage_exploration.py for causal, fresh end-to-end qualification.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.control.contact_retreat import transform_support_region


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--door-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contact-motion-profile',choices=['feedback','fast'],default='fast')
    parser.add_argument('--handle-then-push',action='store_true')
    args=parser.parse_args()
    grasp=args.source/'doors'/f'door_{args.door_index:03d}'/'grasp'
    args.output.mkdir(parents=True,exist_ok=False)
    # Simulator access here is private reset only.
    model=mujoco.MjModel.from_binary_path(str(grasp/'state_recording/model.mjb'))
    data=mujoco.MjData(model)
    with h5py.File(grasp/'state_recording/states.h5') as states:
        mujoco.mj_setState(model,data,states['states'][-1],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    saved=np.load(args.source/'navigation_map.npz')
    config=json.loads((args.source/'summary.json').read_text())['config']
    resolution=float(saved['resolution'])
    rr,cc=np.where(saved['floor_views']>=config['minimum_floor_views'])
    xy=saved['origin_xy']+np.c_[cc,rr]*resolution
    floor=unary_union([box(x,y,x+resolution,y+resolution) for x,y in xy])
    poses=np.load(args.source/'slam/estimated_base_poses.npy')
    times=np.load(args.source/'slam/estimated_timestamps.npy')
    index=np.searchsorted(times,data.time,side='right')-1
    if index<0:raise ValueError('No estimated pose precedes the saved grasp')
    runtime=EpisodeRuntime(args.output);runtime.bind(model,data)
    runtime.contact_support_region=transform_support_region(floor,np.linalg.inv(poses[index]))
    from cross_episode_sim.control.door_open_command import door_open_command
    command=door_open_command(grasp,args.output/'opening',contact_motion_profile=args.contact_motion_profile,
                              handle_then_push=args.handle_then_push)
    (args.output/'replay_manifest.json').write_text(json.dumps(dict(
        source=str(args.source),door_index=args.door_index,
        qualification='saved-grasp component with recorded observed-map memory',
        estimated_anchor_timestamp=float(times[index])),indent=2))
    started=time.perf_counter()
    with runtime.activate(),(args.output/'open.log').open('w') as log:
        result=runtime.run(command,log)
    (args.output/'execution_timing.json').write_text(json.dumps(dict(
        wall_seconds=time.perf_counter()-started,profile=args.contact_motion_profile,
        returncode=result.returncode),indent=2))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
