"""Recorded-depth diagnostic; simulator truth is used only to score pose errors."""
import argparse,json
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.frozen_map_localization import FrozenMapLocalizer
from cross_episode_sim.navigation.success_graph_transit import se2

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--raw-session',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
x=np.load(a.run/'door_exploration/wall_evidence.npz');g=ObservedNavigationMap(MapConfig(resolution=float(x['resolution']),height_resolution=float(x['resolution'])))
g.origin=x['origin'];g.evidence=x['evidence'];g.floor_views=x['floor']
localizer=FrozenMapLocalizer(g)
m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
def physical_pose():
    b=m.body('robot_0/base').id;T=np.eye(4);T[:3,:3]=d.xmat[b].reshape(3,3);T[:3,3]=d.xpos[b];return T
with h5py.File(a.run/'door_exploration/state_recording/states.h5') as f:
    mujoco.mj_setState(m,d,f['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d);worldmap=physical_pose()
observed=[json.loads(s) for s in (a.recording/'diagnostics.jsonl').read_text().splitlines()]
def matrix(pose):
    T=np.eye(4);c,s=np.cos(pose[2]),np.sin(pose[2]);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=pose[:2];return T
if a.raw_session:
    old_alignment=np.eye(4);events=iter(json.loads((a.run/'map_localization.json').read_text()));event=next(events,None)
    for row in observed:
        T=np.array(row['pose'])
        while event is not None and event['timestamp']<=row['timestamp']+1e-6:
            if event['accepted']:
                after=np.array(min(observed,key=lambda r:abs(r['timestamp']-event['timestamp']))['pose'])
                before=matrix(se2(after)-np.array(event['correction']))
                old_alignment=after@np.linalg.inv(before)@old_alignment
            event=next(events,None)
        row['pose']=(np.linalg.inv(old_alignment)@T).tolist()
results=[]
alignment=np.eye(4)
with h5py.File(a.recording/'states.h5') as f,OnboardSensorInterface(m,d) as sensors:
    times=f['times'][:]
    for i in range(0,len(times),5 if a.raw_session else 10):
        mujoco.mj_setState(m,d,f['states'][i],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        row=min(observed,key=lambda r:abs(r['timestamp']-times[i]));raw=np.array(row['pose']);estimate=se2(alignment@raw)
        packet=sensors.capture(include_robot_mask=False)
        corrected,detail=localizer.correct(packet.cameras['navigation'],estimate)
        if a.raw_session and detail['accepted']:alignment=matrix(corrected)@np.linalg.inv(raw)
        # Score after estimation; truth is never passed to FrozenMapLocalizer.
        truth=se2(np.linalg.inv(worldmap)@physical_pose())
        result=dict(time=float(times[i]-times[0]),original=estimate.tolist(),corrected=corrected.tolist(),truth=truth.tolist(),
                    original_error=float(np.linalg.norm(estimate[:2]-truth[:2])),corrected_error=float(np.linalg.norm(corrected[:2]-truth[:2])),**detail)
        results.append(result);print(json.dumps(result),flush=True)
(a.output/'results.json').write_text(json.dumps(results,indent=2))
summary={key:float(np.mean([r[key] for r in results])) for key in ['original_error','corrected_error','wall_seconds']}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
