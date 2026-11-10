"""Diagnostic replay: compare RGB-D tracking on an actual recorded crossing.

Recorded physics supplies camera observations. Ground truth is used only for
the printed localization error, never passed to either tracker.
"""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import h5py,mujoco,numpy as np
from PIL import Image
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,default=20.)
p.add_argument('--compare-exclusion',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
names=('keyframe','wide_exclusion') if a.compare_exclusion else ('keyframe','precision')
trackers={name:create_mapper(fuse_dense=False) for name in names}
reference=None;reference_pose=None;reference_time=None;rows=[];wall=time.monotonic()
with h5py.File(a.recording/'states.h5') as f,OnboardSensorInterface(m,d) as sensors:
 times=f['times'][:];ids=np.flatnonzero((times>=a.start)&(times<=a.start+a.duration))
 initial=None
 for k,i in enumerate(ids):
  mujoco.mj_setState(m,d,f['states'][i],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
  packet=sensors.capture(camera_keys=('navigation',),include_robot_mask=False)
  body=d.body('robot_0/base');truth=np.eye(4);truth[:3,:3]=body.xmat.reshape(3,3);truth[:3,3]=body.xpos
  if initial is None:initial=truth.copy()
  truth=np.linalg.inv(initial)@truth;row=dict(timestamp=packet.timestamp,truth=truth.tolist())
  for name,mapper in trackers.items():
   camera=packet.cameras['navigation']
   if name=='wide_exclusion':
    mask=np.zeros(camera.depth_m.shape,bool);h,w=mask.shape
    mask[int(h*.25):,int(w*.28):int(w*.72)]=True
    camera=replace(camera,robot_mask=mask)
   result=mapper.update(camera,packet.timestamp)
   row[name]=dict(accepted=result.accepted,status=result.status,position=None if result.T_map_base is None else result.T_map_base[:2,3].tolist(),error_m=None if result.T_map_base is None else float(np.linalg.norm(result.T_map_base[:2,3]-truth[:2,3])))
   if name=='precision' and result.accepted:
    pose=result.T_map_base
    refresh=reference is None or packet.timestamp-reference_time>=1. or abs(np.arctan2((reference_pose[:3,:3].T@pose[:3,:3])[1,0],(reference_pose[:3,:3].T@pose[:3,:3])[0,0]))>=.2
    if refresh:
     if reference is not None:reference.__exit__(None,None,None)
     reference=mapper.fixed_reference();reference.__enter__();reference_time=packet.timestamp;reference_pose=pose.copy()
  rows.append(row)
  if k%50==0:print(json.dumps({key:value for key,value in row.items() if key!='truth'}),flush=True)
  if k in (0,len(ids)-1):Image.fromarray(packet.cameras['navigation'].rgb).save(a.output/f'view_{k}.png')
 if reference is not None:reference.__exit__(None,None,None)
summary={name:dict(accepted=sum(r[name]['accepted'] for r in rows),maximum_error_m=max(r[name]['error_m'] for r in rows if r[name]['error_m'] is not None),final=rows[-1][name]) for name in trackers}
summary.update(frames=len(rows),wall_seconds=time.monotonic()-wall)
(a.output/'trajectory.json').write_text(json.dumps(rows));(a.output/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
