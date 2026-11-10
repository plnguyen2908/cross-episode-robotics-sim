"""Offline component benchmark on recorded RGB-D states, not a success replay."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np,h5py,mujoco,torch
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig

p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start-seconds',type=float,default=123);p.add_argument('--frames',type=int,default=60);p.add_argument('--warmup',type=int,default=10);p.add_argument('--map-run',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
meta=json.loads((a.recording/'metadata.json').read_text());m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
results={}
def sync():
 if torch.cuda.is_available():torch.cuda.synchronize()
def timed(fn):
 sync();t=time.perf_counter();v=fn();sync();return v,time.perf_counter()-t
for mode in ('previous_10hz_both','optimized_2hz_nav'):
 timings=defaultdict(list);counts=defaultdict(int);masks=[]
 mapper,startup=timed(lambda:create_mapper(fuse_dense=False));g=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.05,pixel_stride=4))
 if a.map_run:
  saved=np.load(a.map_run/'wall_evidence.npz');g.evidence=saved['evidence'].copy();g.floor_views=saved['floor'].copy();g.origin=saved['origin'].copy()
  rows=[json.loads(s) for s in (a.map_run/'progress.jsonl').read_text().splitlines()]
  row=min((r for r in rows if 'estimated_base_xy' in r),key=lambda r:abs(r['simulated_seconds']-a.start_seconds))
  yaw=row['estimated_base_yaw'];reference=np.eye(4);reference[:2,:2]=[[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]];reference[:2,3]=row['estimated_base_xy']
  center=np.array(json.loads((a.map_run/'report.json').read_text())['opening_results'][0]['doorway_center_map'])[:2]
  g.fusion_region=(center,2.)

 with h5py.File(a.recording/'states.h5') as f,OnboardSensorInterface(m,d) as sensors:
  original=sensors._robot_mask
  def mask(*args,**kwargs):
   result,dt=timed(lambda:original(*args,**kwargs));masks.append(dt);return result
  sensors._robot_mask=mask
  first=int(np.searchsorted(f['times'][:],float(f['times'][0])+a.start_seconds))
  accepted=0;last_fused=None;last_pose=None;anchor=None
  for j in range(a.warmup+a.frames):
   i=first+j
   mujoco.mj_setState(m,d,f['states'][i],meta['state_spec']);mujoco.mj_forward(m,d)
   masks.clear();packet,capture=timed(sensors.capture)
   tracking,localization=timed(lambda:mapper.update(packet.cameras['navigation'],packet.timestamp))
   if a.map_run:
    from types import SimpleNamespace
    if anchor is None:
     anchor=reference@np.linalg.inv(tracking.T_map_base)
     g.refresh_opened_portal(center,timestamp=packet.timestamp)
    tracking=SimpleNamespace(accepted=tracking.accepted,timestamp=tracking.timestamp,T_map_base=anchor@tracking.T_map_base)
   pose=tracking.T_map_base
   moved=last_pose is None or np.linalg.norm(pose[:2,3]-last_pose[:2,3])>=.2 or abs(np.arctan2((last_pose[:3,:3].T@pose[:3,:3])[1,0],(last_pose[:3,:3].T@pose[:3,:3])[0,0]))>=.35
   fuse=mode=='previous_10hz_both' or last_fused is None or packet.timestamp-last_fused>=.5 or moved
   nav=wrist=0.
   if fuse:
    _,nav=timed(lambda:g.update(packet.cameras['navigation'],tracking));last_fused=packet.timestamp;last_pose=pose.copy()
   if tracking.accepted and mode=='previous_10hz_both':_,wrist=timed(lambda:g.update_support(packet.cameras['wrist'],tracking,geometry=True))
   _,physics=timed(lambda:mujoco.mj_step(m,d,nstep=round(.1/m.opt.timestep)))
   if j>=a.warmup:
    accepted+=bool(tracking.accepted)
    for key,value in [('camera_rgbd_render_and_read',capture-sum(masks)),('robot_mask',sum(masks)),('localization',localization),('navigation_map_fusion',nav),('wrist_map_fusion',wrist),('physics_100ms',physics)]:timings[key].append(value)
    counts['nav_fusions']+=int(fuse);counts['wrist_fusions']+=int(mode=='previous_10hz_both' and tracking.accepted)
   if j%10==0:print(mode,j,flush=True)
 results[mode]=dict(startup_mapper_seconds=startup,accepted_frames=accepted,frames=a.frames,represented_sim_seconds=a.frames/meta['fps'],counts=dict(counts),timings={k:dict(total_seconds=sum(v),mean_ms=1000*np.mean(v),p95_ms=1000*np.percentile(v,95)) for k,v in timings.items()})
 (a.output/'components.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
