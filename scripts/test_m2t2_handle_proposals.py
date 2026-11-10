"""Sensor-only handle proposal smoke test. No physical grasp claim."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from cross_episode_sim.sensors.onboard import OnboardSensorInterface
from cross_episode_sim.perception.molmo_pointing import MolmoPointer
from cross_episode_sim.robot.observations import depth_rgb
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--crop-radius',type=float,default=.15);p.add_argument('--kind',choices=['door','cabinet','drawer','fridge','microwave'],required=True);p.add_argument('--target-description',help='Optional semantic visual target description, never coordinates');p.add_argument('--reach-occurrence',type=int,help='Harness-only index among reach snapshots; never passed to perception');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
meta=json.loads((a.source/'metadata.json').read_text());rec=np.load(a.source/'actions.npz');stages=rec['stage'];reach=meta['stages'].index('reach cabinet handle')
starts=np.flatnonzero((stages==reach)&np.r_[True,stages[1:]!=stages[:-1]])
chosen=None
for i in starts:
 future=[meta['stages'][int(x)] for x in stages[i:]]
 # The first pull after this reach identifies the fixture without supplying its pose.
 next_pull=next((x for x in future if x in ['pull drawer open','pull cabinet open','push room door by handle']),None)
 expected={'door':'push room door by handle','cabinet':'pull cabinet open','drawer':'pull drawer open','fridge':'pull cabinet open','microwave':'pull cabinet open'}[a.kind]
 if next_pull==expected:chosen=int(i);break
if a.reach_occurrence is not None:
 if not 0<=a.reach_occurrence<len(starts):raise ValueError('Reach occurrence outside recording')
 chosen=int(starts[a.reach_occurrence])
if chosen is None:raise RuntimeError('No pre-reach snapshot for requested fixture')
m=mujoco.MjModel.from_binary_path(str(a.source/'model.mjb'));d=mujoco.MjData(m);initial=np.load(a.source/'initial_state.npz');mujoco.mj_setState(m,d,initial['state'],int(initial['state_type']));d.qpos[:]=rec['qpos'][chosen];d.qvel[:]=0;d.ctrl[:]=rec['ctrl'][chosen];mujoco.mj_forward(m,d)
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir(exist_ok=True)
mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None)
spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state)
pointer=MolmoPointer();events=[];cloud=[];colors=[];targets=[];panels=[]
with OnboardSensorInterface(m,d) as sensors:
 packet=sensors.capture()
 for name,c in packet.cameras.items():
  panels.append(np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1))
  result=pointer.point(Image.fromarray(c.rgb),a.target_description or {'door':'room door handle','cabinet':'cabinet door handle','drawer':'drawer handle','fridge':'vertical handle of the upper hinged refrigerator door, not the horizontal freezer drawer handle','microwave':'microwave door handle'}[a.kind]);events.append(dict(camera=name,**result));print(name,result,flush=True)
  valid=c.valid_depth.copy()
  if c.robot_mask is not None:valid &= ~c.robot_mask
  v,u=np.where(valid);z=c.depth_m[v,u];pts=np.c_[(u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z];pts=pts@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3];cloud.append(pts);colors.append(c.rgb[v,u])
  for x,y in result['points']:
   u,v=int(x),int(y)
   if 0<=v<valid.shape[0] and 0<=u<valid.shape[1] and valid[v,u]:
    z=c.depth_m[v,u];targets.append((np.array([(u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z])@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3]).tolist())
Image.fromarray(np.concatenate(panels)).save(a.output/'observation.png')
report=dict(kind=a.kind,reset_frame=chosen,prepositioned=True,crop_radius=a.crop_radius,physical_grasp_tested=False,pointing=events,targets=targets)
(a.output/'report.json').write_text(json.dumps(report,indent=2))
if not targets:print('No visible handle localized',flush=True);sys.exit(0)
pts=np.concatenate(cloud);rgb=np.concatenate(colors)
np.savez(a.output/'observed_world.npz',points=pts,rgb=rgb)
if not 0<a.crop_radius<=.7:raise ValueError('Crop radius must be in (0, .7]')
for i,target in enumerate(targets):
 target=np.array(target);keep=np.linalg.norm(pts-target,axis=1)<a.crop_radius
 np.savez(a.output/f'input_{i}.npz',points=pts[keep],rgb=rgb[keep],target=target)
 subprocess.run([sys.executable,'scripts/m2t2_worker.py','--input',str(a.output/f'input_{i}.npz'),'--output',str(a.output/f'grasps_{i}.json'),'--checkpoint','/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth'],check=True)
