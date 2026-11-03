"""Private test initialization only: translate robot + held object, then validate."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--distance',type=float,default=1.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
roots=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
lookup=contact_root_lookup(m.body_parentid,roots);pads={m.body(n).id for n in ('robot_0/gripper/left_pad','robot_0/gripper/right_pad')}
def held_roots():
 held=set()
 for ct in d.contact:
  b1,b2=map(int,m.geom_bodyid[[ct.geom1,ct.geom2]])
  if b1 in pads and lookup[b2]>=0:held.add(int(lookup[b2]))
  if b2 in pads and lookup[b1]>=0:held.add(int(lookup[b1]))
 return held
held=held_roots()
if len(held)!=1:raise RuntimeError('Private reset requires one held object')
yaw=float(d.joint('robot_0/base_theta').qpos[0]);delta=-a.distance*np.array([np.cos(yaw),np.sin(yaw),0.])
d.joint('robot_0/base_x').qpos[0]+=delta[0];d.joint('robot_0/base_y').qpos[0]+=delta[1]
for root in held:
 j=int(m.body_jntadr[root]);adr=int(m.jnt_qposadr[j]);d.qpos[adr:adr+3]+=delta
d.qvel[:]=0;mujoco.mj_forward(m,d)
c=MolmoSpacesController(m,d);hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')};hold['gripper']=np.array([255.])
robot=np.array([(m.body(i).name or '').startswith('robot_0/') for i in range(m.nbody)])
def forbidden_contacts():
 result=[]
 for ct in d.contact:
  if ct.dist>=-.001:continue
  g1,g2=int(ct.geom1),int(ct.geom2);b1,b2=map(int,m.geom_bodyid[[g1,g2]])
  if robot[b1] and robot[b2]:continue
  if (robot[b1] and lookup[b2] in held) or (robot[b2] and lookup[b1] in held):continue
  if not(robot[b1] or robot[b2] or lookup[b1] in held or lookup[b2] in held):continue
  names=[m.geom(g).name or '' for g in (g1,g2)]
  if abs(ct.frame[2])>.9 and any('floor' in n.lower() or 'ground' in n.lower() for n in names):continue
  result.append(dict(geoms=names,distance=float(ct.dist)))
 return result
bad=forbidden_contacts()
if not bad:
 for _ in range(25):
  c.step(hold)
  for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
  bad+=forbidden_contacts()
success=not bad and bool(held&held_roots())
report=dict(initialization_valid=success,private_held_root_ids=list(held),backward_offset_m=a.distance,forbidden_contacts=bad[:20],scope='Private far-start initialization; no policy receives object truth or offset. No demonstration success claim.')
(a.output/'report.json').write_text(json.dumps(report,indent=2))
if not success:raise RuntimeError('Far-start physical qualification failed: '+str(bad[:3]))
cp=a.output/'private_checkpoint';cp.mkdir();mujoco.mj_saveModel(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(cp/'state.npy',state)
print(json.dumps(report),flush=True)
