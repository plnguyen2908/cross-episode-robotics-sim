"""Observed bowl-rim grasp experiment; all object truth is private final scoring."""
import argparse,json,math
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.observed_arm import ObservedArmPlanner,robot_calibration
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import ContactAudit
from curobo._src.geom.types import Cuboid,SceneCfg
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
r=json.loads((a.source/'report.json').read_text());T=np.load(a.source/'slam/estimated_base_poses.npy')[-1];target=T[:3,:3].T@(np.array(r['selected_object']['position'])-T[:3,3])
g=ObservedNavigationMap(MapConfig(**json.loads((a.source/'summary.json').read_text())['config']));saved=np.load(a.source/'navigation_map.npz');g.origin=saved['origin_xy'];g.evidence=saved['evidence'];g.floor_views=saved['floor_views']
controller=MolmoSpacesController(m,d);hold={n:controller.robot_view.get_move_group(n).noop_ctrl.copy() for n in ('base','arm','gripper')};hold['gripper']=np.array([0.]);geo=RobotGeometry(m);audit=ContactAudit(m);start=d.time;steps=0;phase='initial';events=[]
# Diagnostic body snapshots remain private to the final scorer.
bodies=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and 'bowl' in (m.body(int(m.jnt_bodyid[j])).name or '').lower()];initial={b:d.xpos[b].copy() for b in bodies}
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'grasp.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
 def execute(qcmd,grip=0.):
  global steps
  action={k:v.copy() for k,v in hold.items()};action['arm']=qcmd;action['gripper']=np.array([grip]);controller.step(action)
  for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d);audit.observe(d)
  if steps%5==0:
   packet=sensors.capture();frame=Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0));draw=ImageDraw.Draw(frame);draw.rectangle((0,0,frame.width,24),fill='white');draw.text((4,4),f'Observed rim grasp | {phase} | {d.time-start:.2f}s',fill='black');video.append_data(np.array(frame));frame.save(a.output/'latest_frame.png')
  steps+=1
 def measured():return controller.robot_view.get_move_group('arm').joint_pos.copy()
 def move(planner,goal,R,grip=0.):
  try:trajectory=planner.plan(measured(),goal,R)
  except RuntimeError:
   from scipy.optimize import least_squares
   rng=np.random.default_rng(0);trajectory=None;lo,hi=geo.limits[:,0]+.001,geo.limits[:,1]-.001
   def residual(q):
    tcp=geo.fk(q)[geo.tcp_body]@geo.tcp_local
    return np.r_[tcp[:3,3]-goal,.2*Rotation.from_matrix(R@tcp[:3,:3].T).as_rotvec()]
   statistics=[]
   for seed in [measured()]+[rng.uniform(lo,hi) for _ in range(24)]:
    solution=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=150)
    statistics.append([float(np.linalg.norm(residual(solution.x)[:3])),float(np.linalg.norm(residual(solution.x)[3:])),planner.arm.self_clearance(solution.x),planner.arm.world_clearance(solution.x,planner.boxes)])
    if np.linalg.norm(residual(solution.x)[:3])>.003 or np.linalg.norm(residual(solution.x)[3:])>.01:continue
    if planner.arm.self_clearance(solution.x)<.005 or planner.arm.world_clearance(solution.x,planner.boxes)<.005:continue
    try:trajectory=planner.arm.plan_joints(measured(),solution.x);break
    except RuntimeError:continue
   print('IK_DIAGNOSTICS',statistics,flush=True)
   if trajectory is None:raise RuntimeError('No collision-clear joint plan for grasp pose')
  times=np.arange(len(trajectory))*planner.dt
  for t in np.arange(0,times[-1]+.019,.02):execute(np.array([np.interp(t,times,trajectory[:,j]) for j in range(7)]),grip)
  stable=0
  for _ in range(100):
   execute(trajectory[-1],grip);error=np.max(np.abs(measured()-trajectory[-1]));vel=np.max(np.abs([d.joint(n).qvel[0] for n in geo.joint_names]));stable=stable+1 if error<.015 and vel<.04 else 0
   if stable>=5:break
  if stable<5:raise RuntimeError('Arm failed measured settling')
  events.append(dict(phase=phase,time=float(d.time-start),goal=np.asarray(goal).tolist()))
 try:
  packet=sensors.capture();points=[]
  for c in packet.cameras.values():
   valid=c.valid_depth.copy()

   v,u=np.where(valid);z=c.depth_m[v,u];xyz=np.column_stack(((u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z));points.append(xyz@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3])
  points=np.concatenate(points)
  keep=np.ones(len(points),bool)
  for box in geo.boxes(measured()):keep &= ~np.all(np.abs((points-box.center)@box.axes)<=box.half+.015,axis=1)
  points=points[keep];near=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.13)&(abs(points[:,2]-target[2])<.15)]
  if len(near)<200:raise RuntimeError('Insufficient observed bowl points')
  rim_z=float(np.quantile(near[:,2],.99));rim=near[near[:,2]>rim_z-.006]
  fit=np.linalg.lstsq(np.c_[2*rim[:,:2],np.ones(len(rim))],(rim[:,:2]**2).sum(1),rcond=None)[0];center=fit[:2];radius=float(np.sqrt(fit[2]+sum(center**2)))
  if not .04<radius<.14:raise RuntimeError('Unverified bowl rim fit')
  bins,counts=np.unique(np.round(near[:,2]/.003),return_counts=True);support=float(bins[counts.argmax()]*.003)
  radial=-center/np.linalg.norm(center);contact_xy=center+radius*radial
  grasp=np.r_[contact_xy,rim_z-.015];R=Rotation.from_euler('z',math.atan2(radial[1],radial[0])-math.pi/2).as_matrix()@np.diag([1.,-1.,-1.])
  (a.output/'observed_grasp.json').write_text(json.dumps(dict(center=center.tolist(),radius=radius,rim_z=rim_z,support=support,grasp=grasp.tolist()),indent=2))
  planner=ObservedArmPlanner(robot_calibration(m,d,a.output/'robot'),g,T,extra_obstacle_capacity=4096)
  print('START_CLEARANCE',planner.arm.self_clearance(measured()),planner.arm.world_clearance(measured(),planner.boxes),flush=True)
  # Intentional contact with the inferred bowl volume only. Keep counter and
  # all occupancy above the measured rim; simulator collisions remain enabled.
  boxes=[box for box in planner.boxes if not (np.linalg.norm(np.array(box.pose[:2])-center)<.30 and .78<float(box.pose[2])<1.55)]
  local=points[(np.linalg.norm(points[:,:2]-center,axis=1)<.38)&(points[:,2]>.70)&(points[:,2]<1.65)]
  allowed=(np.linalg.norm(local[:,:2]-center,axis=1)<radius+.015)&(local[:,2]>support+.006)&(local[:,2]<rim_z+.01)
  local=local[~allowed]
  voxels=np.unique(np.floor(local/.02).astype(int),axis=0)
  for index,voxel in enumerate(voxels):
   boxes.append(Cuboid(name=f'fine_observed_{index}',pose=((voxel+.5)*.02).tolist()+[1.,0.,0.,0.],dims=[.025]*3))
  print('FINE_WORLD',len(boxes),flush=True)
  planner.arm.planner.update_world(SceneCfg(cuboid=boxes));planner.boxes=boxes
  phase='pregrasp'
  for flip in (0.,np.pi):
   candidate_R=R@Rotation.from_euler('z',flip).as_matrix()
   try:move(planner,grasp+np.array([0,0,.02]),candidate_R);R=candidate_R;break
   except RuntimeError:
    if flip:raise
  phase='descend_to_rim';move(planner,grasp,R)
  phase='close_gripper';q=measured();last=None;stable=0
  for _ in range(100):
   execute(q,255.);angles=np.array([d.joint(n).qpos[0] for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')]);stable=stable+1 if last is not None and np.max(np.abs(angles-last))<.0002 else 0;last=angles
   if stable>=10:break
  events.append(dict(phase=phase,driver_angles=angles.tolist(),time=float(d.time-start)))
  phase='lift_clear_rim';move(planner,grasp+np.r_[radial*.04,.03],R,255.)
  phase='lift';lift_R=Rotation.from_euler('y',-25,degrees=True).as_matrix()@R
  move(planner,grasp+np.r_[radial*.15,.14],lift_R,255.)
  phase='verify_hold';q=measured()
  for _ in range(50):execute(q,255.)
 except Exception as e:events.append(dict(phase=phase,error=str(e),time=float(d.time-start)))
lifts={m.body(b).name:float(d.xpos[b,2]-initial[b][2]) for b in bodies};success=phase=='verify_hold' and any(v>.10 for v in lifts.values())
report=dict(success=success,events=events,private_bowl_lift_m=lifts,contacts=list(audit.events.values()),simulated_seconds=float(d.time-start),limitations='Experimental sensor rim fit. Bowl-volume contact allowance does not certify other arm links or held-payload clearance.');(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
