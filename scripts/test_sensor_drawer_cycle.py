"""Complete the sensor-only drawer open/release/close/release cycle in one live scene.

Initial grasp/open is the tested handle script. Its private report is never read
by this controller. The returned live robot/perception objects continue forward;
no checkpoint is restored between the skills.
"""
import argparse,json,runpy,sys,time
from pathlib import Path
import numpy as np,mujoco
from dataclasses import asdict
from curobo._src.geom.types import Cuboid,SceneCfg
from cross_episode_sim.control.sensor_drawer import SensorDrawerMotion
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface
from cross_episode_sim.recording.deferred import StateVideoRecorder

from cross_episode_sim.control.sensor_drawer_geometry import adaptive_partitions

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/m2t2_handles_v2/drawer'));p.add_argument('--output',type=Path,required=True);p.add_argument('--target',type=int,default=2);p.add_argument('--motion-step',type=float,default=.04);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from contextlib import ExitStack
from cross_episode_sim.runtime.episode import EpisodeRuntime,current_runtime
with ExitStack() as episode_scope:
 runtime=current_runtime()
 if runtime is None:
  runtime=episode_scope.enter_context(EpisodeRuntime(artifact_root=a.output).activate())
 original=sys.argv
 try:
  sys.argv=['scripts/test_m2t2_handle_execution.py','--source',str(a.source),'--output',str(a.output/'opening'),'--target',str(a.target),'--surface-filter','--grasp-ranking','horizontal','--voxel-size','.01','--approach-stand-off','.12','--align-panel','--open-to-stop','--open-step',str(a.motion_step)]
  ns=runpy.run_path(sys.argv[0],run_name='__main__')
 finally:sys.argv=original
 if ns.get('chosen') is None:raise RuntimeError('No executable observed grasp')
 m,d,geo,controller,arm=map(ns.get,['m','d','geo','controller','arm']);normal=ns['normal'];hold=ns['hold'];initial_private=ns['private_initial_qpos'];phase='setup';start=d.time
 results={};progress=[];last_cloud=None;refresh=0;private_open_qpos=None
 # Only robot state and observed RGB-D enter the controller below.
 q=lambda:controller.robot_view.get_move_group('arm').joint_pos.copy()
 def tcp(positions=None):return geo.fk(q() if positions is None else positions)[geo.tcp_body]@geo.tcp_local
 finger_names=['robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint']
 fingers=lambda:np.array([d.joint(n).qpos[0] for n in finger_names])

 with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'cycle_recording') as recording:
  def notify(stage,detail):
   global phase
   phase=stage;event=dict(stage=stage,simulated_time=float(d.time-start),**detail);progress.append(event);print(json.dumps(event),flush=True)
  def command(arm_q,grip):
   action={k:v.copy() for k,v in hold.items()};action['arm']=arm_q;action['gripper']=np.array([grip]);controller.step(action)
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
   recording.record(phase)
  def observed_planner(current,target,contact_patch):
   global refresh,last_cloud
   packet=sensors.capture();cloud=[]
   for c in packet.cameras.values():
    valid=c.valid_depth.copy()
    if c.robot_mask is not None:valid &= ~c.robot_mask
    v,u=np.where(valid);z=c.depth_m[v,u]
    points=np.c_[(u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z]
    points=points@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3];cloud.append(points)
   points=np.concatenate(cloud);keep=(np.linalg.norm(points,axis=1)<1.8)&(points[:,2]>.08)
   for box in geo.boxes(q()):keep &= ~np.all(np.abs((points-box.center)@box.axes)<=box.half+.008,axis=1)
   points=points[keep];last_cloud=points
   if contact_patch:
    # Local intended hand/contact sweep only, not the whole drawer or cabinet.
    # Physics collisions remain active. All other sensed surfaces remain obstacles.
    delta=target[:3,3]-current[:3,3];length=np.dot(delta,delta)
    t=np.clip((points-current[:3,3])@delta/max(length,1e-12),0,1)
    distance=np.linalg.norm(points-(current[:3,3]+t[:,None]*delta),axis=1)
    points=points[distance>.075]
   boxes=[]
   for lo,hi,resolution in adaptive_partitions(points,geo.boxes(q())):
    boxes.append(Cuboid(name=f'live_{len(boxes)}',pose=((lo+hi+1)*(resolution/2)).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*resolution).tolist()))
   refresh+=1;print('observed_world',refresh,len(points),len(boxes),flush=True)
   # ObservedGraspPlanner checks these boxes directly; no cuRobo fallback/cache.
   planner=ObservedGraspPlanner(arm,geo,boxes)
   print("start_clear",planner.clear(q()),getattr(planner,"last_rejection",None),flush=True)
   np.savez(a.output/f"observed_{refresh}.npz",points=points,q=q(),current=current,target=target)
   if getattr(planner,"last_rejection",None):
    rejected=planner.last_rejection.get("obstacle");print("rejected_box",next((b for b in boxes if b.name==rejected),None),flush=True)
   return planner
  motion=SensorDrawerMotion(joint_position=q,tcp_pose=tcp,fingers=fingers,command=command,planner_factory=observed_planner,outward_normal=normal,progress=notify)
  handle_pose=tcp().copy() # Tracked observed handle while physically grasped; robot FK only.
  try:
   results['open_release']=asdict(motion.release_and_withdraw())
   private_open_qpos=d.qpos.copy()
   if not results['open_release']['completed']:raise RuntimeError(results['open_release']['reason'])
   results['close_regrasp']=asdict(motion.regrasp(handle_pose))
   if not results['close_regrasp']['completed']:raise RuntimeError(results['close_regrasp']['reason'])
   results['close_push']=asdict(motion.move_until_resistance(opening=False,step=a.motion_step))
   if not results['close_push']['completed']:raise RuntimeError(results['close_push']['reason'])
   results['close_release']=asdict(motion.release_and_withdraw())
  except Exception as exc:results['error']=str(exc);print('FAILED',exc,flush=True)
  # Capture verification only; nothing below influences further actions.
  private_final_qpos=d.qpos.copy();private_contacts=[(int(c.geom1),int(c.geom2)) for c in d.contact]
  results['motion_events']=motion.events;results['simulated_seconds']=float(d.time-start);results['progress']=progress

 # Private final scoring, after all control.
 sliding=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE and not (m.joint(j).name or '').startswith('robot_0/')]
 score={m.joint(j).name:dict(initial=float(initial_private[m.jnt_qposadr[j]]),final=float(private_final_qpos[m.jnt_qposadr[j]]),limits=m.jnt_range[j].tolist()) for j in sliding}
 pads={m.body(n).id for n in ['robot_0/gripper/left_pad','robot_0/gripper/right_pad']}
 released=not any((int(m.geom_bodyid[g1]) in pads and 'handle' in (m.geom(g2).name or '')) or (int(m.geom_bodyid[g2]) in pads and 'handle' in (m.geom(g1).name or '')) for g1,g2 in private_contacts)
 closed=all(abs(v['final']-v['initial'])<.01 for v in score.values())
 opened=False
 if private_open_qpos is not None:
  opened=any(abs(float(private_open_qpos[m.jnt_qposadr[j]])-float(initial_private[m.jnt_qposadr[j]]))>.05 and min(abs(float(private_open_qpos[m.jnt_qposadr[j]])-limit) for limit in m.jnt_range[j])<.01 for j in sliding)
 results.update(private_fixture_scores=score,private_released=released,private_closed=closed,private_opened=opened,success=opened and closed and released and results.get('close_release',{}).get('completed',False),qualification='One prepositioned continuous drawer cycle; no navigation or cross-scene qualification')
 (a.output/'report.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)

(a.output/'runtime_metrics.json').write_text(json.dumps(runtime.metrics,indent=2))
