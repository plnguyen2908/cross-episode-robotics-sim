"""Continue a physical pick; observed-map navigation to a perceived dining table."""
import argparse,json,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image
from cross_episode_sim.navigation.pipeline_video import PipelineVideoPanel
from tqdm import trange
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.perception.visual_targets import GroundingDinoDetector
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.observed_payload import ObservedPayload,PayloadDocking,refine_held_payload
from cross_episode_sim.navigation.observed_undock import ObservedUndock
from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import ContactAudit
p=argparse.ArgumentParser();p.add_argument('--pick',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=120);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_binary_path(str(a.pick/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.pick/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
meta=json.loads((a.pick/'continuation.json').read_text());report=json.loads((a.pick/'report.json').read_text())
if not report['success']:raise ValueError('Source grasp failed')
gripper_joints={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_0/gripper/')}
geo=RobotGeometry(m,gripper_joint_positions=gripper_joints);x=np.load(meta['input']);mask=observed_target_mask(x['points'],x['target'],report['world_stats']['support_z'])
qgrasp=np.load(a.pick/'validated_paths.npz')['grasp'][-1];T=geo.fk(qgrasp)[geo.tcp_body]@geo.tcp_local
points_tool=(x['points'][mask]-T[:3,3])@T[:3,:3];payload=ObservedPayload(geo,points_tool)
np.save(a.output/'observed_payload_tool.npy',points_tool)
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12.));slam=RGBDMapper();audit=ContactAudit(m);support=[];guard_detail=None
# Reuse prior observed floor and occupied base footprints in the new SLAM origin.
nav=Path(meta['navigation_source'])
from cross_episode_sim.navigation.map_continuation import restore_observed_map
support=restore_observed_map(grid,nav)
from cross_episode_sim.mapping.continuity import load_memory,save_memory,table_targets
from cross_episode_sim.mapping.room_memory import ObservedRoomMemory
memory=load_memory(nav)
from cross_episode_sim.control.observed_payload import invalidate_moved_surface
source_config=json.loads((nav/'summary.json').read_text())['config']
invalidated=invalidate_moved_surface(grid,x['points'][mask],x['points'][~mask],[source_config['resolution'],source_config['resolution'],source_config['height_resolution']])
(a.output/'dynamic_map_update.json').write_text(json.dumps(dict(invalidated_cached_voxels=invalidated,reason='Moved observed grasp component; static depth samples protected; cells become unknown')))
packet=None;tracking=None;checker=None;checker_stamp=None

def guard(forward,yaw,dt):
 global guard_detail,checker,checker_stamp
 if tracking is None or not tracking.accepted:return False
 if any(abs(float(d.joint(name).qpos[0])-value)>.03 for name,value in gripper_joints.items()):
  guard_detail={'reason':'finger_configuration_changed'};return False
 T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
 if not support or np.linalg.norm(pose[:2]-support[-1][:2])>.05:support.append(pose.copy())
 # Geometry and map change only on a new sensor packet. Reuse the same
 # conservative checker between 50-Hz commands; still sweep every command.
 if checker is None or checker_stamp!=packet.timestamp:
  checker=payload.checker(grid,packet.arm.position,pose,support)
  checker_stamp=packet.timestamp
 horizon=max(dt,.3);end=pose.copy();end[:2]+=forward*horizon*np.array([np.cos(pose[2]),np.sin(pose[2])]);end[2]+=yaw*horizon
 ok=checker.swept(pose,end);guard_detail=checker.detail;return ok

base=OnboardBaseActuator(m,d,motion_guard=guard)
from cross_episode_sim.perception.daaam_grounding import DaaamGrounder,LocalVLM
vlm=LocalVLM('Qwen/Qwen3-VL-8B-Instruct',device='cuda:1')
rooms=ObservedRoomMemory(vlm,remembered=memory['rooms'])
detector=DaaamGrounder(vlm,'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',a.output/'grounding',categories=('dining table',))
policy=SemanticNavigator(grid,detector,'dining table',config=HouseExplorationConfig(max_seconds=a.seconds,arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos)),preferred_distance=1.1)
policy.targets.targets=table_targets(memory)
departure=ObservedUndock(PayloadDocking(grid,payload));departure_done=False;departure_map_stamp=None;payload_refined=False;attachment_reference=points_tool.copy()
start=d.time;wall=time.monotonic();status='running';rows=[];map_ready=False;video_panel=PipelineVideoPanel()
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,imageio.get_writer(str(a.output/'slam_carry.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as map_video,open(a.output/'progress.jsonl','w',buffering=1) as log:
 for tick in trange(round(a.seconds*50),desc='Carry and dock'):
  if tick%5==0:
   raw=sensors.capture()
   if tick%5==0:
    try:
     points_tool,refinement=refine_held_payload(raw,geo,attachment_reference)
     payload=ObservedPayload(geo,points_tool);departure.docking.payload=payload
     np.save(a.output/'observed_payload_tool.npy',points_tool)
     (a.output/'payload_refinement.json').write_text(json.dumps(refinement,indent=2))
    except ValueError as exc:
     (a.output/'payload_refinement.json').write_text(json.dumps(dict(accepted=False,error=str(exc))))
    payload_refined=True
   packet=payload.filter_packet(raw);tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking,geometry=True)
   rooms.update(packet,tracking,[])
   video.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in raw.cameras.values()]))
   view=video_panel.render(grid,tracking,raw,'carry');map_video.append_data(view)
   if tick%50==0:Image.fromarray(view).save(a.output/'latest_frame.png')
  if not map_ready:
   map_ready=tracking.accepted and guard(0.,0.,.02)
   if not map_ready:
    if tick%5==0:log.write(json.dumps(dict(time=float(d.time-start),status='refresh_post_grasp_geometry',detail=guard_detail))+'\n')
    base.step(0.,0.,diagnostic_observer=audit.observe)
    if float(d.time-start)>=3.:status='post_grasp_geometry_unresolved';break
    continue
  # Leave the close grasp dock before asking the room explorer to rotate.
  # Distance and route come from observed floor/obstacles, not a fixed retreat.
  if not departure_done:
   if departure_map_stamp!=packet.timestamp:
    policy.explorer.layers(tracking.T_map_base[:2,3]);departure_map_stamp=packet.timestamp
   row,col=policy.explorer.cell(tracking.T_map_base[:2,3])
   if departure.follower is None and policy.explorer.free[row,col]:
    departure_done=True
   else:
    command=departure.step(packet,tracking,policy.explorer.free,float(d.time))
    if departure.error:
     status=departure.error;guard_detail=departure.docking.checker.detail if departure.docking.checker else None;break
    departure_done=command['status']=='undock_complete'
  if departure_done:command=policy.step(packet,tracking,float(d.time))
  if tick%5==0:
   row=dict(time=float(d.time-start),**command);rows.append(row);log.write(json.dumps(row)+'\n')
   if tick%100==0:print(json.dumps(row),flush=True)
  try:base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
  except RuntimeError as e:status='carry_clearance_rejected';print(str(e),guard_detail,flush=True);break
  if policy.terminal and base.stationary():status=policy.terminal;break
 else:status='budget_exhausted'
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state)
(a.output/'departure_events.json').write_text(json.dumps(departure.docking.events,indent=2))
grid.save(a.output);slam.save(a.output/'slam');target=policy.follower.target
save_memory(a.output,memory,room_events=rooms.events,targets=policy.targets.targets)
out=dict(status=status,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall,guard_detail=guard_detail,selected_object=None if target is None else dict(label=target.label,position=target.position.tolist()),contact_diagnostics=list(audit.events.values()),docking_success=status=='target_reached',limitations='Observed payload rigid attachment estimated from pregrasp RGB-D and executed grasp pose. No ground-truth object binding or table pose. Transport and placement not certified by detection alone.')
(a.output/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
