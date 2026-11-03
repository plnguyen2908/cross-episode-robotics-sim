"""M2T2 placement at a sensor-detected dock; scene truth is private scoring only."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from scipy.spatial import cKDTree
from curobo._src.geom.types import SceneCfg,Cuboid
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.perception.visual_targets import GroundingDinoDetector,table_points
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.observed_payload import ObservedPayload,reconstruct_held_points
from cross_episode_sim.robot.arm_planner import RobotArmPlanner
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb
p=argparse.ArgumentParser();p.add_argument('--dock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--isolated-initial-state',action='store_true');p.add_argument('--observe-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r=json.loads((a.dock/'report.json').read_text())
if not r.get('docking_success',False) and not (a.isolated_initial_state and r.get('isolated_placement_reset')) and not (a.observe_only and (r.get('initialization_valid') or r.get('approach_complete') or r.get('status') in ('needs_floor_observation','budget_exhausted'))):raise ValueError('Require sensor docking or explicit isolated reset')
m=mujoco.MjModel.from_binary_path(str(a.dock/'private_checkpoint/model.mjb'));d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(a.dock/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
c=MolmoSpacesController(m,d);hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')};geo=RobotGeometry(m);q=c.robot_view.get_move_group('arm').joint_pos.copy();tool_points=np.load(a.dock/'observed_payload_tool.npy') if (a.dock/'observed_payload_tool.npy').exists() else None;payload=ObservedPayload(geo,tool_points) if tool_points is not None else None;tcp=geo.fk(q)[geo.tcp_body]@geo.tcp_local
# Private scoring remembers only the initially held bowl, never a destination pose.
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
bowl_roots=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and any(word in (m.body(int(m.jnt_bodyid[j])).name or '').lower() for word in ('bowl','cup','mug'))]
private_lookup=contact_root_lookup(m.body_parentid,bowl_roots);finger_bodies={i for i in range(m.nbody) if (m.body(i).name or '').startswith('robot_0/gripper/')};initially_held=set()
for contact in d.contact:
 b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
 if private_lookup[b1]>=0 and b2 in finger_bodies:initially_held.add(int(private_lookup[b1]))
 if private_lookup[b2]>=0 and b1 in finger_bodies:initially_held.add(int(private_lookup[b2]))
from cross_episode_sim.navigation.pipeline_video import PipelineVideoPanel
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.slam import RGBDMapper
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,extent_m=12.));slam=RGBDMapper();panel=PipelineVideoPanel();selected_tcp=None;last_frame_stamp=None
if (a.dock/'navigation_map.npz').exists():restore_observed_map(grid,a.dock)
start=d.time;phase='observe';events=[];success=False;error=None;step=0
with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,imageio.get_writer(str(a.output/'slam_m2t2.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as diagnostic_video:
 def frame():
  global last_frame_stamp
  packet=sensors.capture()
  if last_frame_stamp==packet.timestamp:return
  last_frame_stamp=packet.timestamp
  tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking)
  diagnostic_video.append_data(panel.render(grid,tracking,packet,phase,selected_tcp))
  video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in packet.cameras.values()]))
 def command(joints,grip):
  global step
  c.step(dict(base=hold['base'],arm=joints,gripper=np.array([grip])))
  for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
  if step%5==0:frame()
  step+=1
 def execute(path,grip):
  for joints in path:command(joints,grip)
  steady=0
  for _ in range(100):
   command(path[-1],grip);measured=c.robot_view.get_move_group('arm').joint_pos
   velocity=np.array([d.joint(n).qvel[0] for n in geo.joint_names]);steady=steady+1 if np.max(abs(measured-path[-1]))<.015 and np.max(abs(velocity))<.04 else 0
   if steady>=5:return
  raise RuntimeError('Arm did not settle')
 frame()
 try:
  raw=sensors.capture();detector=GroundingDinoDetector(verify_tables=False);surfaces=[];points=[];colors=[]
  detections={k:detector.detect(camera.rgb) for k,camera in raw.cameras.items()}
  for key,camera in raw.cameras.items():
   if not any(v.label=='dining table' for v in detections[key]):
    detections[key].extend(detector.detect(camera.rgb,text='a dining table. a chair. a kitchen counter.'))
  (a.output/'detections.json').write_text(json.dumps({k:[dict(label=v.label,box=v.box,score=v.score) for v in ds] for k,ds in detections.items()},indent=2))
  from cross_episode_sim.perception.daaam_grounding import LocalVLM,masked_views,parse_label
  verifier=LocalVLM(device='cuda:1');verified=[]
  for key,ds in detections.items():
   accepted=[]
   for detection in ds:
    if detection.label!='dining table':accepted.append(detection);continue
    camera=raw.cameras[key];mask=np.zeros(camera.depth_m.shape,bool);x0,y0,x1,y1=map(int,detection.box);mask[max(0,y0):min(mask.shape[0],y1),max(0,x0):min(mask.shape[1],x1)]=True
    from PIL import Image
    context=[Image.fromarray(raw.cameras['navigation'].rgb)]
    answer=verifier.label_with_prompt(context+masked_views(camera.rgb,mask), 'The first image is the current navigation-camera context. The next two show a highlighted furniture region and its crop from the current '+key+' camera. Identify the supporting furniture, using both onboard views. Ignore the held vessel and robot arm. Respond using exactly these JSON keys: {"category": "dining table", "description": "brief visual evidence"}. The category value must be dining table, other, or unknown. Keep description under 20 words. Use unknown if uncertain.',max_new_tokens=160);verified.append(dict(camera=key,response=answer))
    try:
     if parse_label(answer)['category']=='dining table':accepted.append(detection)
    except (ValueError,KeyError):pass
   detections[key]=accepted
  (a.output/'table_verification.json').write_text(json.dumps(verified,indent=2));del verifier
  if payload is None:
   tool_points=reconstruct_held_points(raw,geo);payload=ObservedPayload(geo,tool_points)
  np.save(a.output/'observed_payload_tool.npy',tool_points)
  packet=payload.filter_packet(raw)
  for key,camera in packet.cameras.items():
   for detection in detections[key]:
    if detection.label=='dining table':
     surface=table_points(camera,detection,np.eye(4))
     if len(surface):surfaces.append(surface)
   scene_valid=camera.valid_depth if camera.robot_mask is None else camera.valid_depth&~camera.robot_mask
   v,u=np.where(scene_valid);z=camera.depth_m[v,u];xyz=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z]@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
   valid=np.linalg.norm(xyz,axis=1)<(6. if a.observe_only else 1.6)
   for box in geo.boxes(q):valid &= ~np.all(np.abs((xyz-box.center)@box.axes)<=box.half+.01,axis=1)
   points.append(xyz[valid]);colors.append(camera.rgb[v,u][valid])
  if not surfaces:raise RuntimeError('No freshly observed dining-table surface')
  surface=np.concatenate(surfaces);points=np.concatenate(points);colors=np.concatenate(colors);del detector
  memory=a.dock/'observed_table_surface_base.npy'
  if memory.exists():
   remembered=np.load(memory)
   # Same-episode observed surface memory, re-anchored with SLAM, not a model mesh.
   # Fresh semantic confirmation is still required above.
   color_tree=cKDTree(points)
   surface_colors=np.load(a.dock/'observed_table_surface_rgb.npy') if (a.dock/'observed_table_surface_rgb.npy').exists() else colors[color_tree.query(remembered)[1]]
   surface=np.concatenate([surface,remembered])
   points=np.concatenate([points,remembered]);colors=np.concatenate([colors,surface_colors])
   _,unique=np.unique(np.floor(points/.01).astype(int),axis=0,return_index=True)
   points=points[unique];colors=colors[unique]
   print('REMEMBERED_TABLE_POINTS',len(remembered),flush=True)
  import torch
  torch.cuda.empty_cache()
  held=tool_points@tcp[:3,:3].T+tcp[:3,3]
  np.savez_compressed(a.output/'input.npz',points=points,rgb=colors,held_points=held,ee_pose=tcp,table_surface=surface)
  if a.observe_only:
   center=(held[:,:2].min(0)+held[:,:2].max(0))/2
   radius=float(np.linalg.norm(held[:,:2]-center,axis=1).max()+.03)
   (a.output/'support_filter.json').write_text(json.dumps(dict(radius_m=radius,edge_margin_m=.03),indent=2))
   (a.output/'report.json').write_text(json.dumps(dict(observation_success=True,scope='Fresh onboard table/held-object perception only; no placement executed'),indent=2))
   print('TABLE_OBSERVED',len(surface),flush=True)
   raise SystemExit(0)
  subprocess.run(['.venv-m2t2/bin/python','scripts/m2t2_worker.py','--task','place','--input',str(a.output/'input.npz'),'--output',str(a.output/'predictions.json'),'--checkpoint','/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth'],check=True)
  vox=np.unique(np.floor(points/.025).astype(int),axis=0);boxes=[]
  for i,(lo,hi) in enumerate(merge_voxels(vox)):
   lo=np.array(lo);hi=np.array(hi);boxes.append(Cuboid(name=f'observed_{i}',pose=[*((lo+hi+1)*.0125),1,0,0,0],dims=((hi-lo+1)*.025+.003).tolist()))
  calibration=robot_calibration(m,d,a.output/'robot');fixed_support=[[*s['center'],s['radius']] for s in calibration['collision_spheres'].get('base',[])];arm=RobotArmPlanner(None,{},kinematics=calibration,collision_cache={'cuboid':max(16,len(boxes)),'mesh':2});arm.planner.update_world(SceneCfg(cuboid=boxes));planner=ObservedGraspPlanner(arm,geo,boxes,fixed_floor_support_spheres=fixed_support);tree=cKDTree(surface)
  rows=json.loads((a.output/'predictions.json').read_text())['candidates'];rows=[x for x in rows if tree.query(x['contact'])[0]<.035];rows.sort(key=lambda x:np.linalg.norm(np.array(x['tcp'])[:3,3]-tcp[:3,3])-.1*x['confidence']);chosen=None
  if r.get('placement_target_base') is not None:
   preferred=np.asarray(r['placement_target_base'])
   rows.sort(key=lambda row:np.linalg.norm(np.asarray(row['contact'])-preferred)+.2*np.linalg.norm(np.asarray(row['tcp'])[:3,3]-tcp[:3,3])-.05*row['confidence'])
  from cross_episode_sim.control.observed_placement import supported_interior
  # Use the observed vessel's horizontal radius plus a 3 cm edge margin.
  held_xy=held[:,:2];center=(held_xy.min(0)+held_xy.max(0))/2
  support_radius=float(np.linalg.norm(held_xy-center,axis=1).max()+.03)
  supported=supported_interior(surface,np.array([row['contact'] for row in rows]),support_radius)
  rows=[row for row,ok in zip(rows,supported) if ok]
  (a.output/'support_filter.json').write_text(json.dumps(dict(radius_m=support_radius,edge_margin_m=.03,retained=len(rows)),indent=2))
  print('INTERIOR_PLACE_CANDIDATES',len(rows),'support radius',support_radius,flush=True)
  seen=set();diverse=[]
  for row in rows:
   key=(row['group'],*np.floor(np.asarray(row['contact'])/.04).astype(int))
   if key not in seen:seen.add(key);diverse.append(row)
  # Do not exhaust the budget on one wrist orientation.
  groups={}
  for row in diverse:groups.setdefault(row['group'],[]).append(row)
  diverse=[group[i] for i in range(max(map(len,groups.values()),default=0)) for group in groups.values() if i<len(group)]
  # Keep the learned placement orientation/contact, but allow a small physical
  # release clearance above the measured surface. Every arm/payload path still
  # passes the unchanged collision checks; gravity and support are scored after
  # release. This avoids requiring penetration into voxelized table geometry.
  release_candidates=[]
  for row in diverse[:40]:
   for clearance in (0.,.005,.01,.015,.02,.03,.04):
    candidate=dict(row);goal=np.array(row['tcp']);goal[2,3]+=clearance
    candidate.update(model_tcp=row['tcp'],tcp=goal.tolist(),release_clearance_m=clearance)
    release_candidates.append(candidate)
  for row in release_candidates:
   goal=np.array(row['tcp'])
   try:
    plan_phase='approach';approach=planner.plan(q,goal[:3,3]+[0,0,.08],goal[:3,:3]);plan_phase='lower';lower=planner.plan(approach[-1],goal[:3,3],goal[:3,:3]);plan_phase='retreat';retreat=lower[::-1].copy()
    if not all(planner.clear(joints) for joints in retreat):raise RuntimeError('Reverse lowering path is not clear for release retreat')
    for joints in np.concatenate([approach,lower]):
     T=geo.fk(joints)[geo.tcp_body]@geo.tcp_local;xyz=tool_points[::max(1,len(tool_points)//150)]@T[:3,:3].T+T[:3,3]
     if not planner.world_clear(np.c_[xyz,np.full(len(xyz),.006)][None],margin=.002):raise RuntimeError('Held points collide on placement approach')
    chosen=row;break
   except RuntimeError as e:events.append(plan_phase+': '+str(e));print('PLACE_REJECT',len(events),plan_phase,str(e),flush=True)
  if chosen is None:raise RuntimeError('No validated M2T2 place path')
  selected_tcp=np.asarray(chosen['tcp'])
  (a.output/'chosen.json').write_text(json.dumps(chosen,indent=2));phase='approach';execute(approach,255);phase='lower';execute(lower,255)
  phase='release';last=None;stable=0
  for _ in range(100):
   command(lower[-1],0);angle=np.array([d.joint(n).qpos[0] for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')]);stable=stable+1 if last is not None and np.max(abs(angle-last))<.0002 else 0;last=angle
   if stable>=10:break
  if stable<10:raise RuntimeError('Release did not settle')
  phase='retreat';execute(retreat,0)
  # Wait for observed vessel-region motion to settle, never a hidden contact
  # flag. The private support/stability scoring below remains unchanged.
  from cross_episode_sim.control.release_stability import ReleaseStability
  stability=ReleaseStability(chosen['contact'],support_radius)
  phase='visual_release_stability';visually_stable=False
  for settle_tick in range(500):
   command(retreat[-1],0)
   if settle_tick%5==0:
    observation=sensors.capture()
    visually_stable=stability.update(observation.cameras['wrist'],observation.timestamp)
    if visually_stable:break
  (a.output/'release_stability.json').write_text(json.dumps(stability.events,indent=2))
  if not visually_stable:raise RuntimeError('Released vessel region did not visually settle')
  phase='score';checks=[];score_diagnostics=[]
  # Private semantic physical relation: any released bowl supported by a dining table.
  from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
  bowls=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and any(word in (m.body(int(m.jnt_bodyid[j])).name or '').lower() for word in ('bowl','cup','mug'))]
  tables=[i for i in range(m.nbody) if 'dining_table' in (m.body(i).name or '')];br=contact_root_lookup(m.body_parentid,bowls);tr=contact_root_lookup(m.body_parentid,tables);pads={i for i in range(m.nbody) if (m.body(i).name or '').startswith('robot_0/gripper/')}
  for _ in range(50):
   command(retreat[-1],0);supported=set();touch=set()
   for contact in d.contact:
    b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
    if br[b1] in initially_held and tr[b2]>=0 and abs(contact.frame[2])>.5:supported.add(int(br[b1]))
    if br[b2] in initially_held and tr[b1]>=0 and abs(contact.frame[2])>.5:supported.add(int(br[b2]))
    if br[b1]>=0 and b2 in pads:touch.add(int(br[b1]))
    if br[b2]>=0 and b1 in pads:touch.add(int(br[b2]))
   stable=set();velocities={}
   for body in supported-touch:
    joint=int(m.body_jntadr[body]);adr=int(m.jnt_dofadr[joint])
    linear=float(np.linalg.norm(d.qvel[adr:adr+3]));angular=float(np.linalg.norm(d.qvel[adr+3:adr+6]));velocities[body]=dict(linear_m_s=linear,angular_rad_s=angular)
    if linear<.03 and angular<.1:stable.add(body)
   checks.append(stable)
   score_diagnostics.append(dict(timestamp=float(d.time),initially_held=list(initially_held),supported=list(supported),gripper_contact=list(touch),stable=list(stable),velocities=velocities))
  (a.output/'private_release_score.json').write_text(json.dumps(score_diagnostics,indent=2))
  success=bool(set.intersection(*checks))
 except Exception as e:error=str(e);print(type(e).__name__,error,flush=True)
 frame()
grid.save(a.output);slam.save(a.output/'slam')
(a.output/'report.json').write_text(json.dumps(dict(success=success,phase=phase,error=error,simulated_seconds=float(d.time-start),planning_rejections=events,scope='single released vessel/table support check, not full set-table',isolated_initial_state=a.isolated_initial_state,source_docking_scope=r.get('scope','unspecified')),indent=2));print('PLACE_SUCCESS',success,flush=True)
# Preserve the actual resulting physics state for the next pipeline stage.
# This is harness-only state, never exposed to perception or action selection.
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None)
spec=mujoco.mjtState.mjSTATE_INTEGRATION
state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
np.save(checkpoint/'state.npy',state)
(a.output/'continuation.json').write_text(json.dumps(dict(source_dock=str(a.dock.resolve()),success=success),indent=2))
if (a.output/'chosen.json').exists():
 subprocess.run([os.sys.executable,'scripts/render_m2t2_geometry.py','--place',str(a.output)],check=True)
raise SystemExit(0 if success else 1)
