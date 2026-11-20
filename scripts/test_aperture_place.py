"""M2T2 placement at a sensor-detected dock; scene truth is private scoring only."""
from cross_episode_sim.runtime import artifacts
import argparse,json,os,subprocess,time,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial import cKDTree
from curobo._src.geom.types import SceneCfg,Cuboid
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.perception.visual_targets import table_points
from cross_episode_sim.perception.point_masks import PointMaskDetector
from cross_episode_sim.control.fixed_base_path import FixedBasePath
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.observed_payload import ObservedPayload,reconstruct_held_points
from cross_episode_sim.control.planner_cache import sensor_arm_planner
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
# Depth points left in the held object's place after withdrawing: more means it came out with the hand.
STILL_AT_GRIPPER_POINTS=200
PLACED_OBJECT_MIN_POINTS=100
# Release heights (m above the support) tried for a tilted placement, in order.
# Every passing microwave placement released from 5 or 3 cm (documented runs:
# attempts 20-28 of 32; fix106 replay: 19; fix111 replay: 56, past the old
# budget), never from 1.5 or 0.5 cm, so the higher ones go first.
TILTED_RELEASE_CLEARANCES=(.05,.03,.015,.005)
# Contacts this far (m) from the measured support centre count as centred. Diagnostic override for isolated comparisons only.
CENTRE_TOLERANCE=float(__import__('os').environ.get('PLACE_CENTRE_TOLERANCE_OVERRIDE','.04'))
# Low slide-in for a second object: off. It dragged bowls out (fix88, fix94); the tidy push replaces it.
LOW_PUSH_ENTRY=False
# After a placement into a cabinet: closed hand this far past the release point, this high above the shelf.
TIDY_PUSH_EXTRA=.14
TIDY_PUSH_HEIGHT=.06
p=argparse.ArgumentParser();p.add_argument('--receptacle',default='dining table');p.add_argument('--semantic-receptacle',default=None);p.add_argument('--observed-aperture',type=Path);p.add_argument('--dock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-candidates',type=int,default=0,help='Optional bounded candidate batch; zero preserves the existing exhaustive policy');p.add_argument('--local-paths-only',action='store_true',help='Return after checked local IK/joint paths; let the agent redock instead of global cspace fallback');p.add_argument('--approach-config',type=Path);p.add_argument('--align-only',action='store_true');p.add_argument('--current-pose',action='store_true',help='Attempt sensor-planned placement from the live pose without claiming a completed base dock');p.add_argument('--isolated-initial-state',action='store_true');p.add_argument('--observe-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r=json.loads(artifacts.read_text(a.dock/'report.json'))
if not (a.current_pose and r.get('initialization_valid')) and not r.get('docking_success',False) and not (a.isolated_initial_state and r.get('isolated_placement_reset')) and not (a.observe_only and (r.get('transit_success') is True or r.get('initialization_valid') or r.get('approach_complete') or r.get('status') in ('needs_floor_observation','budget_exhausted'))):raise ValueError('Require sensor docking or explicit isolated reset')
m,d=load_checkpoint(a.dock)
c=MolmoSpacesController(m,d);hold={g:c.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')};geo=RobotGeometry(m);q=c.robot_view.get_move_group('arm').joint_pos.copy();tool_points=artifacts.load_array(a.dock/'observed_payload_tool.npy') if artifacts.exists(a.dock/'observed_payload_tool.npy') else None;payload=ObservedPayload(geo,tool_points) if tool_points is not None else None;tcp=geo.fk(q)[geo.tcp_body]@geo.tcp_local
# Private scoring remembers only the initially held bowl, never a destination pose.
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
bowl_roots=[int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and any(word in (m.body(int(m.jnt_bodyid[j])).name or '').lower() for word in ('bowl','cup','mug'))]
private_lookup=contact_root_lookup(m.body_parentid,bowl_roots);finger_bodies={i for i in range(m.nbody) if (m.body(i).name or '').startswith('robot_0/gripper/')};initially_held=set()
for contact in d.contact:
 b1,b2=map(int,m.geom_bodyid[[contact.geom1,contact.geom2]])
 if private_lookup[b1]>=0 and b2 in finger_bodies:initially_held.add(int(private_lookup[b1]))
 if private_lookup[b2]>=0 and b1 in finger_bodies:initially_held.add(int(private_lookup[b2]))
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.slam import create_mapper
grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,extent_m=12.));slam=create_mapper();selected_tcp=None;last_frame_stamp=None
if artifacts.exists(a.dock/'navigation_map.npz'):restore_observed_map(grid,a.dock)
from cross_episode_sim.recording.deferred import StateVideoRecorder
start=d.time;wall_start=time.monotonic();phase='observe';events=[];success=False;error=None;step=0;warnings=[]
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recording:
 def frame():
  if 'tracking' in globals():recording.record_map(grid,tracking,phase,selected_tcp)
  else:recording.record(phase)
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
  raw=sensors.capture();tracking=slam.update(raw.cameras['navigation'],raw.timestamp)
  detector=PointMaskDetector(categories=(a.receptacle,),free_support=True);surfaces=[];points=[];colors=[];preferred_contacts=[];camera_hints={}
  if a.semantic_receptacle=='microwave':
   from cross_episode_sim.perception.container_interior import microwave_floor_detections
   detections={k:microwave_floor_detections(camera,a.receptacle) for k,camera in raw.cameras.items()}
  elif a.observed_aperture:
   detections={k:[] for k in raw.cameras}
  elif a.semantic_receptacle and ('interior' in a.receptacle or 'inside' in a.receptacle):
   from cross_episode_sim.perception.container_interior import container_interior_detections
   detections={k:container_interior_detections(camera.rgb,a.semantic_receptacle,a.receptacle) for k,camera in raw.cameras.items()}
  else:
   detections={k:detector.detect(camera.rgb) for k,camera in raw.cameras.items()}
  artifacts.write_text(a.output/'detections.json',json.dumps({k:[dict(label=v.label,box=v.box,score=v.score) for v in ds] for k,ds in detections.items()},indent=2))
  from cross_episode_sim.perception.daaam_grounding import LocalVLM,masked_views,parse_label
  verifier=LocalVLM(device='cuda:1');verified=[]
  for key,ds in detections.items():
   accepted=[]
   for detection in ds:
    if detection.label!=a.receptacle:accepted.append(detection);continue
    if a.semantic_receptacle=='microwave':
     from PIL import Image,ImageDraw
     camera=raw.cameras[key];clean=Image.fromarray(camera.rgb);marked=clean.copy()
     u,v=detection.point;ImageDraw.Draw(marked).ellipse((u-4,v-4,u+4,v+4),outline='magenta',width=2)
     answer=verifier.label_with_prompt([clean,marked],
         'The second image marks a point. Is that point on the interior floor or glass turntable '
         'INSIDE the open microwave cavity? The circular turntable is part of the microwave. '
         'Reject kitchen countertop, swinging door, robot and held container. '
         'Return JSON category "microwave", "other" or "unknown", and brief description.',max_new_tokens=160)
     verified.append(dict(camera=key,point=list(detection.point),response=answer))
     try:
      if parse_label(answer,('microwave',))['category']=='microwave':accepted.append(detection)
     except (ValueError,KeyError):pass
     continue
    camera=raw.cameras[key];mask=np.zeros(camera.depth_m.shape,bool);x0,y0,x1,y1=map(int,detection.box);mask[max(0,y0):min(mask.shape[0],y1),max(0,x0):min(mask.shape[1],x1)]=True
    if detection.mask is not None:mask=detection.mask.copy()
    from PIL import Image
    context=[Image.fromarray(raw.cameras['navigation'].rgb)]
    semantic=a.semantic_receptacle or a.receptacle
    prompt=('The first image is scene context. The next two highlight a region and its crop. '
            'Does the highlighted region belong to the '+semantic+'? A basket is a container/receptacle. '
            'Identify the item, not whether its bottom is currently visible. Ignore the robot and held item. '
            'Return JSON category '+json.dumps(semantic)+', other or unknown, and a short description.'
            if a.semantic_receptacle else
            'The first image is the navigation-camera context. The next two show a highlighted region and its crop from the '+key+' camera. Identify whether this is the requested support/receptacle: '+a.receptacle+'. Ignore the held item and robot arm. Respond with JSON category (exact requested name, other, or unknown) and description (under 20 words). Use unknown if uncertain.')
    answer=verifier.label_with_prompt(context+masked_views(camera.rgb,mask),prompt,max_new_tokens=160);verified.append(dict(camera=key,response=answer))
    try:
     if parse_label(answer,(semantic,))['category']==semantic:accepted.append(detection)
    except (ValueError,KeyError):pass
   detections[key]=accepted
  artifacts.write_text(a.output/'table_verification.json',json.dumps(verified,indent=2));del verifier
  if a.semantic_receptacle=='microwave':print('MICROWAVE_FLOOR_VERIFICATION',json.dumps(verified),flush=True)
  if payload is None:
   tool_points=reconstruct_held_points(raw,geo);payload=ObservedPayload(geo,tool_points)
  artifacts.save_array(a.output/'observed_payload_tool.npy',tool_points)
  packet=payload.filter_packet(raw)
  # A carried object is dynamic, not a permanent navigation obstacle. Fuse the
  # same payload-filtered observation used for manipulation scene geometry.
  grid.update(packet.cameras['navigation'],tracking)
  for key,camera in packet.cameras.items():
   for detection in detections[key]:
    if detection.label==a.receptacle:
     if a.semantic_receptacle and ('interior' in a.receptacle or 'inside' in a.receptacle):
      from cross_episode_sim.perception.interior_support import interior_support_points
      surface=interior_support_points(camera,detection)
     else:
      surface=table_points(camera,detection,np.eye(4),support_options=None if a.receptacle=='dining table' else dict(min_height=.03,max_height=1.8,min_span=.12,min_area=.005))
     if len(surface):
      surfaces.append(surface)
      from cross_episode_sim.control.observed_placement import support_hint
      hint=support_hint(camera,detection.point,surface)
      if hint is not None:
       preferred_contacts.append(hint);camera_hints[key]=hint
   scene_valid=camera.valid_depth if camera.robot_mask is None else camera.valid_depth&~camera.robot_mask
   v,u=np.where(scene_valid);z=camera.depth_m[v,u];xyz=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z]@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
   valid=np.linalg.norm(xyz,axis=1)<(6. if a.observe_only else 1.6)
   for box in geo.boxes(q):valid &= ~np.all(np.abs((xyz-box.center)@box.axes)<=box.half+.01,axis=1)
   points.append(xyz[valid]);colors.append(camera.rgb[v,u][valid])
  if a.observed_aperture:
   from cross_episode_sim.perception.aperture_support import aperture_support_points
   aperture=json.loads(artifacts.read_text(a.observed_aperture))
   # For microwaves, use the semantically pointed cavity floor. Historical
   # hinge geometry also includes nearby countertop/roof horizontal patches.
   surface=(np.empty((0,3)) if a.semantic_receptacle=='microwave' else
            aperture_support_points(np.concatenate(points),aperture))
   if len(surface):surfaces.append(surface)
   artifacts.write_text(a.output/'aperture_evidence.json',json.dumps(dict(aperture=aperture,measured_support_points=len(surface)),indent=2))
  if not surfaces:raise RuntimeError('No freshly observed support surface for '+a.receptacle)
  surface=np.concatenate(surfaces);points=np.concatenate(points);colors=np.concatenate(colors);del detector
  memory=a.dock/'observed_table_surface_base.npy'
  if artifacts.exists(memory):
   remembered=artifacts.load_array(memory)
   # Same-episode observed surface memory, re-anchored with SLAM, not a model mesh.
   # Fresh semantic confirmation is still required above.
   color_tree=cKDTree(points)
   surface_colors=artifacts.load_array(a.dock/'observed_table_surface_rgb.npy') if artifacts.exists(a.dock/'observed_table_surface_rgb.npy') else colors[color_tree.query(remembered)[1]]
   surface=np.concatenate([surface,remembered])
   points=np.concatenate([points,remembered]);colors=np.concatenate([colors,surface_colors])
   _,unique=np.unique(np.floor(points/.01).astype(int),axis=0,return_index=True)
   points=points[unique];colors=colors[unique]
   print('REMEMBERED_TABLE_POINTS',len(remembered),flush=True)
  import torch
  torch.cuda.empty_cache()
  held=tool_points@tcp[:3,:3].T+tcp[:3,3]
  from cross_episode_sim.control.observed_payload import invalidate_moved_surface
  map_pose=tracking.T_map_base
  cleared=invalidate_moved_surface(grid,held@map_pose[:3,:3].T+map_pose[:3,3],
      points@map_pose[:3,:3].T+map_pose[:3,3],[.01,.01,.01])
  print('DYNAMIC_PAYLOAD_MAP',json.dumps(dict(invalidated_cells=cleared,
      policy='old held-surface evidence becomes unknown; current static surfaces protected')),flush=True)
  artifacts.save_archive(a.output/'input.npz',points=points,rgb=colors,held_points=held,ee_pose=tcp,table_surface=surface,preferred_contacts=np.asarray(preferred_contacts).reshape(-1,3))
  active_mode=None;approach_options=None;approach_aperture=None;previous_contacts=[]
  if a.approach_config:
   from cross_episode_sim.control.manipulation_approach import approach_config,resolve_mode,opening_frame,front_placement_candidates,plan_front_entry,gripper_points_tool
   _config=json.loads(artifacts.read_text(a.approach_config))
   previous_contacts=_config.pop('previous_contacts_aperture',[])
   approach_options,approach_aperture=approach_config(_config)
   if a.observed_aperture:approach_aperture=json.loads(artifacts.read_text(a.observed_aperture))
   active_mode=resolve_mode(approach_options.approach_mode,surface,points,observed_aperture=approach_aperture is not None)
   print('MANIPULATION_APPROACH',active_mode,approach_options.as_dict(),flush=True)
   if (active_mode=='front_entry' and approach_options.front_wrist=='horizontal' and approach_aperture is not None
       and 'cabinet' in (a.semantic_receptacle or '')):
    # Storage cabinets start empty, so anything resting on the observed shelf
    # inside the opening is an object this episode already placed. Like the
    # oracle, slide the next one in and push those deeper: they are not
    # obstacles for the slide-in or occupied spots. Cabinet walls, doors and
    # the shelf itself stay obstacles.
    from cross_episode_sim.control.manipulation_approach import pushable_placed_points
    pushable=pushable_placed_points(points,surface,*opening_frame(surface,approach_aperture))
    print('PUSHABLE_PLACED_OBJECTS',json.dumps(dict(points=int(pushable.sum()),of=len(points))),flush=True)
    placed_points=points[pushable].copy()
    points=points[~pushable];colors=colors[~pushable]
  if a.observe_only:
   center=(held[:,:2].min(0)+held[:,:2].max(0))/2
   radius=float(np.linalg.norm(held[:,:2]-center,axis=1).max()+.03)
   artifacts.write_text(a.output/'support_filter.json',json.dumps(dict(radius_m=radius,edge_margin_m=.03),indent=2))
   artifacts.write_text(a.output/'report.json',json.dumps(dict(observation_success=True,scope='Fresh onboard table/held-object perception only; no placement executed'),indent=2))
   print('TABLE_OBSERVED',len(surface),flush=True)
   raise SystemExit(0)
  from cross_episode_sim.runtime.episode import current_runtime
  worker_command=[sys.executable,'scripts/m2t2_worker.py','--task','place','--input',str(a.output/'input.npz'),'--output',str(a.output/'predictions.json'),'--checkpoint','/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth']
  if a.semantic_receptacle and ('interior' in a.receptacle or 'inside' in a.receptacle):
   worker_command[1]='scripts/container_place_proposals.py'
  artifacts.write_text(a.output/'predictions.json',json.dumps(dict(candidates=[])))
  print('ISOLATED_GEOMETRY_INSERTION_EXPERIMENT: no learned proposals requested',flush=True)
  # Cabinet front entry: 1 cm observed boxes (as the drawer planner uses).
  # 2.5 cm boxes over-cover the opening's rail and door edge by up to 2.5 cm;
  # live/isolated slide-ins were blocked by 1-8 mm overlaps with them.
  voxel=.01 if (active_mode=='front_entry' and 'cabinet' in (a.semantic_receptacle or '')) else .025
  vox=np.unique(np.floor(points/voxel).astype(int),axis=0);boxes=[]
  for i,(lo,hi) in enumerate(merge_voxels(vox)):
   lo=np.array(lo);hi=np.array(hi);boxes.append(Cuboid(name=f'observed_{i}',pose=[*((lo+hi+1)*voxel/2),1,0,0,0],dims=((hi-lo+1)*voxel+.003).tolist()))
  print('OBSERVED_BOXES',json.dumps(dict(voxel_m=voxel,boxes=len(boxes))),flush=True)
  calibration=robot_calibration(m,d,a.output/'robot');fixed_support=[[*s['center'],s['radius']] for s in calibration['collision_spheres'].get('base',[])];arm=sensor_arm_planner(kinematics=calibration,activation_distance=.012,collision_cache={'cuboid':max(16,len(boxes)),'mesh':2});arm.planner.update_world(SceneCfg(cuboid=boxes));planner=FixedBasePath(arm,geo,boxes,allow_curobo_fallback=not a.local_paths_only,fixed_floor_support_spheres=fixed_support);tree=cKDTree(surface)
  if active_mode=='front_entry':
   from cross_episode_sim.control.calibrated_arm_planner import CalibratedArmPlanner
   planner=CalibratedArmPlanner(arm,geo,boxes,panel_points=np.empty((0,3)),hinge_xy=[0.,0.],reference_tcp=np.zeros(3),arm_body_ids=[m.body(f'robot_0/fr3_link{i}').id for i in range(8)],allow_curobo_fallback=False,fixed_floor_support_spheres=fixed_support)
  retreat_planner=None
  if active_mode=='front_entry' and approach_options.object_tilt_degrees:
   from cross_episode_sim.control.fixture_approach import calibrated_finger_planner
   planner=calibrated_finger_planner(arm,geo,boxes,m,d,arm_boxes=True,palm_box=True,
       allow_curobo_fallback=False,fixed_floor_support_spheres=fixed_support)
   opened={m.joint(j).name:0. for j in range(m.njnt) if '/gripper/' in (m.joint(j).name or '')}
   retreat_planner=calibrated_finger_planner(arm,geo,boxes,m,d,arm_boxes=True,palm_box=True,
       gripper_joint_positions=opened,allow_curobo_fallback=False,fixed_floor_support_spheres=fixed_support)
  rows=json.loads(artifacts.read_text(a.output/'predictions.json'))['candidates'];rows=[x for x in rows if tree.query(x['contact'])[0]<.035];rows.sort(key=lambda x:np.linalg.norm(np.array(x['tcp'])[:3,3]-tcp[:3,3])-.1*x['confidence']);chosen=None
  if r.get('placement_target_base') is not None:
   preferred=np.asarray(r['placement_target_base'])
   rows.sort(key=lambda row:np.linalg.norm(np.asarray(row['contact'])-preferred)+.2*np.linalg.norm(np.asarray(row['tcp'])[:3,3]-tcp[:3,3])-.05*row['confidence'])
  from cross_episode_sim.control.observed_placement import supported_interior,unoccupied_contacts
  from cross_episode_sim.control.manipulation_approach import rank_centered_contacts
  # Use the observed vessel's horizontal radius plus a 3 cm edge margin.
  held_xy=held[:,:2];center=(held_xy.min(0)+held_xy.max(0))/2
  support_radius=float(np.linalg.norm(held_xy-center,axis=1).max()+.03)
  contacts=np.array([row['contact'] for row in rows]).reshape(-1,3)
  supported=supported_interior(surface,contacts,support_radius)
  supported &= unoccupied_contacts(contacts,points,support_radius,float(np.ptp(held[:,2])+.05))
  rows=[row for row,ok in zip(rows,supported) if ok]
  if a.observed_aperture or (not rows and a.semantic_receptacle and ('interior' in a.receptacle or 'inside' in a.receptacle)):
   from cross_episode_sim.control.aperture_placement import interior_placement_candidates
   rows=interior_placement_candidates(surface,points,held,tcp,support_radius)
   print('OBSERVED_CONTAINER_FALLBACK',len(rows),'not learned M2T2 predictions',flush=True)
  if active_mode=='front_entry':
   rows=front_placement_candidates(surface,points,held,tcp,approach_options.wrist_yaw_degrees,opening_normal=(opening_frame(surface,approach_aperture)[1] if approach_options.front_wrist=='horizontal' else None),opening_anchor=opening_frame(surface,approach_aperture)[0],wrist_roll_degrees=approach_options.wrist_roll_degrees,object_tilt_degrees=approach_options.object_tilt_degrees,tilt_outward=opening_frame(surface,approach_aperture)[1])
   print('ALIGNED_FRONT_CANDIDATES',len(rows),flush=True)
  if approach_options is not None and approach_options.placement_preference=='center':
   # A verified wrist turntable point avoids averaging a contradictory distant
   # camera hint into a fictitious center. All candidates already satisfy the
   # observed support/footprint/occupancy conditions above.
   preferred=(np.asarray(camera_hints['wrist']) if 'wrist' in camera_hints else
              np.median(np.asarray(preferred_contacts),axis=0) if preferred_contacts else np.median(surface,axis=0))
   pointed=preferred.copy()
   from cross_episode_sim.control.manipulation_approach import observed_support_center
   preferred=observed_support_center(surface,pointed,opening_frame(surface,approach_aperture)[1])
   rows=rank_centered_contacts(rows,preferred,CENTRE_TOLERANCE)
   artifacts.write_text(a.output/'center_preference.json',json.dumps(dict(
       target_base=preferred.tolist(),source='measured_horizontal_support_extents',semantic_point_base=pointed.tolist(),
       tolerance_m=CENTRE_TOLERANCE,retained=len(rows)),indent=2))
   print('CENTERED_PLACE_CANDIDATES',len(rows),'target',preferred.tolist(),flush=True)
  artifacts.write_text(a.output/'support_filter.json',json.dumps(dict(radius_m=support_radius,edge_margin_m=.03,retained=len(rows)),indent=2))
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
  if previous_contacts and approach_aperture is not None:
   # Same pose as this episode's earlier successful placement into this
   # opening (stored in the door's own frame), tried first: the new object
   # slides in on the same path and pushes the earlier one deeper.
   from cross_episode_sim.control.manipulation_approach import aperture_to_base
   wanted=[(aperture_to_base(approach_aperture,c['contact']),c.get('group')) for c in previous_contacts]
   def _same(row):
    return min((np.linalg.norm(np.asarray(row['contact'])[:2]-w[:2]) for w,g in wanted if g is None or g==row.get('group')),default=9.)
   near=[r for r in rows if _same(r)<.06];near.sort(key=_same)
   if near:
    diverse=near[:2]+[r for r in diverse if not any(r is n for n in near[:2])]
   print('SAME_POSE_FIRST',json.dumps(dict(previous=len(previous_contacts),matching_candidates=len(near))),flush=True)
  release_candidates=[]
  # Try distinct contacts/orientations before revisiting millimeter-offset
  # copies of the same pose. Retain every existing candidate and clearance.
  clearances=TILTED_RELEASE_CLEARANCES if approach_options is not None and approach_options.object_tilt_degrees else (.05,.03,.015,0.,.005,.01,.02,.04)
  # At least 100 points: bowls this episode placed showed 400-700 (fix83, fix86,
  # isolated); stray shelf-level points on an empty shelf must not trigger it.
  push_placed=(active_mode=='front_entry' and locals().get('placed_points') is not None and len(placed_points)>=PLACED_OBJECT_MIN_POINTS)
  if push_placed and LOW_PUSH_ENTRY:
   # Something this episode placed is already on the shelf. Slide the next
   # object in just above the shelf, not 7-9 cm up: at that height it rode onto
   # the first bowl's rim, tipped onto the open finger when released and came
   # back out with the hand (fix83). Low, its side pushes the placed one deeper.
   import dataclasses
   clearances=(.015,.03,.05,.005,.01,.02,.04)
   approach_options=dataclasses.replace(approach_options,entry_height=0.)
   print('LOW_PUSH_ENTRY',json.dumps(dict(placed_points=int(len(placed_points)),first_clearance_m=clearances[0])),flush=True)
  pairs=([(clearance,row) for row in diverse[:40] for clearance in clearances]
         if approach_options is not None and approach_options.object_tilt_degrees else
         [(clearance,row) for clearance in clearances for row in diverse[:40]])
  for clearance,row in pairs:
   candidate=dict(row);goal=np.array(row['tcp']);goal[2,3]+=clearance
   candidate.update(model_tcp=row['tcp'],tcp=goal.tolist(),release_clearance_m=clearance)
   release_candidates.append(candidate)
  if a.max_candidates>0:release_candidates=release_candidates[:a.max_candidates]
  blocked_spots=set()
  for candidate_index,row in enumerate(release_candidates,1):
   # A spot whose slide-in already failed is not retried at other release
   # heights (they share the same entry line; each retry cost ~6 s live).
   spot=(row.get('group'),*np.round(np.asarray(row['contact'])[:2]/.02).astype(int))
   if active_mode=='front_entry' and spot in blocked_spots:continue
   print('PLACE_ATTEMPT',candidate_index,'of',len(release_candidates),'clearance_m',row['release_clearance_m'],'contact',row['contact'],flush=True)
   goal=np.array(row['tcp'])
   try:
    if active_mode=='front_entry':
     plan_phase='front_entry'
     anchor,normal=opening_frame(surface,approach_aperture)
     envelope=np.concatenate([tool_points,gripper_points_tool(geo,q,tcp)])
     if a.align_only:
      from cross_episode_sim.control.manipulation_approach import plan_front_alignment
      approach,waypoints=plan_front_alignment(planner,q,tcp,goal,envelope,anchor,normal,approach_options)
      lower=approach[-1:];retreat=lower.copy()
     else:
      # Oracle release rule: the slide-in may stop short of this candidate's
      # contact once the object footprint is wholly over the observed support
      # and behind the opening plane (same tests the candidate itself passed).
      def release_supported(delta,_contact=np.asarray(row['contact'],float),_radius=float(row.get('support_radius_m',support_radius))):
       from cross_episode_sim.perception.grate_support import supported_interior
       contact=_contact+np.asarray(delta,float)
       inside=-(contact-np.asarray(anchor))@np.asarray(normal)>=_radius+.03
       return bool(inside and supported_interior(surface,contact[None],_radius)[0])
      approach,lower,retreat,waypoints=plan_front_entry(planner,q,tcp,goal,envelope,anchor,normal,approach_options,seed_contact_branches=approach_options.placement_preference=='center',retreat_planner=retreat_planner,release_supported=release_supported if approach_options.front_wrist=='horizontal' else None,settle_drop=float(row.get('release_clearance_m',0.)) if approach_options.front_wrist=='horizontal' else 0.)
     row=dict(row,approach_mode=active_mode,alignment_waypoints=[dict(phase=k,tcp=v.tolist()) for k,v in waypoints])
    elif active_mode=='top_down':
     plan_phase='approach';approach=planner.plan(q,goal[:3,3]+[0,0,max(.08,approach_options.entry_height)],goal[:3,:3]);plan_phase='lower';lower=planner.plan(approach[-1],goal[:3,3],goal[:3,:3]);plan_phase='retreat';retreat=lower[::-1].copy()
    else:
     plan_phase='approach';approach=planner.plan(q,goal[:3,3]+np.asarray(aperture['closed_normal'])*.18,goal[:3,:3]);plan_phase='lower';lower=planner.plan(approach[-1],goal[:3,3],goal[:3,:3]);plan_phase='retreat';retreat=lower[::-1].copy()
    # Exact reverse of the just-validated lower path in the same static world.
    # No second identical arm collision pass; payload validation below remains.
    # Held-object check on the approach to alignment only; the push in
    # (lower) is unchecked for front entry (user decision, Nov 16).
    for joints in (approach if (active_mode=='front_entry' and approach_options.front_wrist=='horizontal') else np.concatenate([approach,lower])):
     T=geo.fk(joints)[geo.tcp_body]@geo.tcp_local;xyz=tool_points[::max(1,len(tool_points)//150)]@T[:3,:3].T+T[:3,3]
     if not ObservedGraspPlanner.world_clear(planner,np.c_[xyz,np.full(len(xyz),.006)][None],margin=.002):raise RuntimeError('Held points collide on placement approach: '+str(getattr(planner,'last_rejection',None)))
    chosen=row
    if approach_aperture is not None:
     from cross_episode_sim.control.manipulation_approach import base_to_aperture
     chosen=dict(row,contact_aperture=base_to_aperture(approach_aperture,row['contact']))
    break
   except RuntimeError as e:
    events.append(plan_phase+': '+str(e));print('PLACE_REJECT',len(events),plan_phase,str(e),flush=True)
    if active_mode=='front_entry' and str(e).startswith(('insert:','entry:')):blocked_spots.add(spot)
  if chosen is None:raise RuntimeError('No validated M2T2 place path')
  if a.align_only:
   if active_mode!='front_entry':raise RuntimeError('Alignment preview requires front_entry')
   phase='align_outside_opening';execute(approach,255)
   artifacts.write_text(a.output/'chosen.json',json.dumps(chosen,indent=2))
   artifacts.write_text(a.output/'report.json',json.dumps(dict(alignment_completed=True,execution_success=False,approach_mode=active_mode,phase=phase,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall_start,release_executed=False,scope='alignment only; object remains held; not task success')))
   print('ALIGNMENT_COMPLETED',float(d.time-start),flush=True)
   raise SystemExit(0)
  selected_tcp=np.asarray(chosen['tcp'])
  artifacts.write_text(a.output/'chosen.json',json.dumps(chosen,indent=2));phase='approach';execute(approach,255);phase='lower';execute(lower,255)
  phase='release';last=None;stable=0
  for _ in range(100):
   command(lower[-1],0);angle=np.array([d.joint(n).qpos[0] for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')]);stable=stable+1 if last is not None and np.max(abs(angle-last))<.0002 else 0;last=angle
   if stable>=10:break
  if stable<10:raise RuntimeError('Release did not settle')
  # Release verification from the robot's own depth. Once the hand is 30 cm
  # from where it let go, nothing should remain where the held object was in
  # the tool frame. A bowl hooked on the open finger came out with the hand and
  # fell during the final lift (fix83) while the placement reported success.
  phase='retreat'
  _fk=lambda joints:(geo.fk(joints)[geo.tcp_body]@geo.tcp_local)
  _let_go=_fk(lower[-1])[:3,3]
  _split=next((i for i,cmd in enumerate(retreat) if np.linalg.norm(_fk(cmd)[:3,3]-_let_go)>=.30),len(retreat)-1)
  execute(retreat[:_split+1],0)
  phase='verify_released'
  from cross_episode_sim.control.sensor_appliance_backend import camera_points as _camera_points
  _tcp=_fk(c.robot_view.get_move_group('arm').joint_pos.copy())
  _expected=np.asarray(tool_points)@_tcp[:3,:3].T+_tcp[:3,3];_centre=_expected.mean(0)
  _radius=float(np.linalg.norm(_expected-_centre,axis=1).max())
  _seen=np.concatenate([_camera_points(camera)[0] for camera in sensors.capture().cameras.values()])
  _near=int((np.linalg.norm(_seen-_centre,axis=1)<.8*_radius).sum())
  print('RELEASE_CHECK',json.dumps(dict(points_at_gripper=_near,limit=STILL_AT_GRIPPER_POINTS,radius_m=round(.8*_radius,3),hand_travel_m=round(float(np.linalg.norm(_tcp[:3,3]-_let_go)),3))),flush=True)
  if _near>=STILL_AT_GRIPPER_POINTS:
   for _ in range(40):command(retreat[_split],255)   # hold what is still on the finger rather than let it fall
   raise RuntimeError('Released object still at the gripper after withdrawal')
  phase='retreat';execute(retreat[_split:],0)
  if active_mode=='front_entry' and locals().get('approach_aperture') is not None and 'cabinet' in (a.semantic_receptacle or '') and TIDY_PUSH_EXTRA>0:
   # Push what was just placed deeper with the CLOSED hand, so the next object
   # has free shelf at the front and can use the normal entry. A closed hand
   # has no finger inside the bowl and cannot drag it back out. (The two ways
   # of placing a second bowl beside/onto the first both failed: riding onto
   # its rim in fix83; hooked on the finger after the low entry in fix88/fix94.)
   phase='tidy_push';done=[]
   try:
    from cross_episode_sim.control.manipulation_approach import plan_aligned_translation
    from cross_episode_sim.control.contact_retreat import tool_pose
    shelf=float(np.median(surface[:,2]));pusher=retreat_planner or planner
    for _ in range(40):command(retreat[-1],255)
    execute(approach,255);done.append(approach)          # aligned pose outside the opening (already swept with the object held)
    entry=tool_pose(geo,approach[-1]);let_go=tool_pose(geo,lower[-1])
    inward=let_go[:3,3]-entry[:3,3];inward[2]=0.;inward/=np.linalg.norm(inward)
    low=entry.copy();low[2,3]=shelf+TIDY_PUSH_HEIGHT
    down=plan_aligned_translation(pusher,approach[-1],low,margin=None);execute(down,255);done.append(down)
    deep=low.copy();deep[:2,3]=let_go[:2,3]+TIDY_PUSH_EXTRA*inward[:2]
    push=plan_aligned_translation(pusher,down[-1],deep,accept_partial=lambda pose:True,push=True);resisted=False
    try:execute(push,255)
    except RuntimeError as exc:
     # The arm stopped short of the planned depth: the object met the back or
     # a side of the cabinet. That is the end of the push, not a failure.
     if 'did not settle' not in str(exc):raise
     resisted=True
    measured=c.robot_view.get_move_group('arm').joint_pos.copy()
    stop=int(np.argmin(np.abs(np.asarray(push)-measured).sum(axis=1)));done.append(push[:stop+1])
    reached=tool_pose(geo,measured)
    print('TIDY_PUSH',json.dumps(dict(hand_past_release_m=round(float((reached[:3,3]-let_go[:3,3])@inward),3),hand_above_shelf_m=TIDY_PUSH_HEIGHT,resisted=resisted)),flush=True)
   except RuntimeError as exc:
    print('TIDY_PUSH_SKIPPED',str(exc)[:160],flush=True)
   for segment in done[::-1]:execute(segment[::-1],255)
   for _ in range(40):command(retreat[-1],0)
  # Wait for observed vessel-region motion to settle, never a hidden contact
  # flag. The private support/stability scoring below remains unchanged.
  from cross_episode_sim.control.release_stability import MultiViewReleaseStability
  stability=MultiViewReleaseStability(chosen['contact'],support_radius)
  phase='visual_release_stability';visually_stable=False
  for settle_tick in range(100):
   command(retreat[-1],0)
   if settle_tick%5==0:
    observation=sensors.capture()
    visually_stable=stability.update(observation.cameras,observation.timestamp)
    if visually_stable:break
  artifacts.write_text(a.output/'release_stability.json',json.dumps(stability.events,indent=2))
  if not visually_stable:
   warnings.append('Visual settling inconclusive after 2s; semantic support scoring decides outcome')
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
  artifacts.write_text(a.output/'private_release_score.json',json.dumps(score_diagnostics,indent=2))
  from cross_episode_sim.evaluation.released_support import released_support_success
  success=released_support_success(score_diagnostics)
 except Exception as e:error=str(e);print(type(e).__name__,error,flush=True)
 frame()
grid.save(a.output);slam.save(a.output/'slam')
execution_success=error is None and phase=='score'
artifacts.write_text(a.output/'report.json',json.dumps(dict(approach_mode=locals().get("active_mode"),execution_success=execution_success,observed_release_stable=bool(locals().get('visually_stable',False)),success=success,phase=phase,error=error,warnings=warnings,wall_seconds=time.monotonic()-wall_start,video_status='deferred',simulated_seconds=float(d.time-start),planning_rejections=events,receptacle=a.receptacle,chosen_contact_aperture=(dict(contact=(locals().get('chosen') or {}).get('contact_aperture'),group=(locals().get('chosen') or {}).get('group')) if (locals().get('chosen') or {}).get('contact_aperture') is not None else None),scope='command execution; legacy vessel/table private diagnostic is not generic task scoring',isolated_initial_state=a.isolated_initial_state,source_docking_scope=r.get('scope','unspecified')),indent=2));print('PLACE_EXECUTION_COMPLETED',execution_success,'LEGACY_BOWL_TABLE_DIAGNOSTIC',success,'FULL_TASK_SCORE', 'pending',flush=True)
# Preserve the actual resulting physics state for the next pipeline stage.
# This is harness-only state, never exposed to perception or action selection.
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
artifacts.save_model(m,str(checkpoint/'model.mjb'),None)
spec=mujoco.mjtState.mjSTATE_INTEGRATION
state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
artifacts.save_array(checkpoint/'state.npy',state)
artifacts.write_text(a.output/'continuation.json',json.dumps(dict(source_dock=str(a.dock.resolve()),success=success),indent=2))
# Videos are rendered explicitly after execution, including failed attempts.
raise SystemExit(0 if (execution_success if __import__('os').environ.get('SENSOR_SKILL_EXECUTION')=='1' else success) else 1)
