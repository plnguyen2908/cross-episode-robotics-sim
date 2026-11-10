"""Live two-room closed-door exploration; private setup/scoring isolated below."""
import argparse,json,time,sys
from pathlib import Path
from contextlib import nullcontext
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.runtime.episode import current_runtime
from dataclasses import asdict
import numpy as np,mujoco
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.house_exploration import HouseExplorer,HouseExplorationConfig
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.observed_room_topology import ObservedRoomTopology
from cross_episode_sim.navigation.room_exploration_graph import RoomExplorationGraph
from cross_episode_sim.perception.observed_portals import ObservedPortalMap
from cross_episode_sim.perception.molmo_pointing import MolmoPointer
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.recording.deferred import StateVideoRecorder
from scripts.demo_local_scan import setup,plot_grid,ContactAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=120.);p.add_argument('--coverage',type=float,default=.9);p.add_argument('--map-resolution',type=float,default=.05,help='Match the production semantic-navigation pipeline');p.add_argument('--discovery-only',action='store_true');p.add_argument('--initial-doors',choices=('closed','open','saved'),default='closed')
    p.add_argument('--record-furniture-device',default=None)
    p.add_argument('--require-stations',action='store_true',help='Confirm kitchen/table labels before handing off; no extra table approach when already known')
    p.add_argument('--finish-station-approach',action='store_true',help='Finish the same exploration skill at the cached table stand-off')
    p.add_argument('--stop-after-crossing',action='store_true',help='Hand off the first opened passage to normal semantic station exploration')
    p.add_argument('--stop-after-opening',action='store_true',help='Component test: finish observed adjacent leaves, without attempting a crossing')
    p.add_argument('--single-left-leaf',action='store_true')
    p.add_argument('--handle-then-push',action='store_true',help='Test the staged handle pull and gripper/base panel push on both observed leaves')
    p.add_argument('--first-leaf',choices=('auto','left','right'),default='auto',help='Robot-view ordering for the paired-opening component comparison')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    # Reset only: close doors before evaluation. No geometry enters the policy.
    m,d,_=setup(a.source,preserve_initial_state=True)
    for j in range(m.njnt):
        if (m.joint(j).name or '').startswith('room_door_') and '_hinge_' in (m.joint(j).name or ''):
            if a.initial_doors=='saved':continue
            limits=m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]]=0. if a.initial_doors=='closed' else limits[np.argmax(np.abs(limits))]
            d.qvel[m.jnt_dofadr[j]]=0.
    mujoco.mj_forward(m,d)
    existing_runtime=current_runtime()
    runtime=existing_runtime or EpisodeRuntime(a.output);runtime.bind(m,d)
    private_audit=ContactAudit(m);private_base=m.body('robot_0/base').id
    private_initial=np.eye(4);private_initial[:3,:3]=d.xmat[private_base].reshape(3,3);private_initial[:3,3]=d.xpos[private_base]
    private_positions=[]
    grid=ObservedNavigationMap(MapConfig(resolution=a.map_resolution,height_resolution=a.map_resolution,pixel_stride=4 if a.map_resolution<.1 else 6));topology=ObservedRoomTopology();exploration_graph=RoomExplorationGraph(a.coverage)
    def new_explorer():
        result=HouseExplorer(grid,HouseExplorationConfig(max_seconds=a.seconds))
        result.frontier_priorities=lambda points:topology.uncovered_points(points,grid.config.resolution,a.coverage)
        result.frontier_priority=lambda points:topology.frontier_leaves_covered_rooms(points,grid.config.resolution,a.coverage)
        # Completion is measured by room coverage, not exhaustion of every
        # residual frontier pixel inside an already covered room.
        result.frontier_filter=lambda points:topology.uncovered_points(points,grid.config.resolution,a.coverage) if any(r.coverage is not None and r.coverage>=a.coverage for r in topology.evidence()) else np.ones(len(points),bool)
        return result
    explorer=new_explorer()
    from cross_episode_sim.perception.daaam_grounding import LocalVLM
    portals=ObservedPortalMap(MolmoPointer(),verifier=LocalVLM());bridge=OnboardBaseActuator(m,d)
    furniture=None
    if a.record_furniture_device:
        from cross_episode_sim.mapping.furniture import FurnitureMap
        from cross_episode_sim.perception.rpc import configured_client,RemoteSegmenter
        from daaam.config import SegmentationConfig
        from daaam.segmentation.services import SegmentationService
        client=configured_client('fastsam')
        segmenter=RemoteSegmenter(client) if client else SegmentationService(SegmentationConfig(model_name=str(Path('external/DAAAM/checkpoints/fastsam/FastSAM-s.pt').resolve()),model_config_path=None,device='cuda',min_mask_region_area=100))
        furniture=FurnitureMap(MolmoPointer(device=a.record_furniture_device),segmenter)

    runtime.observed_navigation_map=grid
    start=float(d.time);wall=time.monotonic();last_point=-999.;current=None;status='running';rows=[]
    from cross_episode_sim.control.live_door_open import LiveDoorOpener
    from cross_episode_sim.navigation.portal_approach import PortalApproach
    from cross_episode_sim.navigation.shaped_docking import RobotGeometry
    robot_geometry=RobotGeometry(m)
    opener=None if a.discovery_only else LiveDoorOpener(runtime,portals.pointer,a.output/'doors',handle_then_push=a.handle_then_push)
    from cross_episode_sim.navigation.crossing_updates import CrossingMapUpdates
    from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
    crossing_monitor=PlannedMotionMonitor()
    crossing_updates=CrossingMapUpdates()
    last_map_mode=None;door_map_dirty=False;paired_center=None;single_portal_id=None
    station_approach=None;coverage_complete=False;packet=None
    last_supported_room=None;last_room_position=None;last_progress_log=-np.inf
    approach=None;crossing=False;active_portal=None;opened_from=None;after_open=False;opening_results=[];last_decision=None;wrist_refresh_until=-np.inf;portal_grid=None;execution_error=None
    with (nullcontext(runtime) if existing_runtime else runtime.activate()),episode_sensors(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder,open(a.output/'progress.jsonl','w',buffering=1) as log:
        # Activate the session before asking the shared factory for a tracker.
        # Standalone component tests need the same control-aware session as
        # pipeline calls, rather than installing a raw RGBDMapper in runtime.
        mapper=create_mapper(fuse_dense=False)
        for tick in range(round(a.seconds*50)):
            if float(d.time-start)>=a.seconds:status='budget_exhausted';break
            if station_approach is not None:
                if packet is None or float(d.time)-packet.timestamp>=.1-1e-8:
                    packet=sensors.capture(include_robot_mask=False)
                    tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                command=station_approach.step(packet,tracking,float(d.time));status=command['status']
                if a.require_stations and not a.finish_station_approach and all(any(x['label']==label and x['observations']>=2 for x in furniture.landmarks) for label in ('kitchen','dining table')):
                    status='observed_topology_complete';break
                if status=='redock_needs_observation':
                    status='station_approach_unreachable';break
                if tick%50==0:print(json.dumps(dict(stage='finish_observed_station_approach',simulated_seconds=float(d.time-start),**command)),flush=True)
                bridge.step(command['forward'],command['yaw'],diagnostic_observer=private_audit.observe)
                recorder.record('station_approach')
                if station_approach.terminal:
                    if station_approach.terminal=='stations_explored':status='observed_topology_complete'
                    break
                continue
            if packet is None or float(d.time)-packet.timestamp>=.1-1e-8:
                packet=sensors.capture(include_robot_mask=False);tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                if furniture is not None and approach is None and not crossing:
                    furniture.observe(packet,tracking)
                from cross_episode_sim.navigation.room_map_freeze import covered_room_at,freeze_ready
                covered=covered_room_at(topology.rooms,tracking.T_map_base[:2,3],grid.config.resolution,a.coverage) if tracking.accepted else None
                # Keep localization, but stop reconstructing/relabeling a covered
                # room during approach. Door motion and entry into unknown space
                # explicitly resume sensor fusion; unknown cells never become free.
                map_frozen=freeze_ready(covered,initial_scan_complete=explorer.scans>0,crossing=crossing,changed_door=door_map_dirty or packet.timestamp<=wrist_refresh_until)
                local_move_frozen=bool(getattr(approach,'freeze_navigation_map',False))
                map_frozen=map_frozen or local_move_frozen
                mode=(map_frozen,covered)
                if mode!=last_map_mode:
                    print(json.dumps(dict(stage='exploration_map_mode',frozen=map_frozen,room=covered,reason='coverage_target_reached' if map_frozen else 'unknown_space_or_changed_door',coverage_target=a.coverage)),flush=True)
                    last_map_mode=mode
                local_door_update=covered is not None and active_portal is not None and (door_map_dirty or crossing)
                crossing_motion=getattr(approach,'crossing',None) if crossing else None
                if crossing_motion is not None and tracking.accepted:
                    # Once the camera approaches the opening, integrate the full
                    # newly visible room rather than clipping fusion to the door.
                    distance_to_plane=float((tracking.T_map_base[:2,3]-crossing_motion.center)@crossing_motion.normal)
                    if distance_to_plane>=-.4:local_door_update=False
                grid.fusion_region=(np.asarray(paired_center if paired_center is not None else active_portal['position'])[:2],2.) if local_door_update else None
                planned_crossing=crossing and getattr(getattr(approach,'crossing',None),'motion',None) is not None and approach.crossing.motion.follower is not None
                pose_for_fusion=np.r_[tracking.T_map_base[:2,3],np.arctan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0])] if tracking.accepted else None
                fuse_now=not planned_crossing or (tracking.accepted and crossing_updates.due(packet.timestamp,pose_for_fusion))
                if not map_frozen and fuse_now:
                    grid.update(packet.cameras['navigation'],tracking)
                    if planned_crossing:crossing_updates.record(packet.timestamp,pose_for_fusion)
                if tracking.accepted and not map_frozen and not planned_crossing and (door_map_dirty or packet.timestamp<=wrist_refresh_until or crossing):
                    grid.update_support(packet.cameras['wrist'],tracking,geometry=True)
                if tracking.accepted and portal_grid is not None:
                    portal_grid.update(packet.cameras['navigation'],tracking)
                    portal_grid.update_support(packet.cameras['wrist'],tracking,geometry=True)
                    support_pose=np.r_[tracking.T_map_base[:2,3],np.arctan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0])]
                    history=portal_grid.inherited_support_poses
                    if not history or np.linalg.norm(support_pose[:2]-history[-1][:2])>.025 or abs(np.arctan2(np.sin(support_pose[2]-history[-1][2]),np.cos(support_pose[2]-history[-1][2])))>.05:
                        history.append(support_pose)
                private_positions.append(d.xpos[private_base].copy())
                if tracking.accepted:
                    current=(covered or current) if (map_frozen or local_door_update) else topology.update(grid,tracking.T_map_base[:2,3])
                    if opened_from is not None and current!=opened_from:
                        exploration_graph.crossed(active_portal['portal'],opened_from,current)
                        active_portal['destination']=current
                        active_portal['reachable']=True
                        opened_from=None;after_open=False
                        explorer=new_explorer()
                    if not map_frozen and not local_door_update and packet.timestamp-last_point>=2.:
                        portals.observe(packet,tracking,current);last_point=packet.timestamp
                    if not map_frozen and not local_door_update:topology.associate_portals(portals.tracks,grid.config.resolution)
                    # A doorway may already be open, so there is no handle
                    # action to register the connection. The accepted trajectory
                    # itself supplies a bidirectional observed room edge.
                    room_evidence=topology.rooms.get(current,{}).get('evidence')
                    if room_evidence is not None and room_evidence.boundary_supported:
                        if last_supported_room is not None and current!=last_supported_room:
                            connection=portals.traversed(last_supported_room,current,last_room_position,tracking.T_map_base[:3,3])
                            print(json.dumps(dict(stage='observed_room_connection',source=last_supported_room,destination=current,portal=connection['portal'])),flush=True)
                        last_supported_room=current;last_room_position=tracking.T_map_base[:3,3].copy()
                    for track in portals.tracks:
                        if track.get('duplicate_of'):exploration_graph.retire_duplicate(track['portal'])
                    exploration_graph.observe(topology.evidence(),portals.evidence(current),current)
                if float(d.time)-last_progress_log>=1.-1e-8:
                    last_progress_log=float(d.time)
                    recorder.record_map(grid,tracking,phase=status)
                    row=dict(wall_seconds=time.monotonic()-wall,simulated_seconds=float(d.time-start),status=status,map_frozen=map_frozen,local_door_update=local_door_update,room=current,rooms=topology.summary(),portals=len(portals.tracks))
                    row.update(tracking_accepted=bool(tracking.accepted),estimated_base_xy=None if tracking.T_map_base is None else tracking.T_map_base[:2,3].tolist(),
                               decision=None if last_decision is None else asdict(last_decision))
                    
                    if approach is not None and tracking.T_map_base is not None:
                        row['estimated_base_xy']=tracking.T_map_base[:2,3].tolist()
                        row['estimated_base_yaw']=float(np.arctan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0]))
                        row['remaining_waypoints']=[v.tolist() for v in approach.follower.path]
                        row['approach_detail']=approach.follower.stop_detail
                        row['approach_events']=approach.follower.events[-3:]
                    rows.append(row);log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if approach is not None:
                command=approach.step(packet,tracking,float(d.time));status=command['status']
                if status=='portal_crossed':
                    door_map_dirty=False
                    if opened_from is not None and current==opened_from:
                        edge=exploration_graph.portals.get(active_portal['portal'])
                        known_destination=None
                        if edge is not None and edge.association_confirmed:
                            known_destination=edge.destination if edge.source==opened_from else edge.source if edge.destination==opened_from else None
                        current=topology.enter_after_crossing(opened_from,current,known_destination)
                        exploration_graph.observe(topology.evidence(),portals.evidence(current),current)
                        exploration_graph.crossed(active_portal['portal'],opened_from,current)
                        active_portal['destination']=current;active_portal['reachable']=True
                    opened_from=None;after_open=False
                    approach=None;crossing=False
                    portal_grid=None
                    explorer=new_explorer()
                    if a.stop_after_crossing:
                        bridge.step(0.,0.);status='opened_passage_ready';break
                    continue
                if status=='handle_view_ready':
                    bridge.step(0.,0.)
                    result=opener.open(sensors,tracking,active_portal['position'])
                    print(json.dumps(dict(stage='open_door',**result)),flush=True)
                    if not result['success'] and result.get('recoverable') and approach.retry_view(tracking):
                        print(json.dumps(dict(stage='handle_view_recovery',portal=active_portal['portal'],
                            attempt=len(approach.rejected_view_headings),reason=result['reason'])),flush=True)
                        continue
                    opening_results.append(result)
                    if not result['success']:status='opening_failed';break
                    door_map_dirty=True
                    opener.stow(sensors,recorder,explorer.config.arm_posture)
                    # The nested skill physically moved the base and arm.
                    # Never plan the next leaf/crossing from the pre-open packet,
                    # even when the outer loop's tick counter is between captures.
                    packet=sensors.capture(include_robot_mask=False)
                    tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
                    print(json.dumps(dict(stage='post_open_observation',timestamp=packet.timestamp,
                        tracking_accepted=bool(tracking.accepted),
                        estimated_base_xy=None if tracking.T_map_base is None else tracking.T_map_base[:2,3].tolist())),flush=True)
                    # The door moved: refresh near-field evidence from both
                    # onboard views during the existing scan, without idle waits.
                    wrist_refresh_until=float(d.time)+1.
                    portals.mark_open(active_portal,result.get('opened_handle_map'));opened_from=current;after_open=True
                    normal=np.asarray(result['portal_normal_map'])[:2];normal/=np.linalg.norm(normal)
                    active_portal['normal']=normal.tolist()
                    from cross_episode_sim.navigation.paired_door_approach import adjacent_closed_leaf
                    sibling=None if a.single_left_leaf else adjacent_closed_leaf(portals.tracks,active_portal,normal)
                    if sibling is not None:
                        paired_center=(np.asarray(active_portal['position'])+np.asarray(sibling['position']))/2
                        # Refresh once BEFORE planning the next local move;
                        # otherwise its first plan uses the old door map and
                        # later fusion changes the obstacles underneath it.
                        if tracking.accepted:
                            grid.fusion_region=(paired_center[:2],2.)
                            grid.update(packet.cameras['navigation'],tracking)
                            grid.update_support(packet.cameras['wrist'],tracking,geometry=True)
                        print(json.dumps(dict(stage='paired_leaf_standard_approach',opened=active_portal['portal'],next_leaf=sibling['portal'],crossing_deferred=True)),flush=True)
                        active_portal=sibling
                        grid.inherited_support_poses=[np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])] for T in mapper.trajectory]
                        # Use the same ordinary approach as for the first leaf.
                        # Its room-clearance route goes around the opened panel;
                        # the former local shortcut cut across that panel.
                        approach=PortalApproach(grid,sibling['position'],geometry=robot_geometry)
                        approach.follower._sync_support();explorer._sync_support()
                        approach.follower.support|=explorer.support
                        door_map_dirty=False;wrist_refresh_until=-np.inf
                        crossing=False;bridge=OnboardBaseActuator(m,d)
                        continue
                    if a.stop_after_opening:
                        status='observed_doors_open';bridge.step(0.,0.);break
                    from cross_episode_sim.navigation.portal_crossing import OpenedPortalHandoff
                    # Retain all observed geometry and measured support. Opening
                    # a panel changes occupancy, not the exploration objective.
                    grid.inherited_support_poses=[np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])] for T in mapper.trajectory]
                    topology.seams.append((active_portal['position'][:2].copy(),normal,1.))
                    portal_grid=None
                    approach=OpenedPortalHandoff(grid,robot_geometry,result.get('doorway_center_map') or active_portal['position'],normal,
                                                after_timestamp=float(d.time),refresh_view=False)
                    crossing=True;after_open=False
                    bridge=OnboardBaseActuator(m,d)
                    print(json.dumps(dict(stage='post_open_handoff',portal=active_portal['portal'],
                                          action='refresh_observed_map_then_cross',forced_scan=False,directed_view_turn=False)),flush=True)
                    continue
                if approach.follower.terminal or status=='handle_alignment_blocked':
                    if not crossing:
                        execution_error=dict(status=status,detail=approach.follower.stop_detail,
                                             events=approach.follower.events)
                        print(json.dumps(dict(stage='door_approach_failed',**execution_error)),flush=True)
                        exploration_graph.failed(active_portal['portal'],status)
                        break
                    active_portal['reachable']=False
                    exploration_graph.failed(active_portal['portal'],status)
                    print(json.dumps(dict(stage='portal_route_blocked',portal=active_portal['portal'],reason=status,detail=approach.follower.stop_detail,events=approach.follower.events)),flush=True)
                    # An opened but rejected passage is not a new frontier in
                    # the already covered room. Only a remaining closed leaf
                    # may justify another opening attempt; never restart scans.
                    if a.single_left_leaf:
                        status='single_leaf_crossing_failed';bridge.step(0.,0.);break
                    alternative=exploration_graph.decide(current,reachable_frontier=False,open_route_available=False)
                    if alternative.action not in ('open_door','navigate_portal'):
                        status='opened_portal_crossing_failed';bridge.step(0.,0.);break
                    approach=None;crossing=False;after_open=False;opened_from=None
                    portal_grid=None
                    explorer=new_explorer()
                    explorer.scans=1;explorer.phase='travel'
                    continue
            else:
                command=explorer.step(packet,tracking,float(d.time));status=command['status']
                if after_open and explorer.terminal is not None:
                    # Exhaustion is not completion: keep this portal unresolved
                    # but permit another observed passage/leaf to be considered.
                    # Frontier exhaustion under circular inflation must not
                    # suppress the shape-checked open-door handoff below.
                    if not (active_portal is not None and active_portal.get('normal') is not None):
                        after_open=False
                # Finish the initial scan and ask the frontier planner once.
                # A live explorer is not evidence of an open route to a NEW room.
                if explorer.scans>0 and explorer.phase=='travel' and not explorer.path and explorer.terminal is None:
                    error=explorer._plan(tracking.T_map_base[:2,3],float(d.time)) if tracking.accepted else None
                    if error:explorer.terminal=error
                if after_open and explorer.scans>0 and active_portal is not None and active_portal.get('normal') is not None:
                    from cross_episode_sim.navigation.portal_crossing import PortalCrossing
                    approach=PortalCrossing(portal_grid if portal_grid is not None else grid,robot_geometry,active_portal['position'],active_portal['normal'],tracking.T_map_base[:2,3],transit_grid=grid)
                    crossing=True;after_open=False;opened_from=current
                    continue
                decision=exploration_graph.decide(current,reachable_frontier=explorer.terminal is None,open_route_available=explorer.terminal is None and explorer.goal_leaves_covered_room)
                if decision!=last_decision:
                    print(json.dumps(dict(stage='exploration_decision',simulated_seconds=float(d.time-start),**asdict(decision),
                        edges=exploration_graph.snapshot()['edges'])),flush=True)
                    last_decision=decision
                if after_open and explorer.scans==0 and explorer.terminal is None:
                    decision=None
                if decision is not None and decision.action in ('open_door','navigate_portal') and explorer.scans>0:
                    if a.discovery_only:status='door_open_required';break
                    active_portal=next(t for t in portals.tracks if t['portal']==decision.target)
                    if a.first_leaf!='auto' and not opening_results:
                        from cross_episode_sim.navigation.paired_door_approach import observed_side_leaf
                        active_portal=observed_side_leaf(portals.tracks,active_portal,tracking.T_map_base[:2,3],a.first_leaf)
                    if a.single_left_leaf:
                        from cross_episode_sim.navigation.paired_door_approach import left_observed_leaf
                        if single_portal_id is None:
                            active_portal=left_observed_leaf(portals.tracks,active_portal,tracking.T_map_base[:2,3])
                            single_portal_id=active_portal['portal']
                            print(json.dumps(dict(stage='single_left_leaf_selected',portal=single_portal_id,reference='left when facing observed doorway',other_leaf_kept_closed=True)),flush=True)
                        else:active_portal=next(t for t in portals.tracks if t['portal']==single_portal_id)

                    if decision.action=='navigate_portal':
                        # An open passage is explored geometrically, never grasped.
                        edge=exploration_graph.portals[decision.target]
                        destination=edge.destination if edge.source==current else edge.source
                        room=topology.rooms.get(destination)
                        if edge.association_confirmed and room and room['footprint']:
                            from cross_episode_sim.navigation.room_approach import RoomApproach
                            approach=RoomApproach(grid,room['footprint'],grid.config.resolution)
                            crossing=True;opened_from=current
                        else:
                            # Keep the active frontier path. Selecting the same
                            # open portal must not restart a 360-degree scan.
                            after_open=True
                        if approach is not None:continue
                    if decision.action=='open_door':
                        approach=PortalApproach(grid,active_portal['position'],geometry=robot_geometry)
                        approach.follower._sync_support();explorer._sync_support();approach.follower.support|=explorer.support
                        print(json.dumps(dict(stage='approach_door',portal=active_portal['portal'],room=current)),flush=True)
                        continue
                if decision is not None and decision.action=='observed_topology_complete' and explorer.scans>0:
                    coverage_complete=True
                    if not (a.finish_station_approach or a.require_stations):status=decision.action;break
                    if furniture is None:
                        status='missing_observed_station';break
                    stations_confirmed=all(any(x['label']==label and x['observations']>=2 for x in furniture.landmarks) for label in ('kitchen','dining table'))
                    if stations_confirmed and not a.finish_station_approach:
                        status='observed_topology_complete';break
                    from cross_episode_sim.navigation.station_exploration import StationExplorer
                    from cross_episode_sim.mapping.continuity import confirmed_furniture_tables
                    from cross_episode_sim.navigation.fixed_map import freeze_map
                    station_approach=StationExplorer(grid,furniture,config=explorer.config,preferred_distance=1.1)
                    station_approach.targets.targets=confirmed_furniture_tables(furniture.landmarks) if stations_confirmed else []
                    station_approach.explorer._sync_support();explorer._sync_support()
                    station_approach.explorer.support|=explorer.support
                    station_approach.allow_exploration=not stations_confirmed
                    if stations_confirmed:
                        station_approach.observe=lambda packet,tracking:None
                        freeze_map(grid)
                    bridge.motion_guard=None
                    print(json.dumps(dict(stage='coverage_complete',action='finish_cached_station_approach' if stations_confirmed else 'complete_missing_station_labels',new_semantic_queries=not stations_confirmed,map_rebuild=False)),flush=True)
                    continue
                if decision is not None and decision.action=='blocked' and explorer.scans>0:
                    status=decision.reason;break
                if explorer.terminal:break
            # A shape-planned crossing uses the same execution contract as
            # fixed-map transit, not the generic nominal-tuck angle veto.
            motion=getattr(getattr(approach,'crossing',approach),'motion',None) if crossing else None
            if motion is not None and motion.follower is not None and motion.error is None:
                bridge.motion_guard=lambda f,w,dt:crossing_monitor.check(packet,tracking,float(d.time),f,w)
            else:
                bridge.motion_guard=None
            try:bridge.step(command['forward'],command['yaw'],diagnostic_observer=private_audit.observe)
            except RuntimeError as exc:
                execution_error=dict(message=str(exc),monitor=crossing_monitor.detail,command=command)
                print(json.dumps(dict(stage='base_execution_rejected',**execution_error)),flush=True)
                status='base_execution_rejected';break
            recorder.record(status)
        if status=='observed_topology_complete':
            for _ in range(100):
                if bridge.stationary():break
                bridge.step(0.,0.,diagnostic_observer=private_audit.observe)
                recorder.record('exploration_settle')
            packet=sensors.capture(include_robot_mask=False)
            tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
        graph_snapshot=exploration_graph.snapshot()
        for node in graph_snapshot['nodes']:
            cells=topology.rooms.get(node['room'],{}).get('footprint',[])
            if cells:node['center_xy']=((np.mean(list(cells),axis=0)+.5)*grid.config.resolution).tolist()
        (a.output/'exploration_graph.json').write_text(json.dumps(graph_snapshot,indent=2))
        if furniture is not None:
            from cross_episode_sim.mapping.continuity import save_memory,confirmed_furniture_tables
            save_memory(a.output,dict(rooms=[],tables=[],furniture=furniture.landmarks),targets=confirmed_furniture_tables(furniture.landmarks))
            (a.output/'furniture_diagnostics.json').write_text(json.dumps(dict(landmarks=furniture.landmarks,events=furniture.events),indent=2))
        mapper.save(a.output/'slam')
        report=dict(opening_complete=status=='observed_doors_open',passage_ready=status=='opened_passage_ready',status=status,success=status=='observed_topology_complete',rooms=topology.summary(),portals=[dict(t,position=t['position'].tolist()) for t in portals.tracks],pointing_events=portals.events,wall_seconds=time.monotonic()-wall,simulated_seconds=float(d.time-start),opening_results=opening_results,limitations='Estimated sensor room coverage; physical whole-house qualification requires private scoring')
        if execution_error is not None:report['execution_error']=execution_error
        if furniture is not None:
            from cross_episode_sim.navigation.route_memory import record_exploration
            report['route_graph_seed']=record_exploration(mapper,furniture.landmarks,report['success'],destination_ready=bool(a.finish_station_approach and station_approach is not None and station_approach.terminal=='stations_explored'))
            print(json.dumps(dict(stage='exploration_route_graph',**report['route_graph_seed'])),flush=True)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
        artifacts.save_model(m,str(checkpoint/'model.mjb'),None)
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec))
        mujoco.mj_getState(m,d,state,spec);artifacts.save_array(checkpoint/'state.npy',state)
        packet=sensors.capture(include_robot_mask=False);from PIL import Image
        Image.fromarray(packet.cameras['navigation'].rgb).save(a.output/'last_navigation.png')
    from scripts.demo_house_exploration import score
    report['private_room_scores']=score(a.output,a.source,np.asarray(private_positions),private_initial,grid)
    report['private_navigation_contacts']=list(private_audit.events.values())
    report['private_door_angles_degrees']={
        m.joint(j).name:float(np.degrees(d.qpos[m.jnt_qposadr[j]]))
        for j in range(m.njnt)
        if (m.joint(j).name or '').startswith('room_door_') and '_hinge_' in (m.joint(j).name or '')}
    report['physical_opening_test_passed']=bool(
        report['opening_complete'] and len(opening_results)>=2
        and sum(abs(angle)>=80. for angle in report['private_door_angles_degrees'].values())>=len(opening_results)
        and not private_audit.events)
    report['physical_test_passed']=bool(report['success'] and len(report['rooms'])>=len(report['private_room_scores']) and all(r['interior_visit_seconds']>0 for r in report['private_room_scores']) and not private_audit.events)
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    grid.save(a.output)
    if not existing_runtime:mapper.save(a.output/'slam')
    if not existing_runtime:plot_grid(a.output,'Closed-door exploration')
    np.savez_compressed(a.output/'wall_evidence.npz',evidence=grid.evidence,floor=grid.floor_views,origin=grid.origin,resolution=grid.config.resolution)
    from scripts.plot_exploration_decisions import plot
    if not existing_runtime:plot(a.output)
    return 0 if report['success'] or report['passage_ready'] or report['opening_complete'] else 1
if __name__=='__main__':
    from cross_episode_sim.runtime.logging import TimestampedStream
    if not isinstance(sys.stdout,TimestampedStream):sys.stdout=TimestampedStream(sys.stdout)
    if not isinstance(sys.stderr,TimestampedStream):sys.stderr=TimestampedStream(sys.stderr)
    raise SystemExit(main())
