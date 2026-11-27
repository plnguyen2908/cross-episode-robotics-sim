"""Two-room sensor-only navigation test; no manipulation or oracle destinations."""
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.recording.deferred import StateVideoRecorder
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext
import cv2
import numpy as np
from tqdm import tqdm
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.mapping.hydra_mapping import HydraFrameWriter
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
from cross_episode_sim.perception.point_masks import PointMaskDetector
from scripts.demo_local_scan import setup,ContactAudit,plot_grid


def main(*, docking_factory=None, pointing_factory=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preserve-initial-state',action='store_true',help='Restore exact recorded state without resetting doors or arm posture')
    p.add_argument('--continuation',type=Path,help='Harness checkpoint and observed map from the preceding physical stage; no scene reset')
    p.add_argument('--fixed-map',action='store_true',help='Use a frozen continuation map; request exploration for unknown space')
    p.add_argument('--target',default='cup',help='Semantic object/fixture description; never a simulator ID')
    p.add_argument('--preferred-distance',type=float,default=.55,help='Preferred horizontal base-to-object distance in metres; collision clearance unchanged')
    p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--object-map',choices=['hydra','visual'],default='hydra')
    p.add_argument('--initial-backoff',type=float,default=0.,help='Standalone reset offset before evaluation; never a policy action')
    p.add_argument('--initial-yaw-offset',type=float,default=0.,help='Radians applied at standalone reset before evaluation')
    p.add_argument('--perception',choices=['daaam','point-mask'],default='daaam')
    p.add_argument('--vlm-model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--vlm-device',default='cuda',help='Local verifier device, e.g. cuda:1 with two visible GPUs')
    p.add_argument('--vlm-url',default=None)
    p.add_argument('--explorer',choices=['frontier','tare'],default='tare')
    p.add_argument('--navigation-mode',choices=['semantic','hybrid'],default='semantic')
    p.add_argument('--ring-checkpoint',type=Path)
    p.add_argument('--pointing-device',default=None,help='Enable experimental Molmo2 supervised handoff on this device')
    p.add_argument('--ring-device',default='cuda:0')
    p.add_argument('--station-exploration',action='store_true',help='Explore using only cached kitchen and dining-table landmarks')
    p.add_argument('--no-cabinet-survey',action='store_true',help='Skip the whole-kitchen cabinet survey (door already opened by this robot)')
    p.add_argument('--max-target-distance',type=float,default=None,help='Only targets within this distance (m) of where the search starts (default: any)')
    p.add_argument('--route-station',default=None,help='Semantic station for a successful local approach in the episode route graph')
    p.add_argument('--record-furniture-device',default=None,help='Label observed furniture during exploration using Molmo2 on this device')
    p.add_argument('--record-room-memory',action='store_true',help='Record visually confirmed room labels during empty-hand exploration')
    p.add_argument('--goal-room',default='kitchen',help='Visual room category; empty string skips room search for local tests')
    p.add_argument("--map-resolution",type=float,default=.10,help="Observed collision-map voxel size in metres")
    p.add_argument('--memory-index',type=Path,default=None,help='With a confident visual memory (not log-only): build the semantic memory from this memory index and adopt the memory tour map, replacing exploration')
    p.add_argument('--visual-memory-log-only',action='store_true',help='With --visual-memory: only log the relocalization and always explore fully. Default: when the first-turn relocalization is confident (fit <= MEMORY_FIT_GATE) exploration stops and hands over the memory map; otherwise it explores normally and the memory is not used.')
    p.add_argument('--visual-memory',type=Path,default=None,help='Optional saved visual memory: a merged map folder (has graph.json) or a memory root with merged/<house12>_<task>. During the first full turn the robot matches its views against it and logs its pose in the memory frame; motion is unchanged. Default: none')
    a=p.parse_args()
    if a.navigation_mode=='hybrid' and (a.perception!='daaam' or a.vlm_url):
        p.error('Hybrid navigation requires DAAAM and local Qwen')
    if a.object_map=='hydra' and (a.target not in ('cup','bowl') or a.perception!='daaam'):
        p.error('Hydra object navigation requires DAAAM masks and a cup/bowl target; use --object-map visual for the ablation.')
    if a.preserve_initial_state and (a.initial_backoff or a.initial_yaw_offset):
        p.error('Saved-state evaluation forbids spawn offsets')
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    if a.station_exploration:
        if a.navigation_mode!='semantic' or a.object_map!='visual' or not a.record_furniture_device or a.record_room_memory:
            p.error('Station exploration requires semantic/visual mode, furniture device, and no room labeling')
        from types import SimpleNamespace
        from daaam.config import SegmentationConfig
        from daaam.segmentation.services import SegmentationService
        from cross_episode_sim.perception.rpc import configured_client,RemoteSegmenter
        segmentation_client=configured_client('fastsam')
        detector=SimpleNamespace(segmenter=RemoteSegmenter(segmentation_client) if segmentation_client else SegmentationService(SegmentationConfig(model_name=str(Path('external/DAAAM/checkpoints/fastsam/FastSAM-s.pt').resolve()),model_config_path=None,device='cuda',min_mask_region_area=100)))
    elif a.perception=='daaam':
        from cross_episode_sim.perception.daaam_grounding import DaaamGrounder,LocalVLM,EndpointVLM
        vlm=EndpointVLM(a.vlm_url,a.vlm_model) if a.vlm_url else LocalVLM(a.vlm_model,device=a.vlm_device)
        categories=('cup','bowl') if a.target in ('cup','bowl') else (a.target,)
        detector=DaaamGrounder(vlm,'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',a.output/'grounding',categories=categories)
    else:detector=PointMaskDetector(categories=tuple(dict.fromkeys(('cup','bowl',a.target))))
    if a.continuation:
        if a.initial_backoff or a.initial_yaw_offset:p.error('Continuation forbids spawn offsets')
        import mujoco
        model,data=load_checkpoint(a.continuation);doors=[]
    else:
        model,data,doors=setup(a.source,a.initial_backoff,preserve_initial_state=a.preserve_initial_state)
    if a.initial_yaw_offset:
        import mujoco
        data.joint('robot_0/base_theta').qpos[0]+=a.initial_yaw_offset
        mujoco.mj_forward(model,data)
    # Private evaluation origin; never passed to any navigation component.
    private_initial=np.eye(4);private_bid=model.body('robot_0/base').id
    private_initial[:3,:3]=data.xmat[private_bid].reshape(3,3).copy();private_initial[:3,3]=data.xpos[private_bid].copy()
    base=OnboardBaseActuator(model,data);audit=ContactAudit(model)
    if a.fixed_map and not a.continuation:raise ValueError('--fixed-map requires --continuation')
    slam=create_mapper(fuse_dense=False);grid=ObservedNavigationMap(MapConfig(extent_m=12.,resolution=a.map_resolution,height_resolution=a.map_resolution,pixel_stride=4 if a.map_resolution<.1 else 6));config=HouseExplorationConfig(max_seconds=a.seconds)
    if a.continuation:
        from cross_episode_sim.navigation.map_continuation import restore_observed_map
        restore_observed_map(grid,a.continuation)
    from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest
    frozen_digest=freeze_map(grid) if a.fixed_map else None
    from cross_episode_sim.mapping.continuity import load_memory,save_memory,object_hypotheses
    memory=load_memory(a.continuation) if a.continuation else dict(rooms=[],tables=[])
    planner_context=nullcontext();explorer=None
    if a.explorer=='tare':
        from cross_episode_sim.navigation.tare import TarePlanner
        from cross_episode_sim.navigation.tare_exploration import TareExplorer
        planner_context=TarePlanner(a.output/'tare');explorer=TareExplorer(grid,planner_context,config)
    from cross_episode_sim.mapping.multiview_hydra import MultiCameraHydraProcess
    mapper_context=MultiCameraHydraProcess(a.output/'native_hydra') if a.object_map=='hydra' else nullcontext()
    policy=SemanticNavigator(grid,detector,a.target,explorer,config,object_mapper=mapper_context if a.object_map=='hydra' else None,preferred_distance=a.preferred_distance)
    policy.remembered_object_hypotheses=object_hypotheses(memory)
    policy.remembered_furniture=memory.get('furniture',[])
    policy.search_station=a.route_station
    ring_context=nullcontext();motion_policy=policy
    if a.navigation_mode=='hybrid':
        from cross_episode_sim.navigation.onering import OneRingProcess,OneRingLocalNavigator
        from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
        from cross_episode_sim.mapping.room_memory import ObservedRoomMemory
        from cross_episode_sim.navigation.readiness import ManipulationReadiness
        ring_context=OneRingProcess(a.ring_checkpoint,a.output/'onering',device=a.ring_device)
        local=OneRingLocalNavigator(grid,ring_context,f'Find a {a.target}',config)
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry,ShapedDocking
        from cross_episode_sim.navigation.workspace_docking import episode_workspace_ranker
        ranker=episode_workspace_ranker(model,data,a.output/'workspace_robot')
        # Keep the baseline constructor compatible with persistent runtimes
        # that loaded ShapedDocking before the optional ranker was installed.
        factory=ShapedDocking if docking_factory is None else docking_factory
        docking=(factory(grid,RobotGeometry(model)) if ranker is None else
                 factory(grid,RobotGeometry(model),workspace_ranker=ranker))
        # Rank only within the validated execution envelope; a robot-only IK
        # margin does not establish actual object approach feasibility.
        readiness=ManipulationReadiness(distance_band=(.50,.72))
        motion_policy=HybridNavigator(policy,local,ObservedRoomMemory(vlm,remembered=memory['rooms']),readiness,a.goal_room,docking=docking)
        motion_policy.max_target_distance=a.max_target_distance
        motion_policy.fixture_handle=a.target.startswith('handle on ')
        motion_policy.confined_target=a.target.endswith((' inside fridge',' inside microwave'))
    if a.pointing_device:
        if a.navigation_mode!='hybrid':raise ValueError('Pointing supervisor requires hybrid navigation')
        from cross_episode_sim.perception.molmo_pointing import MolmoPointer,PointingSupervisor
        from cross_episode_sim.perception.vessel_verification import VESSELS,VerifiedVesselSupervisor
        # Molmo2 confuses bowls with cups standing next to them; confirm each pointed vessel.
        default_supervisor=VerifiedVesselSupervisor if a.target in VESSELS else PointingSupervisor
        motion_policy.pointing=(pointing_factory or default_supervisor)(MolmoPointer(device=a.pointing_device),detector.segmenter,a.target)
    survey_rooms=None
    if a.record_room_memory:
        if a.perception!='daaam':raise ValueError('Room memory requires the local VLM')
        from cross_episode_sim.mapping.room_memory import ObservedRoomMemory
        survey_rooms=ObservedRoomMemory(vlm,remembered=memory['rooms'])
    furniture=None
    if a.record_furniture_device:
        if a.fixed_map:raise ValueError('Furniture discovery belongs to exploration, not frozen-map execution')
        from cross_episode_sim.mapping.furniture import FurnitureMap
        from cross_episode_sim.perception.molmo_pointing import MolmoPointer
        furniture=FurnitureMap(MolmoPointer(device=a.record_furniture_device),detector.segmenter,memory.get('furniture',[]))
    if a.station_exploration:
        from cross_episode_sim.navigation.station_exploration import StationExplorer
        policy=StationExplorer(grid,furniture,explorer,config,a.preferred_distance)
        motion_policy=policy
    start=float(data.time);wall=time.monotonic();truth=[];last_status=None;last_saved_detection=None
    # A fresh tracker has no map and needs a first view; a continuation resumes
    # one and only looks back for a stored view after losing tracking.
    from cross_episode_sim.navigation.startup_look import StartupLook
    startup_look=StartupLook(tracked=bool(a.continuation))
    from cross_episode_sim.navigation.cabinet_survey import CabinetSurvey
    survey=CabinetSurvey(a.target) if (CabinetSurvey.applies(a.target) and getattr(motion_policy,'pointing',None) is not None and not a.no_cabinet_survey) else None
    (a.output/'detections').mkdir()
    # Navigation needs an occupancy map, not diagnostic 3D scene exports.
    hydra_frames=None
    artifacts.write_text(a.output/'configuration.json',json.dumps(dict(station_exploration=a.station_exploration,dense_3d_reconstruction=False,navigation_representation='2d_occupancy_with_height_clearance',source=str(a.source),preserve_initial_state=a.preserve_initial_state,navigation_mode=a.navigation_mode,goal_room=a.goal_room,ring_checkpoint=str(a.ring_checkpoint) if a.ring_checkpoint else None,ring_device=a.ring_device,target=a.target,preferred_distance=a.preferred_distance,object_map=a.object_map,explorer=a.explorer,perception=a.perception,proposal_backend='molmo2_fastsam',vlm_model=a.vlm_model,vlm_device=a.vlm_device,seconds=a.seconds,initial_backoff=a.initial_backoff,initial_yaw_offset=a.initial_yaw_offset,doors_initialized_open=doors),indent=2))
    # Station exploration is qualified on matched keyframe poses; weak-match refinement resumes for later stages.
    keyframe_context=slam.matched_keyframes() if a.station_exploration and hasattr(slam,'matched_keyframes') else nullcontext()
    relocalization=None;memory_localized=False;relocalization_attempts=[];RELOCALIZATION_ATTEMPTS=3
    if a.visual_memory is not None:
        from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization,memory_directory
        from cross_episode_sim.evaluation.text_memory import house_of
        house_id,_=house_of(a.source);task=json.loads((a.source/'config.json').read_text()).get('task') if (a.source/'config.json').exists() else None
        relocalization=ScanRelocalization(memory_directory(a.visual_memory,house_id,task))
    with keyframe_context,ring_context,mapper_context,planner_context,OnboardSensorInterface(model,data) as sensors,StateVideoRecorder(model,data,a.output/'state_recording') as recording,open(a.output/'progress.jsonl','w',buffering=1) as log:
        for tick in tqdm(range(round(a.seconds*50)+10),desc='Visual target navigation'):
            if tick%5==0:
                packet=sensors.capture(include_robot_mask=False);tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
                if relocalization is not None and not relocalization.done and relocalization.observe(packet.cameras['navigation'],tracking,float(packet.timestamp-start)):
                    relocalization.save(a.output);tqdm.write(json.dumps(dict(visual_relocalization={k:relocalization.result.get(k) for k in ('status','decision','fine_fit_mean_capped_distance_m','position_in_memory','yaw_in_memory_degrees','views','wall_seconds')})))
                    relocalization_attempts.append({k:relocalization.result.get(k) for k in ('status','decision','fine_fit_mean_capped_distance_m','views','trigger')})
                    if relocalization.result.get('use_memory') and not a.visual_memory_log_only:
                        memory_localized=True;break  # confident: the memory map replaces the rest of exploration
                    if len(relocalization_attempts)<RELOCALIZATION_ATTEMPTS and not relocalization.result.get('use_memory'):
                        # not confident: keep exploring and match the next full turn of views (station turns)
                        relocalization=ScanRelocalization(relocalization.merged)
                memory_scan=(relocalization is not None and not relocalization.done and not a.visual_memory_log_only
                             and len(relocalization_attempts)<RELOCALIZATION_ATTEMPTS)
                # during the memory scan the live map / room / furniture layers are not built: a confident match
                # replaces them with the memory's map and semantics (they start only if the memory fails)
                if not a.fixed_map and not memory_scan:
                    grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
                if survey_rooms is not None and not memory_scan:survey_rooms.update(packet,tracking,[])
                if furniture is not None and not memory_scan:furniture.observe(packet,tracking)
            if survey is not None and not survey.done and tick%5==0:
                command=survey.step(dict(status='cabinet_survey',forward=0.,yaw=0.),packet,tracking,float(data.time),motion_policy.pointing,motion_policy)
            elif survey is not None and not survey.done:
                command=dict(command,forward=0.,yaw=survey.velocity)
            elif relocalization is not None and not relocalization.done and not a.visual_memory_log_only and len(relocalization_attempts)<RELOCALIZATION_ATTEMPTS:
                # memory first: turn in place and match the memory's views; the exploration engine (travel,
                # planning) only starts if the memory cannot be matched confidently (run 20: the engine's travel
                # during the scan cost ~95 s of a 2 min explore while matching itself took ~3 s)
                command=dict(status='memory_relocalization_scan',forward=0.,yaw=.8)
            else:
                command=motion_policy.step(packet,tracking,float(data.time))
            command=startup_look.step(command,packet,tracking,float(data.time))
            if tick%5==0:
                row=dict(time=float(data.time-start),mode=motion_policy.mode,object_search_success=getattr(motion_policy,'object_search_success',False),**command,estimated_position=tracking.T_map_base[:2,3].tolist() if tracking.accepted else None,targets=[dict(id=t.track_id,label=t.label,position=t.position.tolist(),observations=t.observations) for t in policy.targets.targets])
                log.write(json.dumps(row)+'\n');truth.append(data.xpos[model.body('robot_0/base').id].copy())
                recording.record_map(grid,tracking,motion_policy.mode)
                if policy.last_detection!=last_saved_detection and packet.timestamp==policy.last_detection:
                    last_saved_detection=policy.last_detection
                    if hydra_frames is not None:
                        hydra_frames.update(packet.cameras['navigation'],tracking,policy.last_detections.get('navigation',[]),packet.timestamp)
                if command['status']!=last_status or tick%250==0:tqdm.write(json.dumps(row));last_status=command['status']
            base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
            if motion_policy.terminal and base.stationary():break
    if relocalization is not None and not relocalization.done:
        relocalization.finish('exploration_ended_before_full_turn');relocalization.save(a.output)
    # Private checkpoint for reproducible continuation, never a policy observation.
    import mujoco
    checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
    artifacts.save_model(model,str(checkpoint/'model.mjb'),None)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(model,spec))
    mujoco.mj_getState(model,data,state,spec);artifacts.save_array(checkpoint/'state.npy',state)
    memory_world=None
    if memory_localized and relocalization is not None:
        # exploration replaced by memory: SLAM poses in the memory map frame, the tour map as the observed map
        from cross_episode_sim.navigation.memory_navigation import adopt_memory_frame
        memory_world=adopt_memory_frame(slam,relocalization.result['T_memory_from_slam'],grid,relocalization.merged)
        print('MEMORY_FRAME_ADOPTED',json.dumps(memory_world),flush=True)
    grid.save(a.output);slam.save(a.output/'slam')
    if artifacts.owner(a.output) is None:plot_grid(a.output,'Partial sensor map and target navigation')
    if furniture is not None:
        memory['furniture']=furniture.landmarks
        artifacts.write_text(a.output/'furniture_observations.json',json.dumps(furniture.events,indent=2))
    if memory_world is not None and a.memory_index is not None:
        from cross_episode_sim.navigation.memory_navigation import memory_semantics
        from cross_episode_sim.evaluation.text_memory import house_of
        memory['tables'],memory['furniture']=memory_semantics(Path(a.memory_index)/f'{house_of(a.source)[0][:12]}.json')
    save_memory(a.output,memory,room_events=getattr(survey_rooms or getattr(motion_policy,'rooms',None),'events',()),targets=policy.targets.targets)
    artifacts.save_array(a.output/'diagnostic_positions.npy',truth)
    selected=getattr(motion_policy,'found_target',None) or policy.follower.target
    handoff=None if selected is None else dict(native_id=selected.native_id,label=selected.label,position=selected.position.tolist(),frame='estimated_slam_map',last_observed=selected.last_seen,grasp_ready=False)
    report=dict(memory_handover=None if not memory_localized else dict(memory_directory=relocalization.result.get('memory_directory'),T_memory_from_slam=(np.eye(4).tolist() if memory_world is not None else relocalization.result.get('T_memory_from_slam')),frame_adopted=memory_world is not None,fit_m=relocalization.result.get('fine_fit_mean_capped_distance_m')),cabinet_survey=None if survey is None else survey.summary,startup_look=startup_look.summary(),pointing_events=getattr(getattr(motion_policy,'pointing',None),'events',[]),fixed_map_unchanged=map_digest(grid)==frozen_digest if a.fixed_map else None,exploration_request=command.get('exploration_request'),selected_object=handoff,status='memory_localized' if memory_localized else motion_policy.terminal,target=a.target,object_map=a.object_map,simulated_seconds=float(data.time-start),wall_seconds=time.monotonic()-wall,
        memory_localized=memory_localized,
        sensor_navigation_success=(memory_localized or motion_policy.terminal in {'target_reached','manipulation_ready','stations_explored','fixture_handle_found','confined_target_found'} or getattr(motion_policy,'docking_success',False)) and not audit.events,
        detection_wall_seconds=policy.detection_wall_seconds,detection_calls=policy.detection_calls,map_guided_proposals=getattr(detector,'guided_proposal_count',0),
        external_contact_events=list(audit.events.values()),routes=policy.follower.events,
        targets=[{**asdict(t),'position':t.position.tolist()} for t in policy.targets.targets],
        limitations='Navigation stand-off and visual confirmation only. Not a grasp-ready pose certificate or full set_table success. DAAAM uses predicted masks; point-mask mode uses Molmo2 points and FastSAM masks. Target identity and fresh confirmation still require validation.')
    report['visual_relocalization_attempts']=relocalization_attempts
    report['visual_relocalization']=None if relocalization is None else {k:v for k,v in relocalization.result.items() if k not in ('query_timestamps',)}
    if a.station_exploration:
        from cross_episode_sim.navigation.route_memory import record_exploration
        report['route_graph_seed']=record_exploration(
            slam,furniture.landmarks,report['sensor_navigation_success'] and motion_policy.terminal=='stations_explored')
        report.update(scope='exploration to cached station stand-off',furniture_calls=len(furniture.events),furniture_inference_seconds=sum(e['inference_seconds'] for e in furniture.events),room_label_calls=0,manipulation_ready=False,limitations='Confirmed station memory and observed route only; no fresh manipulation-readiness claim.')
    if a.navigation_mode=='hybrid':
        if a.route_station and report['sensor_navigation_success']:
            from cross_episode_sim.navigation.route_memory import episode_routes
            route_memory=episode_routes()
            if route_memory is not None and hasattr(slam,'anchor'):
                route_memory.record_dock(slam.trajectory,slam.anchor,a.route_station)
        report.update(navigation_mode='hybrid',mode_transitions=motion_policy.events,
            docking_success=motion_policy.docking_success and not audit.events,
            docking_events=motion_policy.docking.events,
            arm_pregrasp_ik_feasible=motion_policy.docking.reach_joints is not None,
            arm_path_validated=False,
            object_search_success=motion_policy.object_search_success and not audit.events,
            manipulation_ready=motion_policy.terminal=='manipulation_ready' and not audit.events,
            room_observations=motion_policy.rooms.events,ring_actions=motion_policy.local.events,
            readiness=None if motion_policy.last_readiness is None else motion_policy.last_readiness.to_dict(),
            limitations='Shape-aware base docking uses known robot geometry and observed 3D occupancy. Candidate pregrasp IK is checked, but the arm approach path is not validated or executed. No manipulation-success claim.')
    from scripts.demo_house_exploration import score
    report['private_room_scores']=score(a.output,a.source,np.asarray(truth),private_initial,grid)
    report['private_rooms_visited']=sum(r['interior_visit_seconds']>=1. for r in report['private_room_scores'])
    report['cross_room_navigation_success']=report['sensor_navigation_success'] and report['private_rooms_visited']>=2
    artifacts.write_text(a.output/'report.json',json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
