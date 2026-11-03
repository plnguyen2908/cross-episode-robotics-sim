"""Two-room sensor-only navigation test; no manipulation or oracle destinations."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext
import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.mapping.hydra_mapping import HydraFrameWriter
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
from cross_episode_sim.perception.visual_targets import GroundingDinoDetector
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import setup,ContactAudit,plot_grid
from cross_episode_sim.navigation.map_video import MapVideoPanel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preserve-initial-state',action='store_true',help='Restore exact recorded state without resetting doors or arm posture')
    p.add_argument('--continuation',type=Path,help='Harness checkpoint and observed map from the preceding physical stage; no scene reset')
    p.add_argument('--target',choices=['cup','bowl','dining table'],default='cup')
    p.add_argument('--preferred-distance',type=float,default=.55,help='Preferred horizontal base-to-object distance in metres; collision clearance unchanged')
    p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--object-map',choices=['hydra','visual'],default='hydra')
    p.add_argument('--initial-backoff',type=float,default=0.,help='Standalone reset offset before evaluation; never a policy action')
    p.add_argument('--initial-yaw-offset',type=float,default=0.,help='Radians applied at standalone reset before evaluation')
    p.add_argument('--perception',choices=['daaam','grounding-dino'],default='daaam')
    p.add_argument('--detector-model',default='IDEA-Research/grounding-dino-tiny',help='RGB proposal checkpoint; masks and VLM verification remain enabled')
    p.add_argument('--vlm-model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--vlm-device',default='cuda',help='Local verifier device, e.g. cuda:1 with two visible GPUs')
    p.add_argument('--vlm-url',default=None)
    p.add_argument('--explorer',choices=['frontier','tare'],default='tare')
    p.add_argument('--navigation-mode',choices=['semantic','hybrid'],default='semantic')
    p.add_argument('--ring-checkpoint',type=Path)
    p.add_argument('--ring-device',default='cuda:0')
    p.add_argument('--goal-room',default='kitchen',help='Visual room category; empty string skips room search for local tests')
    p.add_argument("--map-resolution",type=float,default=.10,help="Observed collision-map voxel size in metres")
    a=p.parse_args()
    if a.navigation_mode=='hybrid' and (a.object_map!='hydra' or a.perception!='daaam' or a.vlm_url or not a.ring_checkpoint):
        p.error('Hybrid smoke test requires native Hydra, DAAAM, local Qwen and --ring-checkpoint')
    if a.object_map=='hydra' and (a.target=='dining table' or a.perception!='daaam'):
        p.error('Hydra object navigation requires DAAAM masks and a cup/bowl target; use --object-map visual for the ablation.')
    if a.preserve_initial_state and (a.initial_backoff or a.initial_yaw_offset):
        p.error('Saved-state evaluation forbids spawn offsets')
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    if a.perception=='daaam':
        from cross_episode_sim.perception.daaam_grounding import DaaamGrounder,LocalVLM,EndpointVLM
        vlm=EndpointVLM(a.vlm_url,a.vlm_model) if a.vlm_url else LocalVLM(a.vlm_model,device=a.vlm_device)
        detector=DaaamGrounder(vlm,'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',a.output/'grounding',categories=tuple(dict.fromkeys(('cup','bowl',a.target))),detector_model=a.detector_model)
    else:detector=GroundingDinoDetector(model=a.detector_model)
    if a.continuation:
        if a.initial_backoff or a.initial_yaw_offset:p.error('Continuation forbids spawn offsets')
        import mujoco
        model=mujoco.MjModel.from_binary_path(str(a.continuation/'private_checkpoint/model.mjb'))
        data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,np.load(a.continuation/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model,data);doors=[]
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
    slam=RGBDMapper();grid=ObservedNavigationMap(MapConfig(extent_m=12.,resolution=a.map_resolution,height_resolution=a.map_resolution,pixel_stride=4 if a.map_resolution<.1 else 6));config=HouseExplorationConfig(max_seconds=a.seconds)
    if a.continuation:
        from cross_episode_sim.navigation.map_continuation import restore_observed_map
        restore_observed_map(grid,a.continuation)
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
    ring_context=nullcontext();motion_policy=policy
    if a.navigation_mode=='hybrid':
        from cross_episode_sim.navigation.onering import OneRingProcess,OneRingLocalNavigator
        from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
        from cross_episode_sim.mapping.room_memory import ObservedRoomMemory
        from cross_episode_sim.navigation.readiness import ManipulationReadiness
        ring_context=OneRingProcess(a.ring_checkpoint,a.output/'onering',device=a.ring_device)
        local=OneRingLocalNavigator(grid,ring_context,f'Find a {a.target}',config)
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry,ShapedDocking
        docking=ShapedDocking(grid,RobotGeometry(model))
        motion_policy=HybridNavigator(policy,local,ObservedRoomMemory(vlm,remembered=memory['rooms']),ManipulationReadiness(),a.goal_room,docking=docking)
    map_panel=MapVideoPanel()
    start=float(data.time);wall=time.monotonic();truth=[];last_status=None;last_saved_detection=None
    (a.output/'detections').mkdir()
    hydra_frames=HydraFrameWriter(a.output/'hydra_frames')
    (a.output/'configuration.json').write_text(json.dumps(dict(source=str(a.source),preserve_initial_state=a.preserve_initial_state,navigation_mode=a.navigation_mode,goal_room=a.goal_room,ring_checkpoint=str(a.ring_checkpoint) if a.ring_checkpoint else None,ring_device=a.ring_device,target=a.target,preferred_distance=a.preferred_distance,object_map=a.object_map,explorer=a.explorer,perception=a.perception,detector_model=a.detector_model,vlm_model=a.vlm_model,vlm_device=a.vlm_device,seconds=a.seconds,initial_backoff=a.initial_backoff,initial_yaw_offset=a.initial_yaw_offset,doors_initialized_open=doors),indent=2))
    with ring_context,mapper_context,planner_context,OnboardSensorInterface(model,data) as sensors,imageio.get_writer(str(a.output/'navigation.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,imageio.get_writer(str(a.output/'navigation_with_map.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as map_video,open(a.output/'progress.jsonl','w',buffering=1) as log:
        for tick in tqdm(range(round(a.seconds*50)+10),desc='Visual target navigation'):
            if tick%5==0:
                packet=sensors.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
            command=motion_policy.step(packet,tracking,float(data.time))
            if tick%5==0:
                row=dict(time=float(data.time-start),mode=motion_policy.mode,object_search_success=getattr(motion_policy,'object_search_success',False),**command,estimated_position=tracking.T_map_base[:2,3].tolist() if tracking.accepted else None,targets=[dict(id=t.track_id,label=t.label,position=t.position.tolist(),observations=t.observations) for t in policy.targets.targets])
                log.write(json.dumps(row)+'\n');truth.append(data.xpos[model.body('robot_0/base').id].copy())
                panels=[]
                for name,c in packet.cameras.items():
                    # Clean camera panels; detection stays active in the policy
                    # and diagnostic logs, but boxes/labels are not burned in.
                    panels.append(np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1))
                panel=Image.fromarray(np.concatenate(panels,axis=0));draw=ImageDraw.Draw(panel)
                draw.rectangle((0,0,panel.width,24),fill='white');draw.text((5,5),f'{row["time"]:.1f}s | {motion_policy.mode} | {command["status"]} | {a.target}',fill='black')
                if a.navigation_mode=='hybrid':
                    draw.rectangle((0,24,panel.width,44),fill='white')
                    draw.text((5,26),f"Object search verified: {motion_policy.object_search_success} | Manipulation ready: {motion_policy.terminal=='manipulation_ready'}",fill='black')
                video.append_data(np.asarray(panel))
                online_map=map_panel.render(grid,tracking,panel.height-80,row['time'])
                combined=Image.new('RGB',(panel.width+online_map.width,panel.height),'white')
                combined.paste(panel,(0,0));combined.paste(online_map,(panel.width,0))
                map_video.append_data(np.asarray(combined))
                if tick%250==0 or motion_policy.terminal:combined.save(a.output/'latest_map_frame.png')
                if policy.last_detection!=last_saved_detection and packet.timestamp==policy.last_detection:
                    panel.save(a.output/'detections'/f'frame_{tick:06d}.png');last_saved_detection=policy.last_detection
                    hydra_frames.update(packet.cameras['navigation'],tracking,policy.last_detections.get('navigation',[]),packet.timestamp)
                if tick%250==0 or motion_policy.terminal:panel.save(a.output/'latest_frame.png')
                if command['status']!=last_status or tick%250==0:tqdm.write(json.dumps(row));last_status=command['status']
            base.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
            if motion_policy.terminal and base.stationary():break
    # Private checkpoint for reproducible continuation, never a policy observation.
    import mujoco
    checkpoint=a.output/'private_checkpoint';checkpoint.mkdir()
    mujoco.mj_saveModel(model,str(checkpoint/'model.mjb'),None)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(model,spec))
    mujoco.mj_getState(model,data,state,spec);np.save(checkpoint/'state.npy',state)
    grid.save(a.output);slam.save(a.output/'slam');plot_grid(a.output,'Partial sensor map and target navigation')
    save_memory(a.output,memory,room_events=getattr(getattr(motion_policy,'rooms',None),'events',()),targets=policy.targets.targets)
    np.save(a.output/'diagnostic_positions.npy',truth)
    selected=getattr(motion_policy,'found_target',None) or policy.follower.target
    handoff=None if selected is None else dict(native_id=selected.native_id,label=selected.label,position=selected.position.tolist(),frame='estimated_slam_map',last_observed=selected.last_seen,grasp_ready=False)
    report=dict(selected_object=handoff,status=motion_policy.terminal,target=a.target,object_map=a.object_map,simulated_seconds=float(data.time-start),wall_seconds=time.monotonic()-wall,
        sensor_navigation_success=(motion_policy.terminal in {'target_reached','manipulation_ready'} or getattr(motion_policy,'docking_success',False)) and not audit.events,
        detection_wall_seconds=policy.detection_wall_seconds,detection_calls=policy.detection_calls,map_guided_proposals=getattr(detector,'guided_proposal_count',0),
        external_contact_events=list(audit.events.values()),routes=policy.follower.events,
        targets=[{**asdict(t),'position':t.position.tolist()} for t in policy.targets.targets],
        limitations='Navigation stand-off and visual confirmation only. Not a grasp-ready pose certificate or full set_table success. DAAAM uses predicted masks; grounding-dino mode uses boxes. Target identity and fresh confirmation still require validation.')
    if a.navigation_mode=='hybrid':
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
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
