"""Fresh wrist-depth correction for cookware pickup after map-based docking."""
import json,math
import numpy as np
import mujoco
from PIL import Image
from .shaped_docking import RobotGeometry,ShapedDocking
from .dock_follower import DockFollower
from ..control.onboard_base import OnboardBaseActuator
from ..control.sensor_appliance_backend import camera_points,pixel_point
from ..perception.molmo_pointing import MolmoPointer
from ..recording.deferred import StateVideoRecorder
from ..runtime import artifacts


# Fresh wrist-depth distance (m) accepted for a cookware pickup, and the number of looks.
COOKWARE_PICKUP_BAND=(.59,.67)
COOKWARE_REFINE_LOOKS=4


def refine_cookware_view(runtime,source,output):
    """Correct the local distance using RGB-D, not an old map target centroid."""
    output.mkdir(parents=True,exist_ok=False)
    tracker=runtime.mapper.view();geometry=RobotGeometry(runtime.model)
    from .household_map_handoff import bind_observed_map
    docking=ShapedDocking(bind_observed_map(runtime,source),geometry)
    base=OnboardBaseActuator(runtime.model,runtime.data);pointer=MolmoPointer()
    label=json.loads(artifacts.read_text(source/'report.json'))['selected_object']['label']
    start=float(runtime.data.time);events=[]
    with StateVideoRecorder(runtime.model,runtime.data,output/'state_recording') as rec:
        # Four looks, three moves: the last move is measured too (fix184 call 21:
        # 0.666 m, then 0.660 m after the third move was never measured, and
        # the pick ended 'outside the pickup band').
        for attempt in range(COOKWARE_REFINE_LOOKS):
            packet=runtime.sensors.capture()
            estimate=tracker.update(packet.cameras['navigation'],packet.timestamp)
            if not estimate.accepted:raise RuntimeError('Cookware alignment needs localization')
            camera=packet.cameras['wrist'];answer=pointer.point(Image.fromarray(camera.rgb),label)
            if not answer['points']:raise RuntimeError('Cookware not visible in wrist view')
            _,_,valid=camera_points(camera);local=pixel_point(camera,valid,answer['points'][0])
            T=estimate.T_map_base;target=T[:3,:3]@local+T[:3,3]
            distance=float(np.linalg.norm(local[:2]));heading=float(np.arctan2(local[1],local[0]))
            events.append(dict(attempt=attempt,local_target=local.tolist(),distance=distance,heading=heading))
            if COOKWARE_PICKUP_BAND[0]<=distance<=COOKWARE_PICKUP_BAND[1] and abs(heading)<.06:break
            if attempt==COOKWARE_REFINE_LOOKS-1:continue
            pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
            forward=target[:2]-pose[:2];forward/=np.linalg.norm(forward)
            goal=np.r_[target[:2]-.64*forward,math.atan2(forward[1],forward[0])]
            checker=docking.make_checker(pose,packet.arm.position)
            turn=(goal[2]-pose[2]+np.pi)%(2*np.pi)-np.pi
            # The pick redock stops about 0.55 m from the target, inside this
            # band, so the base often backs off a few cm next to the counter.
            # fix167: the base stood 7 cm from a counter corner (private), inside
            # the padded footprint, so every plan was 'start_blocked'. Back
            # straight out while the overlap never grows, then measure again.
            from .observed_undock import straight_escape
            escape=None
            if not checker.clear(pose):
                # Straight to the pickup pose itself when it lies straight
                # ahead/behind (fix167 replay: three short escapes to the edge
                # of the margin used every attempt and ended 'outside band').
                if abs(turn)<.08 and checker.clear(goal):
                    count=checker.contact_count(pose);steps=max(1,int(np.ceil(np.linalg.norm(goal[:2]-pose[:2])/.01)))
                    for k in range(1,steps+1):
                        p=np.r_[pose[:2]+(goal[:2]-pose[:2])*k/steps,pose[2]+turn*k/steps];now=checker.contact_count(p)
                        if now>count:break
                        count=now
                    else:escape=goal.copy()
                if escape is None:escape=straight_escape(checker,pose)
            print('COOKWARE_REFINE',attempt,'distance',round(distance,3),'heading',round(heading,3),'start_clear',bool(checker.clear(pose)),flush=True)
            if escape is not None:
                events[-1].update(start_inside_margin=True,escape=escape.tolist())
                print('COOKWARE_REFINE_STRAIGHT_ESCAPE',round(float(np.linalg.norm(escape[:2]-pose[:2])),3),flush=True)
                docking.path=[escape]
            elif abs(turn)<.08 and checker.swept(pose,goal):
                docking.path=[goal]
            else:
                # A counter corner may block a straight advance. Keep the same
                # sensed target and try nearby facing docks on either side.
                candidates=[goal]
                bearing=math.atan2(forward[1],forward[0])
                for offset in (.15,-.15,.30,-.30,.45,-.45):
                    yaw=bearing+offset
                    candidates.append(np.r_[target[:2]-.64*np.array([math.cos(yaw),math.sin(yaw)]),yaw])
                if not docking.plan_candidates(pose,packet.arm.position,candidates,source='fresh_cookware_depth'):
                    raise RuntimeError('No checked cookware refinement: '+str(docking.events[-2:]))
                from .observed_undock import departure_route
                docking.path=departure_route(pose,docking.path,docking.checker,keep_final_rotation=True)
            follower=DockFollower(docking.path,position_tolerance=.01,max_speed=.12,max_yaw_speed=.4)
            for tick in range(15*50):
                if tick%5==0:
                    packet=runtime.sensors.capture(include_robot_mask=False)
                    estimate=tracker.update(packet.cameras['navigation'],packet.timestamp)
                    rec.record_map(runtime.observed_navigation_map,estimate,'fresh_cookware_dock')
                if not estimate.accepted:base.step(0.,0.);continue
                P=estimate.T_map_base;p=np.r_[P[:2,3],math.atan2(P[1,0],P[0,0])]
                speed,yaw=follower.command(p);base.step(speed,yaw)
                if follower.complete and base.stationary():break
            else:raise RuntimeError('Cookware refinement motion budget exhausted')
        else:raise RuntimeError('Fresh cookware depth remains outside the pickup band')
    report=json.loads(artifacts.read_text(source/'report.json'))
    report['selected_object']['position']=target.tolist()
    report['fresh_cookware_alignment']=events
    artifacts.write_text(source/'report.json',json.dumps(report))
    tracker.save(source/'slam')
    # Publish the new live handoff after actual base motion. This does not
    # restore or reset physics; capture refuses stale state snapshots.
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(runtime.model,spec))
    mujoco.mj_getState(runtime.model,runtime.data,state,spec)
    artifacts.save_array(source/'private_checkpoint/state.npy',state)
    result=dict(success=True,events=events,simulated_seconds=float(runtime.data.time)-start,
        provenance='fresh wrist RGB-D and continuous estimated SLAM poses')
    (output/'report.json').write_text(json.dumps(result,indent=2));return result
