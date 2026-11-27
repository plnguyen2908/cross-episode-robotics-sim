"""Loaded local docking perpendicular to a sensor-observed fixture opening."""
import json,math
import numpy as np
from .shaped_docking import RobotGeometry,ShapedDocking,ObservedShapeChecker
from .dock_follower import DockFollower
from .observed_undock import departure_route
from ..control.observed_payload import ObservedPayload
from ..control.onboard_base import OnboardBaseActuator
from ..recording.deferred import StateVideoRecorder


def observed_open_panel_points(points,aperture):
    """Measured door surface near the remembered leaf, in the SLAM map frame.

    A handle's height does not specify a panel's bottom/top. In particular, a
    counter-mounted microwave door must never become a floor-to-ceiling wall.
    Keep actual depth returns; older observed obstacles remain in the map.
    """
    points=np.asarray(points,float).reshape(-1,3)
    if 'open_handle' not in aperture:return np.empty((0,3))
    hinge=np.asarray(aperture['hinge'])[:2];handle=np.asarray(aperture['open_handle'])[:2]
    delta=handle-hinge;length=float(np.linalg.norm(delta))
    if length<.03:return np.empty((0,3))
    along=(points[:,:2]-hinge)@(delta/length)
    normal=np.array([-delta[1],delta[0]])/length
    across=np.abs((points[:,:2]-hinge)@normal)
    remembered=(along>=-.05)&(along<=length+.08)&(across<.06)
    # A passive door does not stay where it was opened: it drifted from 90
    # to 47 degrees (bumped during pick and carry) and a re-dock that only
    # avoided the remembered leaf pushed it shut. Also keep every observed
    # point inside the leaf's swing sector: within its length of the hinge,
    # in front of the closed face, at the door's height band.
    swing=np.zeros(len(points),bool)
    if 'closed_normal' in aperture:
        outward=np.asarray(aperture['closed_normal'])[:2];outward=outward/max(np.linalg.norm(outward),1e-9)
        if 'closed_handle' in aperture:
            closed=np.asarray(aperture['closed_handle'])
            low=float(closed[2])-.35;high=float(closed[2])+.35
        else:low,high=.04,np.inf
        radial=np.linalg.norm(points[:,:2]-hinge,axis=1)
        swing=(radial<=length+.05)&((points[:,:2]-hinge)@outward>.02)&(points[:,2]>=low)&(points[:,2]<=high)
    if 'closed_handle' in aperture:
        # the remembered-line points take the door's height band too: floor returns 4-6 cm up under the leaf's line
        # beside the robot counted as door inside its own base and refused every dock route (runs 21, 23, 2026-11-23)
        zc=float(np.asarray(aperture['closed_handle'])[2]);remembered&=(points[:,2]>=zc-.35)&(points[:,2]<=zc+.35)
    return points[(remembered|swing)&(points[:,2]>.04)]


def dock_to_aperture(runtime,aperture,tool_points,output,seconds=60.,distances=(.30,.35,.40,.45,.50,.55),multi_goal=False,
                     offsets=(0.,.06,-.06,.12,-.12,.18,-.18,.24,-.24,.30,-.30),require_observed_support=True,
                     arm_clear=None,attempts=4,side_docks=None):
    """Aperture input is in the continuous estimated map frame, not world truth.

    With `arm_clear(attempt)`, a reached dock must also leave the arm a valid
    planning start in the observed world; otherwise that goal is excluded and
    the base re-docks to the remaining candidates from where it stands.
    """
    output.mkdir(parents=True,exist_ok=False)
    excluded=[];blocked=[]
    for attempt in range(attempts if arm_clear is not None else 1):
        report=_dock_once(runtime,aperture,tool_points,output/(f'attempt_{attempt}' if attempt else '.'),seconds,distances,
                          multi_goal,offsets,require_observed_support,excluded,first=attempt==0,side_docks=side_docks)
        if arm_clear is None or arm_clear(attempt):
            report.update(arm_start_blocked_docks=blocked)
            (output/'report.json').write_text(json.dumps(report,indent=2));return report
        excluded.append(np.asarray(report['goal'],float));blocked.append(report['goal'])
        print('APERTURE_DOCK_ARM_BLOCKED',json.dumps(dict(attempt=attempt,goal=report['goal'])),flush=True)
    raise RuntimeError('Every reached aperture dock leaves the arm start blocked: '+json.dumps(blocked))


# Planning margins (m) tried, largest first, when the standing pose is inside the nominal 0.035 m.
STANDING_MARGINS=(.02,.01,0.)


def outside_robot(points_base,boxes,pad=.04):
    """Depth returns (base frame) that are not on the robot's own arm, base or held object.

    The opened-door swing sector is filled from raw depth, and the navigation
    camera sees the raised arm. Standing 0.52 m from the microwave the elbow
    is inside that sector, so its own returns were kept as the door and the
    re-dock 8 cm closer was refused (fix106 state: shoulder against points
    10 cm from the elbow). Returns within `pad` of the robot's own boxes at
    its current joint positions are the robot, not the door.
    """
    points=np.asarray(points_base,float).reshape(-1,3)
    own=np.zeros(len(points),bool)
    for box in boxes:own|=np.all(np.abs((points-box.center)@box.axes)<=box.half+pad,axis=1)
    return points[~own]


def side_dock_poses(aperture,side_docks):
    """The datagen oracle's cabinet stand poses, from the measured door only.

    The oracle does not stand centred on the opening: it stands beside it, on
    the handle side (away from the hinge and the swung leaf), `beyond` metres
    past the closed handle and `distance` out from the door plane, facing the
    cabinet squarely. The hand then enters straight through the opening with
    the elbow free, instead of folding in front of a centred base.
    """
    hinge=np.asarray(aperture['hinge'])[:2];handle=np.asarray(aperture['closed_handle'])[:2]
    normal=np.asarray(aperture['closed_normal'])[:2];normal=normal/np.linalg.norm(normal)
    across=handle-hinge;across=across-normal*float(across@normal)
    width=float(np.linalg.norm(across))
    if width<.05:return []
    across=across/width;heading=math.atan2(-normal[1],-normal[0])
    return [np.r_[handle-normal*float((handle-hinge)@normal)+normal*distance+across*beyond,heading] for distance,beyond in side_docks]


def _dock_once(runtime,aperture,tool_points,output,seconds,distances,multi_goal,offsets,require_observed_support,excluded,first=True,side_docks=None,escaped=False):
    if not first:output.mkdir(parents=True,exist_ok=False)
    geometry=RobotGeometry(runtime.model);payload=ObservedPayload(geometry,tool_points)
    grid=runtime.observed_navigation_map;tracker=runtime.mapper.view()
    raw=runtime.sensors.capture(include_robot_mask=False)
    tracking=tracker.update(raw.cameras['navigation'],raw.timestamp)
    if not tracking.accepted:raise RuntimeError('Aperture docking requires localization')
    T=tracking.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
    packet=payload.filter_packet(raw)
    support=list(getattr(grid,'inherited_support_poses',[]))+[pose]
    panel_points=None
    robot_boxes=geometry.boxes(packet.arm.position)+[payload.box(packet.arm.position)]
    if 'open_handle' in aperture:
        from ..control.sensor_appliance_backend import camera_points
        observed=np.concatenate([camera_points(cam)[0] for cam in packet.cameras.values()])
        seen=len(observed);observed=outside_robot(observed,robot_boxes)
        if seen-len(observed):print('APERTURE_DOCK_OWN_RETURNS_DROPPED',json.dumps(dict(points=seen-len(observed))),flush=True)
        mapped=observed@T[:3,:3].T+T[:3,3]
        panel_points=observed_open_panel_points(mapped,aperture)
    checker=ObservedShapeChecker(grid,robot_boxes,pose,support_poses=support,additional_points=panel_points,require_observed_support=require_observed_support)
    if not checker.clear(pose):
        # Standing inside the planning margin of something the robot is not
        # touching (the opened door's free edge beside the raised elbow once
        # it is docked: 0.22 m to the side, fix111). No route can start from
        # such a pose, so the re-dock 8 cm closer was refused. As for the
        # undock from an opened cabinet, plan with the largest smaller margin
        # the standing pose satisfies; a pose in real overlap stays refused.
        nominal=checker.margin
        for margin in STANDING_MARGINS:
            checker.margin=margin
            if checker.clear(pose):
                print('APERTURE_DOCK_STANDING_MARGIN',json.dumps(dict(nominal_m=nominal,used_m=margin)),flush=True);break
        else:
            checker.margin=nominal
            # Not even the zero margin holds (fix128: carrying the food, the
            # robot stopped with one observed point of the opened door 4 cm
            # from its upper arm, and no dock route could start). Back straight
            # out of the margin first, never deeper into any observed point,
            # then look for the dock again from the clear pose.
            from .observed_undock import straight_escape
            escape=None if escaped else straight_escape(checker,pose)
            if escape is not None:
                print('APERTURE_DOCK_BLOCKED_START_ESCAPE',json.dumps(dict(blocked_by=(checker.detail or {}).get('part'),
                    along_heading_m=round(float((escape[:2]-pose[:2])@np.array([math.cos(pose[2]),math.sin(pose[2])])),3))),flush=True)
                back=DockFollower([escape],position_tolerance=.015,max_speed=.12,max_yaw_speed=.5,recover_turn_drift=True)
                drive=OnboardBaseActuator(runtime.model,runtime.data,motion_guard=lambda forward,yaw,dt:True)
                (output/'escape').mkdir(parents=True,exist_ok=True)
                with StateVideoRecorder(runtime.model,runtime.data,output/'escape'/'state_recording') as rec:
                    for tick in range(15*50):
                        if tick%5==0:
                            raw=runtime.sensors.capture(include_robot_mask=False)
                            tracking=tracker.update(raw.cameras['navigation'],raw.timestamp)
                            rec.record_map(grid,tracking,'loaded_aperture_escape')
                        if not tracking.accepted:drive.step(0.,0.);continue
                        T=tracking.T_map_base;here=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
                        drive.step(*back.command(here))
                        if back.complete and drive.stationary():break
                    drive.step(0.,0.)
                return _dock_once(runtime,aperture,tool_points,output/'after_escape',seconds,distances,multi_goal,offsets,
                    require_observed_support,excluded,first=False,side_docks=side_docks,escaped=True)
    class LoadedDock(ShapedDocking):
        def make_checker(self,pose,q):return checker
    docking=LoadedDock(grid,geometry)
    normal=np.asarray(aperture['closed_normal'])[:2];normal/=np.linalg.norm(normal)
    center=(np.asarray(aperture['closed_handle'])[:2]+np.asarray(aperture['hinge'])[:2])/2
    heading=math.atan2(-normal[1],-normal[0])
    tangent=np.array([-normal[1],normal[0]])
    candidates=[np.r_[center+normal*distance+tangent*offset,heading]
                for distance in distances for offset in offsets]
    if excluded:
        # A requested dock left the arm start blocked (the swung door beside
        # the elbow). The microwave asks for one centred 0.60 m pose, so
        # widen to nearby poses: small shifts away from the open leaf first,
        # then toward it, at the requested and +/-5 cm stand-offs. A goal
        # within 4 cm of a blocked one is not offered again.
        side=1.
        if 'open_handle' in aperture:
            leaf=np.asarray(aperture['open_handle'])[:2]-center
            side=-1. if float(leaf@tangent)>0 else 1.
        shifts=[side*k for k in (.06,.12,.18)]+[-side*k for k in (.06,.12)]
        candidates=candidates+[np.r_[center+normal*(distance+dd)+tangent*o,heading]
            for distance in distances for dd in (0.,-.05,.05) for o in ([0.] if dd else [])+shifts]
        candidates=[g for g in candidates if all(np.linalg.norm(g[:2]-e[:2])>.04 for e in excluded)]
        multi_goal=False  # keep the preference order above; plan_candidates tries it in order
    if not candidates:raise RuntimeError('No aperture-facing dock left after arm-start rejections')
    planned=False
    if side_docks and 'closed_handle' in aperture:
        preferred=[g for g in side_dock_poses(aperture,side_docks) if all(np.linalg.norm(g[:2]-e[:2])>.04 for e in excluded)]
        planned=bool(preferred) and docking.plan_candidates(pose,packet.arm.position,preferred,source='oracle_side_dock')
        print('APERTURE_SIDE_DOCK',json.dumps(dict(offered=len(preferred),planned=bool(planned))),flush=True)
    if planned:pass
    elif multi_goal:
        from .placement_docks import plan_to_any_dock
        # Reach binds an interior placement: try the closest stand-off ring
        # first and move outward only when no dock on it is reachable. One
        # search over every ring returns whichever dock is nearest the robot's
        # current pose, so the distance to the opening varied run to run
        # (0.77 m contact passed, 0.89 m was out of reach). The arm-start
        # check in dock_to_aperture still rejects a dock that is too tight.
        planned=False
        standoff=lambda g:float((g[:2]-center)@normal)
        for ring in sorted({round(standoff(g),3) for g in candidates}):
            tier=[g for g in candidates if abs(standoff(g)-ring)<.005]
            if plan_to_any_dock(docking,pose,packet.arm.position,tier):
                planned=True;break
    else:planned=docking.plan_candidates(pose,packet.arm.position,candidates,source='observed_aperture_front')
    if not planned:
        raise RuntimeError('No loaded aperture-facing dock: '+str(docking.events[-1:]))
    path=departure_route(pose,docking.path,checker,keep_final_rotation=True)
    final_goal=path[-1].copy()
    follower=DockFollower(path,position_tolerance=.015,max_speed=.18,max_yaw_speed=.5)
    base=OnboardBaseActuator(runtime.model,runtime.data,motion_guard=lambda forward,yaw,dt:follower is not None);start=float(runtime.data.time)
    with StateVideoRecorder(runtime.model,runtime.data,output/'state_recording') as rec:
        for tick in range(round(seconds*50)):
            if tick%5==0:
                raw=runtime.sensors.capture(include_robot_mask=False)
                tracking=tracker.update(raw.cameras['navigation'],raw.timestamp)
                rec.record_map(grid,tracking,'loaded_aperture_dock')
            if not tracking.accepted:base.step(0.,0.);continue
            T=tracking.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
            try:
                speed,yaw=follower.command(pose)
            except RuntimeError as exc:
                if str(exc)!='Dock turn drift requires replanning':raise
                base.step(0.,0.)
                checker.remember_support(pose)
                if not docking.plan_candidates(pose,packet.arm.position,[final_goal],source='aperture_turn_drift_replan'):
                    raise RuntimeError('No checked aperture route after turn drift') from exc
                path=departure_route(pose,docking.path,checker,keep_final_rotation=True)
                follower=DockFollower(path,position_tolerance=.015,max_speed=.18,max_yaw_speed=.5)
                continue
            base.step(speed,yaw)
            if tick%50==0:print('APERTURE_DOCK',float(runtime.data.time-start),pose.tolist(),flush=True)
            if follower.complete and base.stationary():
                report=dict(success=True,simulated_seconds=float(runtime.data.time-start),goal=path[-1].tolist(),events=docking.events,unknown_floor_policy='reject' if require_observed_support else 'allow_local_redock',observed_obstacle_checks=True)
                (output/'report.json').write_text(json.dumps(report,indent=2));return report
    raise RuntimeError('Aperture docking budget exhausted')
