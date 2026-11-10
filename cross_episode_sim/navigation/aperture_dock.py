"""Loaded local docking perpendicular to a sensor-observed fixture opening."""
import json,math
import numpy as np
from .shaped_docking import RobotGeometry,ShapedDocking,ObservedShapeChecker
from .dock_follower import DockFollower
from .observed_undock import departure_route
from ..control.observed_payload import ObservedPayload
from ..control.onboard_base import OnboardBaseActuator
from ..recording.deferred import StateVideoRecorder


def dock_to_aperture(runtime,aperture,tool_points,output,seconds=60.,distances=(.30,.35,.40,.45,.50,.55)):
    """Aperture input is in the continuous estimated map frame, not world truth."""
    output.mkdir(parents=True,exist_ok=False)
    geometry=RobotGeometry(runtime.model);payload=ObservedPayload(geometry,tool_points)
    grid=runtime.observed_navigation_map;tracker=runtime.mapper.view()
    raw=runtime.sensors.capture(include_robot_mask=False)
    tracking=tracker.update(raw.cameras['navigation'],raw.timestamp)
    if not tracking.accepted:raise RuntimeError('Aperture docking requires localization')
    T=tracking.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
    packet=payload.filter_packet(raw)
    support=list(getattr(grid,'inherited_support_poses',[]))+[pose]
    panel_points=None
    if 'open_handle' in aperture:
        # Preserve the observed opened leaf even when the mast camera cannot
        # currently see its lower half. This is sensor-estimated articulation
        # memory, not simulator panel geometry or a changed global map.
        hinge=np.asarray(aperture['hinge']);handle=np.asarray(aperture['open_handle'])
        along=np.linspace(hinge[:2],handle[:2],max(2,int(np.linalg.norm(handle[:2]-hinge[:2])/.015)+1))
        heights=np.arange(.05,max(.5,2*handle[2])+.02,.025)
        panel_points=np.array([[xy[0],xy[1],z] for xy in along for z in heights])
    checker=ObservedShapeChecker(grid,geometry.boxes(packet.arm.position)+[payload.box(packet.arm.position)],pose,support_poses=support,additional_points=panel_points)
    class LoadedDock(ShapedDocking):
        def make_checker(self,pose,q):return checker
    docking=LoadedDock(grid,geometry)
    normal=np.asarray(aperture['closed_normal'])[:2];normal/=np.linalg.norm(normal)
    center=(np.asarray(aperture['closed_handle'])[:2]+np.asarray(aperture['hinge'])[:2])/2
    heading=math.atan2(-normal[1],-normal[0])
    tangent=np.array([-normal[1],normal[0]])
    candidates=[np.r_[center+normal*distance+tangent*offset,heading]
                for distance in distances for offset in (0.,.06,-.06,.12,-.12,.18,-.18,.24,-.24,.30,-.30)]
    if not docking.plan_candidates(pose,packet.arm.position,candidates,source='observed_aperture_front'):
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
                report=dict(success=True,simulated_seconds=float(runtime.data.time-start),goal=path[-1].tolist(),events=docking.events)
                (output/'report.json').write_text(json.dumps(report,indent=2));return report
    raise RuntimeError('Aperture docking budget exhausted')
