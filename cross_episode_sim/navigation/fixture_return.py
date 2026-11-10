"""Return to a previously observed fixture view with the current held posture.

This is local docking after normal frozen-map transit. Goals come from accepted
SLAM poses in this episode, never scene reset configuration or simulator poses.
"""
import math
import json
import numpy as np
from .shaped_docking import RobotGeometry
from .carry_departure import FrozenBaseDepartureDocking
from .fixed_map import freeze_map
from .dock_follower import DockFollower
from ..control.onboard_base import OnboardBaseActuator
from ..recording.deferred import StateVideoRecorder


def fixture_return_planner(grid, geometry):
    # Use the same frozen 2D footprint planner as carry departure. Do not
    # veto a route using an estimated arm/full-stroke gripper volume.
    freeze_map(grid)
    return FrozenBaseDepartureDocking(grid,geometry)


def return_fixture_view(runtime, goal, output, seconds=30.):
    geometry=RobotGeometry(runtime.model)
    docking=fixture_return_planner(runtime.observed_navigation_map,geometry)
    base=OnboardBaseActuator(runtime.model,runtime.data,motion_guard=lambda forward,yaw,dt:follower is not None)
    tracking=runtime.mapper.view();follower=None;events=[]
    with StateVideoRecorder(runtime.model,runtime.data,output/'state_recording') as recorder:
        for tick in range(round(seconds*50)):
            if tick%5==0:
                packet=runtime.sensors.capture(include_robot_mask=False)
                estimate=tracking.update(packet.cameras['navigation'],packet.timestamp)
                recorder.record_map(runtime.observed_navigation_map,estimate,'remembered_fixture_view')
            if not estimate.accepted:
                base.step(0.,0.);continue
            T=estimate.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
            if follower is None:
                if not docking.plan_candidates(pose,packet.arm.position,[np.asarray(goal)],source='remembered_fixture_view'):
                    raise RuntimeError('No checked local route to observed fixture view: '+str(docking.events[-2:]))
                from .observed_undock import departure_route
                docking.path=departure_route(pose,docking.path,docking.checker,keep_final_rotation=True)
                follower=DockFollower(docking.path,max_speed=.18,max_yaw_speed=.6,
                                      recover_turn_drift=True)
                print("FIXTURE_RETURN_ROUTE",[v.tolist() for v in docking.path],flush=True)
            try:speed,yaw=follower.command(pose)
            except RuntimeError as exc:
                if str(exc)!='Dock turn drift requires replanning':raise
                events.append(dict(reason=str(exc),pose=pose.tolist()));follower=None
                base.step(0.,0.);continue
            base.step(speed,yaw)
            if tick%50==0:
                print("FIXTURE_RETURN",round(tick/50,2),pose.tolist(),"remaining",len(follower.route),flush=True)
            if follower.complete and base.stationary():
                return dict(success=True,events=events+docking.events,simulated_seconds=(tick+1)/50)
    raise RuntimeError('Remembered fixture docking budget exhausted')
