"""Local semantic approach/redock from RGB-D and continuous estimated poses."""
import json
import math
import time
import numpy as np
from PIL import Image
from cross_episode_sim.navigation.shaped_docking import RobotGeometry, ShapedDocking, wrap
from cross_episode_sim.navigation.dock_follower import DockFollower
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.control.sensor_appliance_backend import camera_points, pixel_point
from cross_episode_sim.perception.molmo_pointing import MolmoPointer


class SensorRedocking:
    """Retain the arm/gripper posture when its swept shape permits local travel.

    No object truth, reference docking poses, or reference actions are inputs.
    A fresh candidate/point is required after motion; callers invalidate grasps.
    This component's navigation is local visual approach, not exploration or
    replacement of the existing cross-room navigation skill.
    """
    def __init__(self, runtime, sensors, mapper):
        self.runtime, self.sensors, self.mapper = runtime, sensors, mapper
        self.geometry = RobotGeometry(runtime.model)
        self.pointer = MolmoPointer()
        self.target_point = None;self.target_label = None

    def observe(self, target):
        packet = self.sensors.capture(include_robot_mask=False)
        tracking = self.mapper.update(packet.cameras['navigation'], packet.timestamp)
        if not tracking.accepted:
            raise RuntimeError('Local redock needs an estimated camera pose')
        for key in ('navigation', 'wrist'):
            camera = packet.cameras[key]
            result = self.pointer.point(Image.fromarray(camera.rgb), target)
            if result['points']:
                _, _, valid = camera_points(camera)
                T = tracking.T_map_base
                candidates=[]
                for pixel in result['points']:
                    local = pixel_point(camera, valid, pixel)
                    if np.isfinite(local).all():candidates.append(T[:3,:3] @ local + T[:3,3])
                if candidates:
                    point=min(candidates,key=lambda p:np.linalg.norm(p-self.target_point)) if self.target_label==target and self.target_point is not None else candidates[0]
                    self.target_point=point.copy();self.target_label=target
                    return packet, tracking, point, key
        raise RuntimeError('Target not visible; use navigate/explore before redocking')

    def move(self, target, purpose, distance, output):
        if not isinstance(target, str) or not target.strip():
            raise ValueError('Provide a semantic target')
        if purpose not in ('pick', 'place', 'handle', 'approach'):
            raise ValueError('Unsupported docking purpose')
        distance = (.55 if purpose == 'pick' else .65) if distance is None else float(distance)
        if not np.isfinite(distance) or not .35 <= distance <= 1.5:
            raise ValueError('Local standoff must be 0.35–1.5 m')
        output.mkdir(parents=True, exist_ok=False)
        wall = time.monotonic();start = float(self.runtime.data.time)
        packet, tracking, point, camera = self.observe(target)
        grid = self.runtime.observed_navigation_map
        # Accumulate actual measured depth, without a scanning/tuck cycle.
        if grid.last_timestamp is None or tracking.timestamp > grid.last_timestamp:
            grid.update(packet.cameras['navigation'], tracking)
            grid.update_support(packet.cameras['wrist'], tracking)
        T = tracking.T_map_base
        pose = np.r_[T[:2,3], math.atan2(T[1,0], T[0,0])]
        delta = point[:2] - pose[:2];initial_distance = float(np.linalg.norm(delta))
        bearing = math.atan2(delta[1], delta[0])
        docking = ShapedDocking(grid, self.geometry, max_nodes=1800)
        checker = docking.make_checker(pose, packet.arm.position)
        candidates = [np.r_[point[:2] - distance*np.array([math.cos(bearing+a), math.sin(bearing+a)]), bearing+a]
                      for a in (0., .15, -.15, .30, -.30)]
        route = None
        # Prefer a continuous direct approach before lattice/A* fallback.
        for goal in candidates:
            travel_heading=math.atan2(goal[1]-pose[1],goal[0]-pose[0]) if np.linalg.norm(goal[:2]-pose[:2])>.015 else pose[2]
            if abs(wrap(travel_heading-pose[2]))>math.pi/2:travel_heading=wrap(travel_heading+math.pi)
            direct=[np.r_[pose[:2],travel_heading],np.r_[goal[:2],travel_heading],goal]
            if all(checker.swept(a,b) for a,b in zip([pose]+direct,direct)):
                route = direct;break
        if route is None and docking.plan_candidates(pose, packet.arm.position, candidates, source='qwen_sensor_redock'):
            checker = docking.checker;route = docking.path
        if route is None:
            result = dict(success=False, reason='no_observed_collision_clear_local_route',
                initial_distance_m=initial_distance, planning_events=docking.events, collision_detail=checker.detail)
            (output/'report.json').write_text(json.dumps(result, indent=2));return result
        path_length = sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip([pose]+route, route))
        turns = sum(abs(wrap(b[2]-a[2])) for a,b in zip([pose]+route, route))
        # Route-dependent speed under calibrated actuator limits; executed at 50 Hz.
        speed = float(np.clip(path_length / max(3., 10.-turns/.5), .12, .35))
        follower = DockFollower(route, max_speed=speed, max_yaw_speed=.6,
                                approach_gain=1.5, recover_turn_drift=True)
        base = OnboardBaseActuator(self.runtime.model, self.runtime.data,
            motion_guard=lambda f,w,dt: abs(f)<=speed+1e-8 and abs(w)<=.6+1e-8 and dt<=.02)
        # The guard authorizes the swept-checked retained posture, not arbitrary travel.
        previous = np.zeros(2); success = False
        with StateVideoRecorder(self.runtime.model, self.runtime.data, output/'state_recording') as rec:
            for tick in range(30*50):
                if tick % 5 == 0:
                    packet = self.sensors.capture(include_robot_mask=False)
                    tracking = self.mapper.update(packet.cameras['navigation'], packet.timestamp)
                    rec.record_map(grid, tracking, 'redock' if purpose != 'approach' else 'navigate')
                if not tracking.accepted:
                    base.step(0., 0.);continue
                T = tracking.T_map_base;p = np.r_[T[:2,3], math.atan2(T[1,0], T[0,0])]
                command = np.asarray(follower.command(p))
                # Smooth actual commands, preserving terminal stopping and physics.
                command = previous + np.clip(command-previous, [-.45*.02,-1.2*.02], [.45*.02,1.2*.02])
                base.step(*command);previous = command
                if follower.complete and np.max(np.abs(command)) < .001 and base.stationary():
                    success=True;break
                if tick % 250 == 0:
                    print(json.dumps(dict(stage=purpose, simulated_seconds=float(self.runtime.data.time)-start,
                        target_distance_m=float(np.linalg.norm(point[:2]-p[:2])))), flush=True)
            base.step(0.,0.)
        # Refresh both view and SLAM after motion; no old observation is handed off.
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
        final=float(np.linalg.norm(point[:2]-tracking.T_map_base[:2,3]))
        result=dict(success=success, reason='' if success else 'local_motion_budget',
            initial_distance_m=initial_distance,final_distance_m=final,requested_distance_m=distance,
            target=target,purpose=purpose,camera=camera,simulated_seconds=float(self.runtime.data.time)-start,
            wall_seconds=time.monotonic()-wall,path_length_m=float(path_length),speed_limit_mps=speed,
            retained_posture=True,posture_reason='swept robot geometry checked on observed map',
            duration_exception=None if self.runtime.data.time-start<=15 else 'route turning/settling exceeded target; physical execution retained')
        (output/'report.json').write_text(json.dumps(result,indent=2));return result
