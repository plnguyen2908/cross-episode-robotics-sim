"""Opt-in waypoint interpolation experiments; physical validation is mandatory."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator


def smooth_joint_samples(start, path, waypoint_dt, sample_dt=.02):
    """C1 interpolation through joint waypoints, stationary at both endpoints."""
    points = np.vstack((start, path))
    times = np.arange(len(points)) * waypoint_dt
    slopes = PchipInterpolator(times, points, axis=0).derivative()(times)
    slopes[[0, -1]] = 0.
    curve = CubicHermiteSpline(times, points, slopes, axis=0)
    steps = max(1, int(np.ceil(times[-1] / sample_dt)))
    dt = times[-1] / steps
    return curve(np.arange(1, steps+1)*dt), dt


@dataclass
class SmoothRoute:
    start: np.ndarray
    end: np.ndarray
    duration: float
    curve: object
    kind: str = 'smooth'

    def pose(self, u):
        return self.curve(np.clip(u, 0., 1.) * self.duration)

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.curve(np.linspace(0,self.duration,200))[:,:2],axis=0),axis=1).sum())

    @property
    def rotation(self):
        return float(np.abs(np.diff(self.curve(np.linspace(0,self.duration,200))[:,2])).sum())


def smooth_route(segments, pose_clear):
    """Join a blended route only after densely checking the new swept curve.

    Keep reverse undocking unchanged. Shrink tangent magnitudes if rounding a
    corner collides; retain the entire original route if no curve is feasible.
    Timing is never shortened here, to separate continuity from faster limits.
    """
    if len(segments) < 2 or any(s.kind == 'reverse' for s in segments):
        return segments, False
    points = np.array([segments[0].start, *[s.end for s in segments]])
    times = np.r_[0., np.cumsum([s.duration for s in segments])]
    slopes = np.zeros_like(points)
    slopes[1:-1] = (points[2:]-points[:-2])/(times[2:]-times[:-2])[:,None]
    for scale in (1., .5, .25):
        curve = CubicHermiteSpline(times, points, slopes*scale, axis=0)
        # At most 20 ms, then subdivide further for large spatial/yaw steps.
        ts = np.linspace(0.,times[-1],int(np.ceil(times[-1]/.02))+1)
        poses = curve(ts)
        safe = True
        for a,b in zip(poses, poses[1:]):
            n=max(1,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/.01)),int(np.ceil(abs(b[2]-a[2])/np.radians(2))))
            if not all(pose_clear(p) for p in np.linspace(a,b,n+1)):
                safe=False;break
        if safe:
            return [SmoothRoute(points[0].copy(),points[-1].copy(),times[-1],curve)], True
    return segments, False


def checked_contact_samples(c, path):
    """Interpolate and recheck robot + held-object meshes before execution."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.control.collision_probe import forward_collision
    ns=c.profile.namespace
    addresses=[c.model.jnt_qposadr[c.model.joint(ns+n).id] for n in c.planner.names]
    start=c.data.qpos[addresses]
    samples,dt=smooth_joint_samples(start,path,getattr(c,'contact_waypoint_seconds',.32),
                                    sample_dt=getattr(c,'contact_sample_seconds',.02))
    dt=bounded_joint_dt(start,samples,dt,c)
    dt=np.ceil(dt/c.model.opt.timestep)*c.model.opt.timestep
    probe=mujoco.MjData(c.model)
    mujoco.mj_copyData(probe,c.model,c.data)
    held=getattr(c,'holding_loaf',False)
    relative=np.linalg.inv(c.tcp())@c.bread_pose()
    obj_address=c.model.jnt_qposadr[c.model.joint(c.object_joint).id]
    live=c.data
    for q in samples:
        probe.qpos[:]=live.qpos
        probe.qpos[addresses]=q
        mujoco.mj_kinematics(c.model,probe)
        if held:
            try:
                c.data=probe
                obj=c.tcp()@relative
            finally:
                c.data=live
            probe.qpos[obj_address:obj_address+3]=obj[:3,3]
            probe.qpos[obj_address+3:obj_address+7]=Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
        forward_collision(c.model,probe)
        if c.navigation_penetration(probe,held)>.001 or c.robot_self_penetration(probe)>.0005:
            return path,.32,False
    return samples,dt,True


def bounded_joint_dt(start, path, dt, controller):
    """Opt-in time scaling bounded by joint speed and acceleration, at rest ends."""
    speed=getattr(controller,'experimental_joint_speed',None)
    acceleration=getattr(controller,'experimental_joint_acceleration',None)
    if speed is None or acceleration is None:return dt
    delta=np.diff(np.vstack((start,path)),axis=0)
    if not len(delta):return dt
    max_delta=float(np.max(np.abs(delta)))
    max_second=float(np.max(np.abs(np.diff(np.vstack((np.zeros_like(delta[0]),delta,np.zeros_like(delta[0]))),axis=0))))
    return max(dt,max_delta/speed,np.sqrt(max_second/acceleration))


def bounded_joint_durations(start, path, minimum_dt, controller):
    """Retiming on the original checked joint path, with local speed/accel bounds.

    One difficult waypoint must not slow every other waypoint in a cuRobo path.
    Acceleration is bounded between segment-average velocities, including rest
    boundaries. These are command bounds; execution still checks physical state.
    """
    delta=np.diff(np.vstack((start,path)),axis=0)
    if not len(delta):return np.empty(0)
    speed=controller.experimental_joint_speed
    acceleration=controller.experimental_joint_acceleration
    durations=np.maximum(minimum_dt,np.max(np.abs(delta),axis=1)/speed)
    quantum=getattr(getattr(getattr(controller,'model',None),'opt',None),'timestep',None)
    for _ in range(200):
        if quantum is not None:
            durations=np.ceil(durations/quantum-1e-10)*quantum
        velocity=delta/durations[:,None]
        spans=np.r_[durations[0]/2,(durations[:-1]+durations[1:])/2,durations[-1]/2]
        changes=np.diff(np.vstack((np.zeros_like(velocity[0]),velocity,np.zeros_like(velocity[0]))),axis=0)
        ratios=np.max(np.abs(changes),axis=1)/(spans*acceleration)
        if np.max(ratios)<=1.+1e-8:return durations
        factors=np.sqrt(np.maximum(1.,ratios))*1.001
        durations*=np.maximum(factors[:-1],factors[1:])
    raise RuntimeError('Local joint retiming did not converge')
