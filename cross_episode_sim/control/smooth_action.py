"""Shared live timing for already checked atomic motion segments.

Geometry and contact decisions belong to each skill. This layer interpolates
only on its validated joint polyline, never across skill/contact boundaries.
Explicit waits, gripper transitions and fixture-feedback loops stay outside it.
"""
from argparse import BooleanOptionalAction
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.control.smooth_waypoints import bounded_joint_durations


def composite_motion_settings(settings=None):
    """Generation always executes optimized atomics, including legacy configs."""
    motion = dict(record_action_replay=True)
    motion.update(settings or {})
    motion.update(optimize_motion_components=True, smooth_action=True)
    return motion


def add_arguments(parser):
    parser.add_argument('--optimize-motion-components', action=BooleanOptionalAction, default=True,
                        help='Optimize routes, posture handoffs and readiness independently of interpolation')
    parser.add_argument('--smooth-action', action=BooleanOptionalAction, default=True, help='Retime checked atomic motion segments; preserve contact checks and waits')
    parser.add_argument('--smooth-action-profile', choices=('conservative', 'fast'), default='fast')
    parser.add_argument('--record-action-replay', action='store_true')
    parser.add_argument('--smooth-action-time-scale', type=float, default=1.916685,
                        help='Fast-profile movement compression from the 35-second replay; waits/gripper events excluded')
    parser.add_argument('--smooth-action-speed', type=float, default=1.5)
    parser.add_argument('--smooth-action-contact-speed', type=float, default=1.1)
    parser.add_argument('--smooth-action-nav-speed', type=float, default=1.1)


def copy_arguments(source, target):
    """Preserve motion switches across scene-builder/controller argument layers."""
    for name in ('optimize_motion_components', 'record_action_replay', 'smooth_action',
                 'smooth_action_profile', 'smooth_action_time_scale', 'smooth_action_speed',
                 'smooth_action_contact_speed', 'smooth_action_nav_speed'):
        setattr(target, name, getattr(source, name))


def enabled(c):
    return bool(getattr(c.args, 'smooth_action', False))


def fast(c):
    return enabled(c) and getattr(c.args, 'smooth_action_profile', 'fast') == 'fast'


def components_enabled(c):
    return fast(c) or bool(getattr(c.args, "optimize_motion_components", False))


def configure(c):
    """Configure the shared optimized live policy across controller families."""
    if not components_enabled(c):
        return
    c.args.base_motion = 'turn_drive'
    c.forward_curve_experiment = True
    c.retain_carry_posture_experiment = True
    c.direct_carry_departure_experiment = True
    c.centered_carry_experiment = True
    c.adaptive_motion_settling = True
    c.args.nav_speed = .8
    c.args.turn_speed = 1.5
    c.args.motion_slowdown = .12


def factor(c, kind):
    name={'free':'smooth_action_speed','contact':'smooth_action_contact_speed',
          'navigation':'smooth_action_nav_speed'}[kind]
    value=float(getattr(c.args,name,{'free':1.5,'contact':1.1,'navigation':1.1}[kind]))
    if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite')
    return value


def joint_commands(c, path, dt, *, contact=False, start=None):
    """Return (commands, durations) without mutating physics or path geometry."""
    path=np.asarray(path,dtype=float)
    times=np.broadcast_to(np.asarray(dt,dtype=float),(len(path),)).copy()
    if not enabled(c) or not len(path):return path,times
    if path.ndim!=2 or not np.all(np.isfinite(path)) or np.any(times<=0):
        raise ValueError('Smooth Action needs finite joint waypoints and positive durations')
    if start is None:
        start=np.array([float(c.data.joint(c.profile.namespace+n).qpos[0]) for n in c.planner.names])
    start=np.asarray(start,dtype=float)
    kind='contact' if contact else 'free'
    requested=times/factor(c,kind)
    limits=SimpleNamespace(experimental_joint_speed=1.8 if fast(c) else 1.5,
                           experimental_joint_acceleration=10. if fast(c) else 8., model=c.model)
    if fast(c):
        compression=float(getattr(c.args, 'smooth_action_time_scale', 1.916685))
        if not np.isfinite(compression) or compression<=0:
            raise ValueError('smooth_action_time_scale must be positive and finite')
        # Parameterize the checked polyline by joint travel, then ease the
        # whole movement once. Retiming every small stroke twice adds artificial
        # rest ramps and can make dense contact paths slower than the baseline.
        points=np.vstack((start,path))
        distances=np.max(np.abs(np.diff(points,axis=0)),axis=1)
        keep=np.r_[True,distances>1e-12]
        points=points[keep]
        arc=np.r_[0.,np.cumsum(distances[distances>1e-12])]
        if arc[-1]<=1e-12:
            return path[-1:],np.array([c.model.opt.timestep])
        total=max(.02,1.875*arc[-1]/limits.experimental_joint_speed,
                  np.sqrt(6.*arc[-1]/limits.experimental_joint_acceleration))/compression
        # The 35-second replay compressed actuator time after the faster live
        # path. Its corresponding command velocity/acceleration bounds scale
        # by k and k². These are simulation settings, not hardware limits.
        limits.experimental_joint_speed*=compression
        limits.experimental_joint_acceleration*=compression**2
        # Invert the monotone quintic clock at each knot so no corner is cut.
        from scipy.optimize import brentq
        ease=lambda u:10*u**3-15*u**4+6*u**5
        knots=np.array([brentq(lambda u:ease(u)-x,0.,1.) for x in arc/arc[-1]])
        clock=np.unique(np.r_[np.linspace(0.,1.,int(np.ceil(total/.02))+1),knots])
        positions=ease(clock)*arc[-1]
        samples=np.column_stack([np.interp(positions,arc,points[:,j]) for j in range(path.shape[1])])[1:]
        # An inverse-clock knot may coincide numerically with a regular sample.
        # Every retained command must receive at least one physics tick.
        durations=np.maximum(np.diff(clock)*total,c.model.opt.timestep)
        durations=bounded_joint_durations(start,samples,durations,limits)
        c.record(smooth_action=dict(kind=kind,profile='fast',original_samples=len(path),
                 samples=len(samples),original_duration_s=float(times.sum()),
                 duration_s=float(durations.sum()),movement_time_scale=compression,
                 control_sample_seconds=.02))
        return samples,durations
    # Densify the existing checked straight segments, retaining every waypoint.
    samples=[];durations=[];previous=start
    quantum=float(c.model.opt.timestep)
    for q,duration in zip(path,requested):
        count=max(1,int(np.ceil(duration/.02)))
        samples.extend(previous+u*(q-previous) for u in np.linspace(0.,1.,count+1)[1:])
        durations.extend([duration/count]*count)
        previous=q
    samples=np.asarray(samples)
    # Local retiming includes rest boundaries, discrete acceleration and timestep rounding.
    durations=bounded_joint_durations(start,samples,np.asarray(durations),limits)
    event=dict(kind=kind,original_samples=len(path),samples=len(samples),
               profile=getattr(c.args, 'smooth_action_profile', 'fast'),
               original_duration_s=float(times.sum()),duration_s=float(durations.sum()),
               requested_speed_factor=factor(c,kind),control_sample_seconds=.02)
    c.record(smooth_action=event)
    return samples,durations


@dataclass
class TimedSegment:
    original: object
    duration: float

    @property
    def start(self):return self.original.start
    @start.setter
    def start(self,value):self.original.start=value
    @property
    def end(self):return self.original.end
    @property
    def kind(self):return self.original.kind
    @property
    def length(self):return self.original.length
    @property
    def rotation(self):return self.original.rotation
    def pose(self,u):return self.original.pose(u)


def navigation_segments(c, segments):
    if not enabled(c) or segments is None:return segments
    # Fast routes already carry explicit speed/curvature timing. Do not scale
    # their bends a second time and invalidate the forward-tracking limits.
    scale=1. if fast(c) else factor(c,'navigation')
    result=[TimedSegment(s,s.duration/(1. if s.kind=='reverse' else scale)) for s in segments]
    if fast(c) and getattr(c, '_atomic_motion_start', None) is not None:
        from .trajectory_budget import remaining_seconds, route_durations, NAVIGATION_LIMITS, FIXTURE_NAVIGATION_LIMITS
        budget = remaining_seconds(c, reserve=max(1., .3*len(segments)))
        limits = (FIXTURE_NAVIGATION_LIMITS
                  if getattr(c, '_public_door_reposition', None) is not None
                  else NAVIGATION_LIMITS)
        durations, floors = route_durations(segments, budget, limits=limits)
        result = [TimedSegment(s,float(t)) for s,t in zip(segments,durations)]
        c.record(trajectory_budget=dict(kind='navigation', remaining_seconds=budget,
                 command_limits=dict(limits),
                 path_length_m=sum(s.length for s in segments),
                 minimum_bounded_seconds=float(floors.sum()),
                 selected_seconds=float(durations.sum()),
                 target_limited=bool(durations.sum()>budget)))

    c.record(smooth_action=dict(kind='navigation',original_duration_s=sum(s.duration for s in segments),
                              duration_s=sum(s.duration for s in result),requested_speed_factor=scale,
                              effective_speed_factor=sum(s.duration for s in segments)/max(1e-9,sum(s.duration for s in result))))
    return result
