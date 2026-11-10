"""Appliance contact motions from observed geometry and robot encoders.

No simulator object poses, fixture joints, contact flags or limits enter this
module. The backend supplies perception, learned grasps and checked arm paths.
Command completion is deliberately separate from private appliance scoring.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ObservedControl:
    center: np.ndarray
    outward: np.ndarray
    surface: np.ndarray
    angle: float | None = None

    def __post_init__(self):
        for name in ('center', 'outward'):
            value=np.asarray(getattr(self,name),float)
            if value.shape!=(3,) or not np.isfinite(value).all():
                raise ValueError('Invalid observed control geometry')
            object.__setattr__(self,name,value.copy())
        norm=np.linalg.norm(self.outward)
        if norm<1e-8:raise ValueError('Missing observed normal')
        object.__setattr__(self,'outward',self.outward/norm)
        surface=np.asarray(self.surface,float)
        if surface.ndim!=2 or surface.shape[1]!=3 or not len(surface) or not np.isfinite(surface).all():
            raise ValueError('Missing observed control surface')
        object.__setattr__(self,'surface',surface.copy())
        if self.angle is not None and not np.isfinite(self.angle):raise ValueError('Invalid observed angle')


def knob_rotation_poses(tcp, control, delta, step=np.radians(6)):
    """Rotate a held grasp about an RGB-D-estimated knob axis, not the wrist."""
    if not np.isfinite(delta) or abs(delta)>np.pi/2+.001 or not 0<step<=np.pi/6:
        raise ValueError('Invalid knob rotation')
    initial=np.asarray(tcp,float)
    for angle in np.linspace(0.,delta,max(1,int(np.ceil(abs(delta)/step)))+1)[1:]:
        rotation=Rotation.from_rotvec(angle*control.outward).as_matrix()
        pose=initial.copy()
        pose[:3,:3]=rotation@initial[:3,:3]
        pose[:3,3]=control.center+rotation@(initial[:3,3]-control.center)
        yield pose


def button_contact_rotations(outward):
    direction=-np.asarray(outward,float);direction/=np.linalg.norm(direction)
    up=np.array([0.,0.,1.]);up-=direction*(up@direction)
    if np.linalg.norm(up)<.1:raise ValueError('Button requires an observed upright panel')
    up/=np.linalg.norm(up);across=np.cross(direction,up)
    frontal=np.column_stack([up,across,direction])
    for tilt in (-35.,-20.,0.,35.,20.,50.):
        inclined=Rotation.from_rotvec(np.radians(tilt)*across).as_matrix()@frontal
        for roll in (0.,180.,90.,-90.):
            yield tilt,inclined@Rotation.from_euler('z',roll,degrees=True).as_matrix()


class SensorApplianceControls:
    """Reuse one live backend across explicit rotate_knob / push_button calls.

    The backend must acquire targets from fresh onboard observations, physically
    grasp using M2T2 proposals, execute at 50 Hz, and release/withdraw. It must
    neither move the base implicitly nor restore a failed scene.
    """
    def __init__(self, backend):
        self.backend=backend

    def handlers(self):
        return dict(rotate_knob=self.rotate_knob,push_button=self.push_button)

    def rotate_knob(self, *, target, state, progress):
        from cross_episode_sim.skills.m2t2_interface import SkillOutcome
        if state not in ('on','off'):raise ValueError('Invalid knob state')
        progress('observe_knob','running')
        observed=self.backend.acquire(target,'knob')
        progress('observe_knob','completed')
        if observed.angle is None:return SkillOutcome(False,'knob_indicator_unobserved')
        # A vertical OFF marker and 60-degree ON position are this backend's
        # supported control convention, not a universal appliance assumption.
        desired=np.pi/3 if state=='on' else 0.
        delta=desired-observed.angle
        if abs(delta)<np.radians(4):return SkillOutcome(True,'observed_indicator_at_requested_state')
        if abs(delta)>np.pi/2:return SkillOutcome(False,'knob_indicator_ambiguous')
        progress('m2t2_grasp','running')
        self.backend.grasp(observed)
        progress('m2t2_grasp','completed')
        progress('rotate_contact','running')
        poses=list(knob_rotation_poses(self.backend.tcp(),observed,delta))
        self.backend.follow_contact(poses,observed)
        progress('rotate_contact','completed')
        progress('release_withdraw','running')
        self.backend.release_withdraw(observed)
        progress('release_withdraw','completed')
        return SkillOutcome(True,'rotation_commands_completed_appliance_state_not_scored')

    def push_button(self, *, target, progress):
        from cross_episode_sim.skills.m2t2_interface import SkillOutcome
        progress('observe_button','running')
        observed=self.backend.acquire(target,'button')
        progress('observe_button','completed')
        progress('press_withdraw','running')
        self.backend.press_withdraw(observed)
        progress('press_withdraw','completed')
        return SkillOutcome(True,'press_commands_completed_appliance_state_not_scored')
