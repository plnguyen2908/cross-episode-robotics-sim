"""Appliance contact motions from observed geometry and robot encoders.

No simulator object poses, fixture joints, contact flags or limits enter this
module. The backend supplies perception, learned grasps and checked arm paths.
Command completion is deliberately separate from private appliance scoring.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


# Extra rotation (rad) past this robot's own ON turn when switching a knob off.
KNOB_OFF_OVERTURN=.6   # fix177 replay: +0.4 left the front-right knob at 0.311 rad, just under the 0.35 ON threshold
# Documented knob positions (rad): OFF vertical, ON 60 deg; fits within the tolerance snap to them.
KNOB_DOCUMENTED_POSITIONS=(0.,np.pi/3)
KNOB_SNAP_TOLERANCE=np.radians(25)   # 40 deg let a bad 1.47 rad fit of a knob at OFF pass as already ON
# Closed-loop knob check: re-turns allowed and the indicator tolerance (rad) that ends them.
KNOB_VERIFY_RETRIES=2
KNOB_VERIFY_TOLERANCE=np.radians(15)


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
        # This robot's own record of turning this knob on in this episode.
        # Switching it off undoes exactly that turn (about the same observed
        # outward axis) instead of trusting a fresh indicator angle: fix168
        # call 16 read the 0.95 rad indicator with the wrong sign and turned
        # the knob on to its 3.14 rad stop (private), so the burner stayed on.
        import json
        turns=self.__dict__.setdefault('own_knob_turns',{})
        undo=turns.get(target) if state=='off' else None
        if undo is not None:
            # Turn a little past the undo: the knob's OFF end is a hard stop, so
            # the extra only presses into it, while grip slip no longer leaves
            # the burner just above its ON threshold (fix175 replay: undo
            # exactly 1.051 rad -> knob 1.062 -> 0.375 rad, still ON above 0.35).
            delta=-min(float(undo)+KNOB_OFF_OVERTURN,np.pi/2)
            # The indicator is where this robot left it (its ON target), not the
            # fresh fit: fix175 call 17 read -1.505 rad for a knob at +1.06, so
            # the grasp was ranked across the indicator bar, slipped, and the
            # burner stayed on. Grasp ranking uses this remembered angle.
            remembered=self.__dict__.setdefault('own_knob_indicator',{}).get(target)
            print('KNOB_OFF_UNDOES_OWN_TURN',json.dumps(dict(target=target,turn_rad=round(-delta,3),
                observed_indicator_rad=None if observed.angle is None else round(observed.angle,3),
                remembered_indicator_rad=None if remembered is None else round(remembered,3))),flush=True)
            if remembered is not None:
                from dataclasses import replace
                observed=replace(observed,angle=float(remembered))
        else:
            if observed.angle is None:return SkillOutcome(False,'knob_indicator_unobserved')
            # The documented two-position stove knobs rest at OFF (vertical) or
            # ON (60 deg). Snap a nearby fit to that position: fix182 call 14 read
            # a knob at OFF as -0.52 rad from a square stand, so the turn was
            # 'ambiguous' and the pinch point missed the bar (replay: empty grasp).
            snapped=min(KNOB_DOCUMENTED_POSITIONS,key=lambda a:abs(observed.angle-a))
            if abs(observed.angle-snapped)<=KNOB_SNAP_TOLERANCE and abs(observed.angle-snapped)>1e-6:
                print('KNOB_INDICATOR_SNAPPED',json.dumps(dict(target=target,observed_rad=round(observed.angle,3),
                    documented_rad=round(snapped,3))),flush=True)
                from dataclasses import replace
                observed=replace(observed,angle=float(snapped))
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
        # Measured result, not command completion: look at the knob again and
        # turn the rest when the grip slipped (fix183 replay: commanded 1.13 rad
        # on, knob reached 0.18, still below ON). Bounded; never claims success
        # from this check, the private scorer still decides.
        final=np.pi/3 if state=='on' else 0.;start_angle=observed.angle;self._verify_backed_off=False
        for attempt in range(KNOB_VERIFY_RETRIES):
            # A fresh look (pointing + knob fit), not the cached track: the
            # tracked refit read the just-released knob as horizontal (-1.4/-1.5
            # rad for 0.18) in fix183 replays.
            tracks=getattr(self.backend,'knob_tracks',None)
            if isinstance(tracks,dict):
                for key in [k for k in tracks if k[0]==target]:tracks.pop(key,None)
            try:check=self.backend.acquire(target,'knob')
            except RuntimeError as exc:
                # The pointed knob pixel can land on the just-withdrawn arm
                # ('no valid non-robot depth', fix185 call 17). Back further out
                # once and look again before giving up.
                print('KNOB_VERIFY_UNOBSERVED',json.dumps(dict(target=target,reason=str(exc)[:120])),flush=True)
                further=getattr(self.backend,'withdraw_further',None)
                if further is None or getattr(self,'_verify_backed_off',False):break
                self._verify_backed_off=True
                try:further(observed);check=self.backend.acquire(target,'knob')
                except RuntimeError as again:print('KNOB_VERIFY_UNOBSERVED',json.dumps(dict(target=target,reason='after backing off: '+str(again)[:100])),flush=True);break
            if check.angle is None:print('KNOB_VERIFY_UNOBSERVED',json.dumps(dict(target=target,reason='indicator not fitted')),flush=True);break
            remaining=final-check.angle
            print('KNOB_VERIFY',json.dumps(dict(target=target,attempt=attempt+1,observed_rad=round(check.angle,3),remaining_rad=round(remaining,3))),flush=True)
            if abs(remaining)<KNOB_VERIFY_TOLERANCE:break
            if state=='on' and abs(remaining)>np.pi/2:break
            # Re-turn only for a reading between the start and the target: a
            # re-look after the release once read 1.51 rad for a knob at 0.18
            # and turned it the wrong way (fix183 replay). Other readings stop.
            planned=final-start_angle if start_angle is not None else remaining
            # OFF: any ON reading gets another turn toward the stop (fix185 call
            # 17: a slip flicked the knob from 0.64 to 2.5 rad, past where the
            # turn started). Turning further toward OFF only presses on the stop,
            # so a misread there is harmless. ON keeps the between-start-and-
            # target rule (a misread once turned an ON knob the wrong way).
            if state=='off':
                if remaining>0:break
            elif np.sign(remaining)!=np.sign(planned) or abs(remaining)>abs(planned)+.1:
                print('KNOB_VERIFY_INCONSISTENT',json.dumps(dict(target=target,observed_rad=round(check.angle,3),planned_rad=round(float(planned),3))),flush=True);break
            step=remaining if state=='on' else -min(abs(remaining)+KNOB_OFF_OVERTURN,np.pi/2)
            self.backend.grasp(check)
            self.backend.follow_contact(list(knob_rotation_poses(self.backend.tcp(),check,step)),check)
            self.backend.release_withdraw(check)
            delta=float(delta)+float(step)
        indicators=self.__dict__.setdefault('own_knob_indicator',{})
        if state=='on':
            turns[target]=turns.get(target,0.)+float(delta)
            if observed.angle is not None:indicators[target]=float(observed.angle)+float(delta)
        else:turns.pop(target,None);indicators.pop(target,None)
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
