"""Opt-in carry handoff: retain the measured arm pose after route preflight."""
from itertools import islice
from time import monotonic
import numpy as np


def retain_if_route_clear(c):
    if not getattr(c,'holding_loaf',False):
        return False
    info=c.object_info[c.object_name]
    skip=getattr(c,'_public_dock_skip',0)
    docks=list(islice(c.dock_candidates(info['destination'],info['destination_position']),skip,skip+4))
    if not docks:return False
    original_undock=getattr(c,'_pickup_pre_nav_undock',0.)
    original_blended=getattr(c,'_cross_room_requires_blended',False)
    # No cached route from another posture may certify this posture.
    c._accepted_route=None
    started=monotonic()
    rejected=[]
    from cross_episode_sim.navigation.checked_route import checked_route
    for dock in docks:
        c._accepted_route=None
        c._pickup_pre_nav_undock=original_undock
        c._cross_room_requires_blended=original_blended
        try:
            path=checked_route(c,dock[:2],True,dock[2])
            break
        except RuntimeError as exc:
            rejected.append(str(exc))
    else:
        c._accepted_route=None
        c._pickup_pre_nav_undock=original_undock
        c._cross_room_requires_blended=original_blended
        c.record(carry_posture_choice=dict(retained=False,reason='; '.join(rejected),
                                          preflight_seconds=monotonic()-started))
        return False
    home=[float(c.data.joint(c.profile.namespace+n).qpos[0]) for n in c.profile.arm_joints]
    c.embodiment.set_loaded_travel_posture(home)
    c._accepted_route=path
    c.grasp_relative=np.linalg.inv(c.tcp())@c.bread_pose()
    c.record(carry_posture_choice=dict(retained=True,arm_joints=home,dock=dock.tolist(),
                                      preflight_seconds=monotonic()-started))
    return True


def centered_payload_target(base, tcp, payload):
    """Align the wrist and held object with chassis forward, preserving tilt."""
    forward=np.array([np.cos(base[2]),np.sin(base[2])])
    offset=payload[:2,3]-base[:2]
    # A rim grasp can put the wrist to the side even when the object is centered.
    # Rotate the whole held assembly about vertical, placing the wrist behind
    # the object on the same forward ray without changing tilt or grasp geometry.
    grasp_offset=payload[:2,3]-tcp[:2,3]
    angle=0. if np.linalg.norm(grasp_offset)<.005 else base[2]-np.arctan2(grasp_offset[1],grasp_offset[0])
    cs,sn=np.cos(angle),np.sin(angle)
    rotation=np.array([[cs,-sn,0.],[sn,cs,0.],[0.,0.,1.]])
    center=payload[:3,3].copy()
    center[:2]=base[:2]+np.linalg.norm(offset)*forward
    target=tcp.copy()
    target[:3,:3]=rotation@tcp[:3,:3]
    target[:3,3]=center+rotation@(tcp[:3,3]-payload[:3,3])
    return target


def center_loaded_payload(c, *, allow_preflight_fallback=False):
    """A checked wrist/object adjustment, retaining the grasp and object tilt."""
    base=c.base_pose()
    forward=np.array([np.cos(base[2]),np.sin(base[2])])
    lateral=np.array([-forward[1],forward[0]])
    before=c.bread_pose().copy()
    target=centered_payload_target(base,c.tcp(),before)
    distance=float(np.linalg.norm(target[:3,3]-c.tcp()[:3,3]))
    rotation_error=float(np.linalg.norm(target[:3,:3]-c.tcp()[:3,:3]))
    if distance>.005 or rotation_error>.01:
        try:
            path=c.plan_contact_path('center held object for forward carry',target)
            c.check_loaded_tuck_path(path)
        except RuntimeError as exc:
            if not allow_preflight_fallback:
                raise
            c._accepted_route=None
            c.record(carry_centering_preflight_rejected=str(exc))
            return False
        c.mesh_contact_move('center held object for forward carry',target,path=path)
    offset=c.bread_pose()[:2,3]-c.base_pose()[:2]
    error=float(abs(offset@lateral))
    wrist_error=float(abs((c.tcp()[:2,3]-c.base_pose()[:2])@lateral))
    c.record(forward_carry_posture=dict(adjustment_m=distance,lateral_offset_m=error,
                                      forward_distance_m=float(offset@forward),wrist_lateral_offset_m=wrist_error))
    if error>.02 or wrist_error>.02 or offset@forward<=0:
        raise RuntimeError('Held object did not reach a centered forward carry pose')
    c._accepted_route=None
    c.grasp_relative=np.linalg.inv(c.tcp())@c.bread_pose()
    return True
