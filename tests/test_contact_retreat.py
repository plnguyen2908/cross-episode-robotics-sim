import numpy as np
import pytest
from cross_episode_sim.control.contact_retreat import ContactRetreatPlanner

class LinearArm:
    limits=np.array([[-2.,2.]]*3);tcp_body=0;tcp_local=np.eye(4)
    def fk(self,q):
        T=np.eye(4);T[:3,3]=q;return {0:T}


def test_retreat_preserves_contact_pose_and_checks_every_sample():
    checked=[];geo=LinearArm();q=np.array([.5,.1,.7]);tcp=geo.fk(q)[0]
    planner=ContactRetreatPlanner(geo,lambda q,offset:checked.append((q,offset)) or True,lambda offset:True)
    rows=planner.plan(q,tcp,distance=.1)
    assert len(rows)==51 and len(checked)==51
    for offset,arm in rows:np.testing.assert_allclose(arm+offset,q,atol=1e-6)


def test_retreat_rejects_unknown_floor_and_obstacles():
    geo=LinearArm();q=np.array([.5,0,.7]);tcp=geo.fk(q)[0]
    with pytest.raises(RuntimeError,match='observed_floor'):
        ContactRetreatPlanner(geo,lambda *args:True,lambda offset:offset[0]>-.01).plan(q,tcp)
    with pytest.raises(RuntimeError,match='sweep_blocked'):
        ContactRetreatPlanner(geo,lambda q,offset:offset[0]>-.01,lambda offset:True).plan(q,tcp)


def test_forward_reposition_requires_observed_support():
    from cross_episode_sim.control.contact_retreat import occupied_support_validator
    from cross_episode_sim.navigation.shaped_docking import RobotBox
    box=RobotBox('base',np.array([0,0,.16]),np.eye(3),np.array([.2405,.2405,.16]))
    guard=occupied_support_validator([box])
    assert guard(np.array([-.04,0,0]))
    assert not guard(np.array([.15,0,0]))
    observed=[(x,y) for x in range(3,5) for y in range(-3,3)]
    guard=occupied_support_validator([box],observed_cells=observed)
    assert guard(np.array([.15,0,0]))


def test_coordinated_push_moves_world_tcp_with_base_instead_of_retracting_arm():
    geo=LinearArm();q=np.array([.5,0,.7]);start=geo.fk(q)[0]
    goal=start.copy();goal[:3,3]+=[.04,.01,0.]
    rows=ContactRetreatPlanner(geo,lambda *args:True,lambda offset:True).plan(q,start,distance=-.04,tcp_end=goal)
    offset,arm=rows[-1]
    np.testing.assert_allclose(offset,[.04,0,0])
    np.testing.assert_allclose(arm+offset,goal[:3,3],atol=1e-6)
    np.testing.assert_allclose(arm[0],q[0],atol=1e-6)


def test_planar_contact_motion_retains_world_goal_and_rejects_vertical_base():
    geo=LinearArm();q=np.array([.5,0,.7]);start=geo.fk(q)[0]
    goal=start.copy();goal[:3,3]+=[.04,.04,0.]
    planner=ContactRetreatPlanner(geo,lambda *args:True,lambda offset:True)
    rows=planner.plan(q,start,distance=-.05,tcp_end=goal,base_offset_end=[.04,.02,0.])
    offset,arm=rows[-1]
    np.testing.assert_allclose(arm+offset,goal[:3,3],atol=1e-6)
    np.testing.assert_allclose(arm,q+[0,.02,0],atol=1e-6)
    with pytest.raises(ValueError,match='planar base offset'):
        planner.plan(q,start,base_offset_end=[0,0,.02])


def test_measured_joint_at_limit_does_not_crash_ik_initialization():
    geo=LinearArm();lo,hi=geo.limits.T
    q=(lo+hi)/2;q[0]=lo[0]
    tcp=geo.fk(q)[0]
    planner=ContactRetreatPlanner(geo,lambda *args:True,lambda offset:True)
    # The solver may reject the motion, but a measured endpoint must not
    # raise scipy's out-of-bounds seed ValueError.
    try:planner.plan(q,tcp,distance=.01)
    except RuntimeError:pass


@pytest.mark.parametrize("yaw", [.08, -.08])
def test_base_yaw_compensates_tool_rotation_with_pose_sweep_checks(yaw):
    from scipy.spatial.transform import Rotation
    geo=LinearArm();q=np.array([.5,0,.7]);start=geo.fk(q)[0]
    R=Rotation.from_euler('z',yaw).as_matrix();offset=np.array([.02,.01,0.])
    goal=start.copy();goal[:3,:3]=R;goal[:3,3]=R@q+offset
    seen=[]
    planner=ContactRetreatPlanner(geo,lambda *args:True,lambda *args:True,
        pose_clear=lambda joints,xy,yaw:seen.append(yaw) or True,
        support_pose_clear=lambda *args:True)
    rows=planner.plan(q,start,distance=.03,tcp_end=goal,base_offset_end=offset,base_yaw_end=yaw)
    np.testing.assert_allclose(rows[-1][1],q,atol=.002)
    assert seen[-1]==pytest.approx(yaw)
    with pytest.raises(ValueError,match='pose sweep'):
        ContactRetreatPlanner(geo,lambda *args:True,lambda *args:True).plan(q,start,base_yaw_end=.08)


def test_combined_base_motion_does_not_double_count_tool_yaw():
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.control.contact_retreat import base_offset_for_tool_motion
    start=np.eye(4);start[:3,3]=[.6,.1,.9]
    rotation=Rotation.from_euler('z',.08).as_matrix();translation=np.array([-.02,.03,0.])
    goal=start.copy();goal[:3,:3]=rotation;goal[:3,3]=rotation@start[:3,3]+translation
    np.testing.assert_allclose(base_offset_for_tool_motion(start,goal,.08),translation,atol=1e-12)


def test_support_memory_transforms_known_floor_without_marking_unknown_free():
    from cross_episode_sim.control.contact_retreat import occupied_support_validator,transform_support_region
    from cross_episode_sim.navigation.shaped_docking import RobotBox
    from shapely.geometry import box
    base=RobotBox('base',np.array([0,0,.16]),np.eye(3),np.array([.2405,.2405,.16]))
    old=box(-.8,-.8,.8,.8)
    T=np.eye(4);T[0,3]=-.1
    memory=transform_support_region(old,T)
    assert memory.bounds==pytest.approx((-.9,-.8,.7,.8))
    guard=occupied_support_validator([base],observed_support=memory)
    assert guard([-.3,0,0])
    assert not guard([.7,0,0])


def test_longer_contact_repark_respects_quintic_peak_speed():
    from cross_episode_sim.control.contact_retreat import contact_motion_duration
    assert contact_motion_duration([.04,0,0])==2.
    assert contact_motion_duration([.12,0,0])==2.
    duration=contact_motion_duration([.15,0,0])
    assert duration>2 and 1.875*.15/duration<=.1125
    assert 1.875*.12/contact_motion_duration([.04,0,0],.12)<=.12


