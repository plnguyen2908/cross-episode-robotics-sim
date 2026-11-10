import numpy as np
import pytest
from cross_episode_sim.control.coordinated_contact import CoordinatedContactSegment, contact_blend
from cross_episode_sim.control.contact_retreat import ContactRetreatPlanner

class LinearArm:
    limits=np.array([[-2.,2.]]*3);tcp_body=0;tcp_local=np.eye(4)
    def fk(self,q):
        T=np.eye(4);T[:3,3]=q;return {0:T}


def test_oracle_endpoint_sweep_preserves_sensor_guards():
    geo=LinearArm();q=np.array([.5,.1,.7]);start=geo.fk(q)[0]
    end=start.copy();end[:3,3]+=[.04,.02,0]
    seen=[]
    planner=ContactRetreatPlanner(geo,lambda q,xy:seen.append(xy.copy()) or True,lambda xy:True)
    rows=planner.plan_oracle_segment(q,start,distance=-.04,tcp_end=end,base_offset_end=[.04,0,0])
    assert len(rows)==21 and len(seen)==21
    np.testing.assert_allclose(rows[-1][0]+rows[-1][1],end[:3,3],atol=1e-6)
    planner.support_clear=lambda xy:xy[0]<.01
    with pytest.raises(RuntimeError,match='observed_floor'):
        planner.plan_oracle_segment(q,start,distance=-.04,tcp_end=end)
    planner.support_clear=lambda xy:True;planner.collision_clear=lambda q,xy:xy[0]<.01
    with pytest.raises(RuntimeError,match='sweep_blocked'):
        planner.plan_oracle_segment(q,start,distance=-.04,tcp_end=end)


def test_shared_cubic_is_original_oracle_ramp_and_has_bounded_rates():
    segment=CoordinatedContactSegment(np.zeros(7),np.ones(7)*.1,np.array([.04,0,0]))
    u=np.linspace(0,1,101)
    np.testing.assert_allclose(contact_blend(u),u*u*(3-2*u),rtol=0,atol=0)
    duration=segment.duration(.08,bounded=True)
    assert 1.5*.04/duration<=.08 and 6*.04/duration**2<=.2
    assert 1.5*.1/duration<=1 and 6*.1/duration**2<=2
    assert duration<2
    np.testing.assert_allclose(segment.sample(1)[1],np.ones(7)*.1)


def test_hinge_policy_mirrors_and_does_not_overshoot_stop():
    from cross_episode_sim.control.coordinated_contact import next_hinge_angle
    for sign in (-1,1):
        assert next_hinge_angle(sign*1.56,sign*np.pi/2)==pytest.approx(sign*np.pi/2)
        assert next_hinge_angle(sign*.4,sign*1.5)==pytest.approx(sign*(.4+np.radians(3)))


def test_unreachable_endpoint_is_rejected_before_execution():
    geo=LinearArm();q=np.array([.5,.1,.7]);start=geo.fk(q)[0]
    end=start.copy();end[2,3]=3.
    planner=ContactRetreatPlanner(geo,lambda *args:True,lambda *args:True)
    with pytest.raises(RuntimeError,match='tracking_error'):
        planner.plan_oracle_segment(q,start,distance=-.04,tcp_end=end)


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('fraction',[0.,.5,1.])
def test_shared_panel_arc_matches_original_oracle_and_preserves_pad_contact(sign,fraction):
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.control.coordinated_contact import panel_arc_geometry,panel_contact_pose
    axis=np.array([0.,0.,1.]);anchor=np.array([.2,.1,0.])
    surface=np.array([sign*.7,.15,.85]);normal=np.array([0.,sign*1.,0.])
    wrist=Rotation.from_euler('xyz',[.2,.4,.3]).as_matrix()
    pads=np.array([[-.02,-.03,.06],[.02,.03,.08]])
    for angle in np.linspace(0,sign*np.pi/2,15):
        patch,direction,rotation=panel_arc_geometry(surface,normal,anchor,axis,angle,wrist,fraction)
        old_turn=Rotation.from_rotvec(axis*angle).as_matrix()
        np.testing.assert_allclose(patch,anchor+old_turn@(surface-anchor))
        np.testing.assert_allclose(direction,old_turn@normal)
        np.testing.assert_allclose(rotation,Rotation.from_rotvec(axis*angle*fraction).as_matrix()@wrist)
        pose=panel_contact_pose(patch,direction,rotation,pads)
        actual=pads@rotation.T+pose[:3,3]
        assert np.max((actual-patch)@direction)==pytest.approx(.0005)


def test_contact_failure_never_plans_reach():
    from cross_episode_sim.control.coordinated_contact import prepare_panel_push
    reached=[]
    def rejected(*args):raise RuntimeError('contact blocked')
    with pytest.raises(RuntimeError,match='contact blocked'):
        prepare_panel_push(np.eye(4),direction=[0.,1.,0.],
            solve_contact=lambda pose:np.zeros(7),check_contact=rejected,
            solve_precontact=lambda *args:None,plan_contact=lambda *args:None,
            check_approach=lambda *args:None,plan_reach=lambda q:reached.append(q))
    assert reached==[]
