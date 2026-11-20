import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.appliance_controls import ObservedControl,knob_rotation_poses,SensorApplianceControls
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface


def control(angle=0.):return ObservedControl(np.array([.5,0.,.9]),np.array([-1.,0.,0.]),np.array([[.5,0.,.9]]),angle)


def test_knob_rotation_about_observed_axis_not_tool_origin():
    c=control();pose=np.eye(4);pose[:3,3]=c.center+[.03,.01,.02]
    final=list(knob_rotation_poses(pose,c,np.pi/3))[-1]
    r=Rotation.from_rotvec(c.outward*np.pi/3).as_matrix()
    np.testing.assert_allclose(final[:3,3],c.center+r@(pose[:3,3]-c.center))
    np.testing.assert_allclose(final[:3,:3],r)
    np.testing.assert_array_equal(pose[:3,:3],np.eye(3))


class Backend:
    def __init__(self,angle=0.):self.angle=angle;self.events=[]
    def acquire(self,target,kind):self.events.append(('observe',kind,target));return control(self.angle)
    def grasp(self,observed):self.events.append('grasp')
    def tcp(self):p=np.eye(4);p[:3,3]=[.48,0.,.9];return p
    def follow_contact(self,poses,observed):self.events.append(('rotate',len(poses)))
    def release_withdraw(self,observed):
        self.events.append('release_withdraw')
        # The knob ends where it was turned (read by the closed-loop check).
        if getattr(self,'finals',None):self.angle=self.finals.pop(0)
        elif getattr(self,'final',None) is not None:self.angle=self.final
    def press_withdraw(self,observed):self.events.append('press_withdraw')


@pytest.mark.parametrize('state,initial',[('on',0.),('off',np.pi/3)])
def test_public_knob_uses_observation_grasp_rotate_release(state,initial):
    b=Backend(initial);b.final=np.pi/3 if state=='on' else 0.;api=M2T2SkillInterface(SensorApplianceControls(b).handlers())
    result=api.execute_skill('rotate_knob',dict(target='stove knob',state=state))
    assert result['success']
    # One closed-loop look after the release; the knob reads its target, so no re-turn.
    assert b.events==[('observe','knob','stove knob'),'grasp',('rotate',10),'release_withdraw',('observe','knob','stove knob')]


def test_missing_indicator_does_not_guess_off_or_move():
    b=Backend(None);api=M2T2SkillInterface(SensorApplianceControls(b).handlers())
    result=api.execute_skill('rotate_knob',dict(target='stove knob',state='on'))
    assert not result['success'] and result['reason']=='knob_indicator_unobserved'
    assert len(b.events)==1


def test_button_has_no_grasp_or_hidden_navigation():
    b=Backend();api=M2T2SkillInterface(SensorApplianceControls(b).handlers())
    result=api.execute_skill('push_button',dict(target='microwave start button'))
    assert result['success']
    assert b.events==[('observe','button','microwave start button'),'press_withdraw']
    assert 'not_scored' in result['reason']


def test_invalid_rotation_does_not_produce_motion():
    with pytest.raises(ValueError):list(knob_rotation_poses(np.eye(4),control(),np.pi))
    with pytest.raises(ValueError):ObservedControl([0,0,0],[0,0,0],[[0,0,0]])


def test_button_contact_frames_are_right_handed_and_face_panel():
    from cross_episode_sim.control.appliance_controls import button_contact_rotations
    normal=np.array([-.8,.6,0.])
    for tilt,r in button_contact_rotations(normal):
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        assert np.linalg.det(r)==pytest.approx(1.)
        assert r[:,2]@(-normal)==pytest.approx(np.cos(np.radians(tilt)))


def test_precise_finger_narrow_phase_preserves_collision_and_rejects_aabb_false_positive():
    from cross_episode_sim.control.convex_finger_contact import convex_shape,convex_box_overlap
    shape=convex_shape([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    assert not convex_box_overlap(shape,[.9,.9,.9],np.eye(3),[.05]*3,.005)
    assert convex_box_overlap(shape,[.2,.2,.2],np.eye(3),[.05]*3,.005)
    assert convex_box_overlap(shape,[-.002,0,0],np.eye(3),[.001]*3,.005)


def test_native_distance_kernel_matches_sat_on_translated_convex_geometry():
    from cross_episode_sim.control.convex_finger_contact import convex_shape,convex_box_overlap,NativeConvexShape
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])+[.2,-.1,.3]
    shape=convex_shape(v);native=NativeConvexShape(v);rng=np.random.default_rng(4)
    for _ in range(20):
        center=rng.uniform(-.2,1.2,3);axes=Rotation.random(random_state=rng).as_matrix();half=np.array([.03,.1,.05])
        assert native.overlap(center,axes,half,0.)==convex_box_overlap(shape,center,axes,half)


def test_tracked_knob_keeps_axis_and_reads_rotated_bar_not_backing_disk():
    from cross_episode_sim.perception.appliance_controls import refit_tracked_knob
    c=control();a=np.linspace(0,2*np.pi,200,endpoint=False)
    rim=c.center+np.c_[np.zeros(len(a)),.03*np.sin(a),.03*np.cos(a)]
    y,z=np.meshgrid(np.linspace(-.004,.004,12),np.linspace(-.025,.025,60))
    bar=np.c_[np.full(y.size,-.02),y.ravel(),z.ravel()]
    previous=ObservedControl(c.center,c.outward,np.r_[rim,c.center+bar],0.)
    rotation=Rotation.from_rotvec(c.outward*.85).as_matrix()
    fresh=np.r_[rim,c.center+bar@rotation.T,[[.5,.1,.8],[.49,.15,.85]]]
    updated=refit_tracked_knob(fresh,previous)
    np.testing.assert_array_equal(updated.center,previous.center)
    assert updated.angle==pytest.approx(.85,abs=.01)


def test_tracked_knob_requires_fresh_indicator_not_only_rim():
    from cross_episode_sim.perception.appliance_controls import refit_tracked_knob
    c=control();a=np.linspace(0,2*np.pi,100,endpoint=False)
    rim=c.center+np.c_[np.zeros(len(a)),.03*np.sin(a),.03*np.cos(a)]
    front=c.center+np.c_[np.full(300,-.02),np.zeros(300),np.linspace(-.025,.025,300)]
    previous=ObservedControl(c.center,c.outward,np.r_[rim,front],0.)
    with pytest.raises(ValueError,match='occluded'):refit_tracked_knob(rim,previous)


def test_knob_slip_is_turned_again_toward_the_target():
    b=Backend(0.);b.final=.2;api=M2T2SkillInterface(SensorApplianceControls(b).handlers())
    result=api.execute_skill('rotate_knob',dict(target='stove knob',state='on'))
    assert result['success']
    from cross_episode_sim.control.appliance_controls import KNOB_VERIFY_RETRIES
    # The fake knob keeps slipping to 0.2 rad: each allowed re-turn is used, then it stops.
    assert b.events.count('grasp')==1+KNOB_VERIFY_RETRIES


def test_knob_reading_past_the_start_does_not_turn_back():
    b=Backend(0.);b.final=-.9;api=M2T2SkillInterface(SensorApplianceControls(b).handlers())
    result=api.execute_skill('rotate_knob',dict(target='stove knob',state='on'))
    assert result['success']
    assert b.events.count('grasp')==1  # inconsistent reading: no wrong-way turn


def test_knob_off_flicked_past_start_is_turned_off_again():
    # A slip flicked the knob from ON further ON (fix185 call 17); OFF re-turns it.
    b=Backend(np.pi/3);b.finals=[2.4,0.];api=M2T2SkillInterface(SensorApplianceControls(b).handlers())
    result=api.execute_skill('rotate_knob',dict(target='stove knob',state='off'))
    assert result['success']
    assert b.events.count('grasp')==2
