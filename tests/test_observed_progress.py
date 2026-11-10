import numpy as np
from cross_episode_sim.control.observed_progress import ObservedProgressGovernor


def test_lagging_servo_stays_within_tracking_lead_and_reaches_goal():
    g=ObservedProgressGovernor(.23);measured=0.;previous=0.
    for i in range(2000):
        command=g.advance(min(1,i/200),measured)
        assert 0 <= (command-previous)*g.length <= .00100001
        assert (command-measured)*g.length <= .00600001
        measured += .15*(command-measured)
        previous=command
        if abs(1-measured)*g.length<.001:break
    assert abs(1-measured)*g.length<.001


def test_stalled_localization_does_not_run_to_end_or_reverse():
    g=ObservedProgressGovernor(.1)
    commands=[g.advance(1.,0.) for _ in range(100)]
    assert np.all(np.diff(commands)>=0)
    assert commands[-1]<=.06000001
    assert g.advance(1.,-.02)>=commands[-1]


def test_contact_lead_tolerates_six_mm_visual_lag_without_exceeding_monitor():
    g=ObservedProgressGovernor(.1,lead=.010);actual=0.
    for i in range(1000):
        measured=max(0.,actual-.006/g.length)
        command=g.advance(min(1.,i/100),measured)
        assert (command-measured)*g.length<=.01000001
        assert (command-actual)*g.length<.015
        actual+=.2*(command-actual)
        if command==1:break
    assert command==1


def test_endpoint_feedback_corrects_residual_with_ten_hz_camera():
    from cross_episode_sim.control.observed_progress import contact_feedback_velocity
    pose=np.eye(4);actual=np.array([.094,0.,0.]);target=np.array([.1,0.,0.])
    # 50 Hz control with observations held between 10 Hz camera updates.
    for i in range(100):
        if i%5==0:pose[:3,3]=actual
        velocity,yaw=contact_feedback_velocity(pose,target,0.)
        assert np.linalg.norm(velocity[:2])<=.05000001 and abs(yaw)<=.15
        actual+=velocity*.02
    assert np.linalg.norm(actual-target)<.001


def test_feedback_is_body_relative_and_wraps_yaw():
    from cross_episode_sim.control.observed_progress import contact_feedback_velocity
    pose=np.eye(4);pose[:2,:2]=[[0.,-1.],[1.,0.]]
    v,w=contact_feedback_velocity(pose,[0.,.01,0.],np.pi/2)
    np.testing.assert_allclose(v,[.05,0.,0.],atol=1e-9)
    assert abs(w)<1e-9


def test_base_yaw_contributes_to_tool_progress_bound():
    from cross_episode_sim.control.observed_progress import contact_progress_length
    offset=np.array([-.07,-.035,0.]);yaw=.0873;tool=np.array([.49,-.33,.93])
    length=contact_progress_length(offset,yaw,tool)
    governor=ObservedProgressGovernor(length,lead=.008)
    measured=.43
    for _ in range(100):fraction=governor.advance(1.,measured)
    delta=fraction-measured
    assert delta*length<=.008+1e-12
    # This is the physical tool-error contribution the old translation-only
    # 10 mm lead failed to bound, before encoder/visual residuals are added.
    assert .010/np.linalg.norm(offset)*length>.012
    assert contact_progress_length(offset,0.,tool)==np.linalg.norm(offset)


def test_feedforward_moves_when_previous_command_is_tracked():
    from cross_episode_sim.control.observed_progress import contact_feedforward_velocity
    pose=np.eye(4)
    v,w=contact_feedforward_velocity(pose,np.zeros(3),0.,[.0004,0.,0.],.001)
    np.testing.assert_allclose(v,[.02,0.,0.])
    assert abs(w-.05)<1e-9


def test_feedforward_retains_endpoint_correction_and_limits():
    from cross_episode_sim.control.observed_progress import contact_feedforward_velocity,contact_feedback_velocity
    pose=np.eye(4);target=np.array([.01,0.,0.])
    a=contact_feedforward_velocity(pose,target,.02,target,.02)
    b=contact_feedback_velocity(pose,target,.02)
    np.testing.assert_allclose(a[0],b[0]);assert a[1]==b[1]
    v,w=contact_feedforward_velocity(pose,[0.,0.,0.],0.,[.1,.1,0.],1.)
    assert np.linalg.norm(v[:2])<=.05000001 and abs(w)<=.15
