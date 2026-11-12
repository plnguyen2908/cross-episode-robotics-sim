import numpy as np
import pytest
from types import SimpleNamespace
from cross_episode_sim.control.raised_payload_posture import plan_loaded_stow


def planner():
    return SimpleNamespace(allow_curobo_fallback=True,events=[],
                           plan_joint_goal=lambda q,goal:np.linspace(q,goal,5))


START=np.array([1.,.5,0.,0.,0.,0.,0.]);STOW=np.zeros(7)
# The held object meets a rim while the base joint swings with the shoulder still lowered.
rim=lambda path:not any(.3<row[0]<.7 and abs(row[1])>.2 for row in path)


def test_clear_direct_stow_is_unchanged():
    path,route=plan_loaded_stow(planner(),START,STOW,lambda path:True)
    assert route=='direct'
    np.testing.assert_allclose(path,np.linspace(START,STOW,5))


def test_blocked_held_sweep_takes_a_checked_staged_route():
    arm=planner()
    path,route=plan_loaded_stow(arm,START,STOW,rim)
    assert route=='shoulder_first' and rim(path)
    np.testing.assert_allclose(path[0],START);np.testing.assert_allclose(path[-1],STOW)
    assert arm.allow_curobo_fallback


def test_no_clear_route_keeps_the_object_and_reports_the_held_surface():
    with pytest.raises(RuntimeError,match='Observed held surface blocks clearance posture'):
        plan_loaded_stow(planner(),START,STOW,lambda path:False)


def test_arm_blocked_everywhere_reports_the_arm_reason():
    def blocked(q,goal):raise RuntimeError('No validated smooth path to selected joint goal')
    arm=SimpleNamespace(allow_curobo_fallback=False,events=[],plan_joint_goal=blocked,
                        geometry=SimpleNamespace(tcp_pose=lambda q:np.eye(4)),plan=lambda *a:blocked(None,None))
    with pytest.raises(RuntimeError,match='No validated smooth path'):
        plan_loaded_stow(arm,START,STOW,lambda path:True)
