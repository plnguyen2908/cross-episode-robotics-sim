import numpy as np
import pytest
from cross_episode_sim.navigation.dock_follower import DockFollower

def test_turn_stays_latched_through_small_position_jitter():
    f=DockFollower([[.1,0,1.2]])
    assert f.command(np.array([.088,0,.1]))[0]==0
    speed,yaw=f.command(np.array([.086,.008,.35]))
    assert speed==0 and yaw>0 and not f.complete
    f.command(np.array([.086,.008,1.2]))
    assert f.complete

def test_large_turn_drift_rejects_route():
    f=DockFollower([[.1,0,1.2]])
    f.command(np.array([.09,0,0]))
    with pytest.raises(RuntimeError):f.command(np.array([.04,0,.3]))


def test_exploration_follower_reacquires_position_after_turn_drift():
    import numpy as np
    from cross_episode_sim.navigation.dock_follower import DockFollower
    f=DockFollower([[0,0,1.]],recover_turn_drift=True)
    f.command(np.array([0.,0.,0.]));assert f.rotating
    forward,yaw=f.command(np.array([.05,0.,0.]))
    assert not f.rotating and forward<0 and not f.complete
    f.command(np.array([.01,0.,0.]));assert f.rotating
    f.command(np.array([0.,0.,1.]));assert f.complete


def test_exploration_profile_shortens_same_route_without_skipping_turns():
    # Both profiles must finish the same translate/rotate path; faster crossing
    # uses the existing exploration limits, rather than changing its endpoints.
    def execute(speed,yaw,gain):
        f=DockFollower([[0,0,1.57],[0,1,1.57],[0,1,3.14],[-1,1,3.14]],
                       max_speed=speed,max_yaw_speed=yaw,approach_gain=gain)
        pose=np.zeros(3);velocity=np.zeros(2)
        for i in range(5000):
            command=np.asarray(f.command(pose))
            if f.complete:return i*.02,pose
            velocity+=np.clip(command-velocity,-np.array([.4,.8])*.02,np.array([.4,.8])*.02)
            pose[:2]+=velocity[0]*.02*np.array([np.cos(pose[2]),np.sin(pose[2])])
            pose[2]+=velocity[1]*.02
        raise AssertionError('Route did not complete')
    slow,_=execute(.2,.3,.8)
    fast,end=execute(.4,.6,1.2)
    assert fast < .8*slow
    assert np.linalg.norm(end[:2]-[-1,1])<.02


def test_room_transit_turn_does_not_chase_small_localization_jitter():
    f=DockFollower([[0,0,1.57],[0,1,1.57]],position_tolerance=.04,
                   recover_turn_drift=True,turn_drift_tolerance=.15)
    f.command(np.zeros(3))
    # A small translation correction during a turn must not repeatedly undo
    # that turn to chase the same intermediate transit waypoint.
    for yaw in np.linspace(.1,1.57,20):
        speed,_=f.command(np.array([0.,-.06,yaw]))
        if len(f.route)==2:assert speed==0.
    assert len(f.route)==1
