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
