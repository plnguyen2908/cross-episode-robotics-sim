import numpy as np
import pytest
from cross_episode_sim.perception.handle_acquisition import observed_handle_near


def test_tracked_target_requires_current_protruding_handle_not_just_a_wall():
    y,z=np.meshgrid(np.linspace(-.15,.15,41),np.linspace(.4,.7,41))
    panel=np.c_[np.full(y.size,.8),y.ravel(),z.ravel()]
    yy,zz=np.meshgrid(np.linspace(-.055,.055,24),np.linspace(.54,.56,6))
    handle=np.c_[np.full(yy.size,.76),yy.ravel(),zz.ravel()]
    target,info=observed_handle_near(np.r_[panel,handle],[.79,.01,.55])
    assert np.linalg.norm(handle-target,axis=1).min()<1e-10
    with pytest.raises(ValueError):observed_handle_near(panel,[.79,.01,.55])
