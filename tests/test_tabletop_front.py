import numpy as np
from cross_episode_sim.perception.tabletop_front import tabletop_front


def test_target_on_support_does_not_select_lower_chair_plane():
    x,y=np.meshgrid(np.linspace(.4,1.4,101),np.linspace(-.5,.5,101))
    table=np.c_[x.ravel(),y.ravel(),np.full(x.size,.77)]
    chair=table.copy();chair[:,2]=.49
    front=tabletop_front(np.vstack([table,chair]),[.55,0,.77])
    assert abs(front['height']-.77)<1e-8
    np.testing.assert_allclose(front['normal_base'],[-1,0,0],atol=1e-5)
