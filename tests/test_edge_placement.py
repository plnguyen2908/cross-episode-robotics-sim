import numpy as np
import pytest
from cross_episode_sim.manipulation.edge_placement import supported_edge_position

@pytest.mark.parametrize('width,depth',[(.08,.16),(.24,.24),(.43,.33)])
def test_size_independent_edge_distance(width,depth):
    v=np.array([[-width/2,-depth/2,-.01],[width/2,depth/2,.01]])
    table=np.array([[1.,-6.,.78],[3.,-5.,.8]])
    placed=v+supported_edge_position(v,table)
    assert placed[:,1].max()==pytest.approx(-5.02)
    assert placed[:,2].min()==pytest.approx(.801)
    assert placed[:,0].mean()==pytest.approx(2.)

def test_oversize_rejected():
    with pytest.raises(ValueError):supported_edge_position(np.array([[-2,-2,0],[2,2,1]]),np.array([[-1,-1,0],[1,1,.8]]))
