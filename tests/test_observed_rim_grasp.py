import numpy as np
from cross_episode_sim.control.observed_rim_grasp import side_rim_candidates


def test_side_rim_uses_observed_edges_and_keeps_jaws_level():
    x=np.linspace(.72,.86,40);y=np.linspace(-.07,.07,40)
    rim=np.vstack([np.c_[x,np.full(40,s*.07),np.ones(40)] for s in (-1,1)]+
                  [np.c_[np.full(40,v),y,np.ones(40)] for v in (.72,.86)])
    rows=side_rim_candidates(rim,[-1,0,0],55,-.001)
    assert rows
    for row in rows:
        t=np.asarray(row['tcp']);r=t[:3,:3]
        assert np.allclose(r.T@r,np.eye(3)) and np.linalg.det(r)>.999
        assert abs(t[1,3])==.07 and .72<=t[0,3]<=.86
        assert abs(r[2,1])<1e-9 and r[0,2]>0 and r[2,2]<0
        assert abs(t[2,3]-.999)<1e-9
    assert side_rim_candidates(np.vstack([rim,rim-[0,0,.3]]),[-1,0,0],55,0)==[]
