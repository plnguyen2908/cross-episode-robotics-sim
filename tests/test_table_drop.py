import numpy as np
from cross_episode_sim.control.table_drop import table_drop_candidates


def test_small_drop_preserves_orientation_and_bottom_clearance():
    x,y=np.meshgrid(np.linspace(.3,.9,61),np.linspace(-.3,.3,61))
    surface=np.c_[x.ravel(),y.ravel(),np.full(x.size,.75)]
    tcp=np.eye(4);tcp[:3,3]=[.3,0,1.]
    held=np.array([[x,y,z] for x in (.26,.34) for y in (-.04,.04) for z in (.88,.98)])
    rows=table_drop_candidates(surface,surface,held,tcp)
    assert rows
    for row in rows:
        pose=np.asarray(row['tcp'])
        np.testing.assert_allclose(pose[:3,:3],tcp[:3,:3])
        released=held+pose[:3,3]-tcp[:3,3]
        assert abs(released[:,2].min()-.8)<1e-9
        assert released[:,0].min()>=.3 and released[:,0].max()<=.9
        assert not row['learned_proposal']
    occupied=surface.copy();occupied[:,2]+=.08
    assert not table_drop_candidates(surface,np.vstack([surface,occupied]),held,tcp)
