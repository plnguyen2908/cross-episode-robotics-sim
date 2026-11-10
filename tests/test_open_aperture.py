import numpy as np
import pytest
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.open_aperture import observed_open_aperture


@pytest.mark.parametrize('limits',[(-.8,0.),(0.,.8),(-.8,.8),(-.4,.4)])
@pytest.mark.parametrize('angle',[0.,.6])
def test_open_segment_midpoint_not_pair_midpoint(limits,angle):
    g=ObservedNavigationMap(MapConfig(extent_m=6,resolution=.025));g.floor_views[:]=2
    n=np.array([np.cos(angle),np.sin(angle)]);t=np.array([-n[1],n[0]])
    rr,cc=np.indices(g.floor_views.shape);xy=g.cell_to_world(np.stack([cc,rr],-1))
    along=xy@n;lateral=xy@t
    wall=(abs(along)<.07)&((lateral<=limits[0])|(lateral>=limits[1]))
    g.evidence[wall]=8
    seed=t*((limits[0]+limits[1])/2+.1)
    result=observed_open_aperture(g,seed,n)
    assert result is not None
    assert abs(np.asarray(result['center'])@t-np.mean(limits))<.05
    assert abs(result['width_m']-(limits[1]-limits[0]))<.075


def test_unknown_unbounded_and_closed_are_not_openings():
    g=ObservedNavigationMap(MapConfig(extent_m=6,resolution=.05))
    assert observed_open_aperture(g,[0,0],[1,0]) is None
    g.floor_views[:]=2
    assert observed_open_aperture(g,[0,0],[1,0]) is None
    g.evidence[:]=8
    assert observed_open_aperture(g,[0,0],[1,0]) is None
