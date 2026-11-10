import numpy as np
from cross_episode_sim.perception.aperture_support import aperture_support_points


def geometry():
    return dict(closed_handle=[.6,.3,.55],hinge=[.6,-.3,.55],closed_normal=[-1.,0.,0.])


def shelf(z):
    x,y=np.meshgrid(np.arange(.63,.95,.008),np.arange(-.26,.26,.008))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]


def test_selects_measured_upper_shelf_not_lower_floor_or_outside_counter():
    points=np.concatenate([shelf(.08),shelf(.37),shelf(.9),shelf(.4)+[0,1,0]])
    result=aperture_support_points(points,geometry())
    assert len(result)>100
    assert np.allclose(result[:,2],.37)
    assert np.max(abs(result[:,1]))<.3


def test_no_invented_support_from_vertical_panel_or_empty_view():
    x,y=np.meshgrid(np.arange(.1,.6,.01),np.arange(-.26,.26,.01))
    wall=np.c_[np.full(x.size,.8),y.ravel(),x.ravel()]
    assert len(aperture_support_points(wall,geometry()))==0
    assert len(aperture_support_points(np.empty((0,3)),geometry()))==0


def test_aperture_is_equivariant_to_base_rotation():
    theta=.8;c,s=np.cos(theta),np.sin(theta);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    offset=np.array([.2,-.1,0]);a=geometry();b={k:(R@np.array(v)+(0 if k=='closed_normal' else offset)).tolist() for k,v in a.items()}
    result=aperture_support_points(shelf(.37)@R.T+offset,b)
    assert len(result)>100 and np.allclose(result[:,2],.37)
