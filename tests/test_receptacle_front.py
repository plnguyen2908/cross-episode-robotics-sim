import numpy as np
from cross_episode_sim.perception.receptacle_front import observed_front,frontal_heading_error


def test_front_estimate_rotates_with_observed_scene_not_world_axes():
    y,z=np.meshgrid(np.linspace(-.45,.45,35),np.linspace(.3,.78,24))
    points=np.c_[np.full(y.size,.8),y.ravel(),z.ravel()]
    target=np.array([1.,0.,1.])
    for angle in (0.,.7,-1.3):
        c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        offset=np.array([.2,-.3,0.])
        result=observed_front(points@R.T+offset,R@target+offset,offset[:2])
        assert result is not None
        assert np.dot(result['normal_base'],R@[-1.,0.,0.])>.99
        assert frontal_heading_error([0,0,angle],result['normal_base'])<.01


def test_horizontal_patch_does_not_invent_a_counter_front():
    x,y=np.meshgrid(np.linspace(.5,1.,20),np.linspace(-.3,.3,20))
    assert observed_front(np.c_[x.ravel(),y.ravel(),np.full(x.size,.6)],[1,0,1]) is None


def test_refined_docks_center_interior_with_exact_front_normal():
    from cross_episode_sim.perception.receptacle_front import interior_dock_anchor,frontal_docks
    x,y=np.meshgrid(np.linspace(.7,.9,20),np.linspace(-.1,.1,20))
    floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,.93)]
    anchor=interior_dock_anchor(np.concatenate([floor,[[.6,.2,1.02]]*20]))
    assert abs(anchor[0]-.8)<.02 and anchor[2]==.93
    normal=np.array([-np.cos(.19),np.sin(.19),0.])
    for proposal in frontal_docks([anchor],normal,[.5,.6]):
        assert frontal_heading_error(proposal.pose,normal)<1e-10
        relative=anchor[:2]-proposal.pose[:2]
        assert abs(np.linalg.norm(relative)-proposal.distance)<1e-10
