import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.container_footprint import interior_placement_candidates
from cross_episode_sim.perception.basket_floor_point import pointed_basket_floor


def test_long_payload_upright_candidate_preserves_surface_support():
    angle=np.linspace(0,2*np.pi,24,endpoint=False)
    p=np.array([[.025*np.cos(a),.025*np.sin(a),z] for z in np.linspace(-.08,.08,10) for a in angle])
    tilt=Rotation.from_euler('y',60,degrees=True).as_matrix()
    center=np.array([.6,0.,1.2]);held=p@tilt.T+center
    tcp=np.eye(4);tcp[:3,3]=center
    xx,yy=np.meshgrid(np.arange(.44,.77,.01),np.arange(-.16,.17,.01))
    surface=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,.9)]
    rows=interior_placement_candidates(surface,surface,held,tcp,.05,prefer_upright=True)
    assert rows
    goal=np.array(rows[0]['tcp']);transformed=(held-center)@goal[:3,:3].T+goal[:3,3]
    axis=np.linalg.svd(transformed-transformed.mean(0),full_matrices=False)[2][0]
    assert abs(axis[2])>.999
    assert abs(transformed[:,2].min()-.9)<1e-8


def test_floor_point_avoids_small_wall_crop():
    wall=np.zeros((40,50),bool);wall[10:13,20:30]=True
    floor=np.zeros_like(wall);floor[13:25,20:30]=True
    class Pointer:
        def point(self,image,target):return {'points':[[25,20]]}
    ds=pointed_basket_floor(np.zeros((40,50,3),np.uint8),'orange basket','interior',Pointer(),[wall,floor])
    assert len(ds)==1
    np.testing.assert_array_equal(ds[0].mask,floor)
    assert ds[0].point==(25,20)
