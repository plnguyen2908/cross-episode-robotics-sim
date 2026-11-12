from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.perception.mapped_support import support_landmark,mapped_support_detection


def scene():
    h=w=100;depth=np.ones((h,w));depth[:10]=1.8
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.8
    mask=np.zeros((h,w),bool);mask[35:65,40:60]=True
    camera=NS(depth_m=depth,valid_depth=np.ones((h,w),bool),robot_mask=mask,
        K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]]),T_base_optical=T)
    landmark=dict(label='dining table',position=[0.,0.,.8],observations=2,
        observed_bounds=[[-.4,-.4,.79],[.4,.4,.81]])
    return camera,landmark


def test_known_identity_uses_fresh_plane_excludes_robot_and_objects():
    camera,landmark=scene();camera.depth_m[65:80,65:80]=.85
    result=mapped_support_detection(camera,np.eye(4),landmark,'dining table')
    assert result is not None
    assert not result.mask[camera.robot_mask].any()
    assert not result.mask[65:80,65:80].any()
    assert not result.mask[:10].any()
    u,v=result.point;assert result.mask[v,u]


def test_map_frame_transform_and_wrong_landmark():
    camera,landmark=scene();T=np.eye(4);T[:2,3]=[4.,-2.]
    assert mapped_support_detection(camera,T,landmark,'dining table') is None
    landmark['position']=[4.,-2.,.8]
    landmark['observed_bounds']=(np.asarray(landmark['observed_bounds'])+[4,-2,0]).tolist()
    assert mapped_support_detection(camera,T,landmark,'dining table') is not None


def test_memory_identity_requires_confirmed_unambiguous_landmark():
    _,item=scene()
    assert support_landmark({'furniture':[item]},'dining table') is item
    assert support_landmark({'tables':[item]},'dining table') is item
    assert support_landmark({'tables':[item,item]},'dining table') is None
    assert support_landmark({'tables':[{**item,'observations':1}]},'dining table') is None
