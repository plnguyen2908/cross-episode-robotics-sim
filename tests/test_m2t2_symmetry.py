import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.m2t2_policy import parallel_gripper_symmetries

def test_parallel_jaw_flip_preserves_contact_approach_and_closing_line():
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.3,.2,.1]).as_matrix();T[:3,3]=[.4,.1,.9]
    original={'tcp':T.tolist(),'confidence':.9,'contact':[.4,.1,.9]}
    a,b=parallel_gripper_symmetries([original]);B=np.array(b['tcp'])
    assert np.allclose(B[:3,3],T[:3,3]) and np.allclose(B[:3,2],T[:3,2])
    assert np.allclose(B[:3,1],-T[:3,1]) and np.isclose(np.linalg.det(B[:3,:3]),1)
    assert b['parallel_jaw_flip'] and not a['parallel_jaw_flip']
    assert original['tcp']==T.tolist()
