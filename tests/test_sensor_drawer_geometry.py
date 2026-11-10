import numpy as np
from types import SimpleNamespace
from cross_episode_sim.control.sensor_drawer_geometry import adaptive_partitions,merge_occupied

def test_merge_never_fills_unobserved_voxels():
    vox=np.unique(np.random.default_rng(5).integers(-5,5,(800,3)),axis=0)
    emitted=[]
    for lo,hi in merge_occupied(vox):
        emitted.extend((x,y,z) for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1) for z in range(lo[2],hi[2]+1))
    assert len(emitted)==len(set(emitted))
    assert set(emitted)==set(map(tuple,vox))

def test_fine_near_robot_and_coarse_far_preserve_every_observation():
    points=np.array([[.13,.01,.01],[.131,.012,.012],[1.,1.,1.]])
    robot=SimpleNamespace(center=np.zeros(3),axes=np.eye(3),half=np.full(3,.1))
    parts=list(adaptive_partitions(points,[robot]))
    assert {r for lo,hi,r in parts}=={.002,.015}
    for point in points:
        assert any(np.all(point>=lo*r) and np.all(point<(hi+1)*r) for lo,hi,r in parts)
    assert all(r==.002 for lo,hi,r in parts if np.linalg.norm((lo+hi+1)*r/2)<.3)
