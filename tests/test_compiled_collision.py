import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.compiled_collision import sphere_box_collision


def test_compiled_predicate_matches_vectorized_rotated_box_distance():
    rng=np.random.default_rng(41)
    for _ in range(100):
        centers=rng.uniform(-1,1,(60,3));half=rng.uniform(.005,.2,(60,3))
        rotations=Rotation.random(60,random_state=rng).as_matrix()
        spheres=np.c_[rng.uniform(-2,2,(40,3)),rng.uniform(.002,.1,40)]
        world_half=np.einsum('nij,nj->ni',abs(rotations),half)
        delta=abs(np.einsum('sbi,bij->sbj',spheres[:,None,:3]-centers[None],rotations))-half[None]
        gap=np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(delta.max(-1),0)-spheres[:,None,3]
        i,j,d=sphere_box_collision(spheres,centers,half,rotations,world_half,.005,np.empty((0,4)))
        assert (i>=0)==bool((gap<.005).any())
        if i>=0:
            assert d<.005
            np.testing.assert_allclose(d,gap[i,j],atol=1e-12)


def test_bvh_preserves_collision_decisions_for_rotated_worlds():
    from cross_episode_sim.control.compiled_collision import build_box_bvh,sphere_box_bvh_collision
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(95)
    for count in (0,1,15,1000):
        centers=rng.uniform(-3,3,(count,3));half=rng.uniform(.001,.3,(count,3))
        rotations=Rotation.random(count,random_state=rng).as_matrix() if count else np.empty((0,3,3))
        world_half=np.einsum('nij,nj->ni',abs(rotations),half)
        bvh=build_box_bvh(centers,half,rotations)
        for _ in range(50):
            spheres=np.c_[rng.uniform(-4,4,(30,3)),rng.uniform(.001,.1,30)]
            supports=spheres[:2].copy();margin=.005
            a=sphere_box_collision(spheres,centers,half,rotations,world_half,margin,supports)
            b=sphere_box_bvh_collision(spheres,centers,half,rotations,margin,supports,*bvh)
            assert (a[0]>=0)==(b[0]>=0)
            if b[0]>=0:
                # Independently confirm the reported leaf really violates the margin.
                i,j,gap=b
                delta=abs((spheres[i,:3]-centers[j])@rotations[j])-half[j]
                expected=np.linalg.norm(np.maximum(delta,0))+min(max(delta),0)-spheres[i,3]
                np.testing.assert_allclose(gap,expected,atol=1e-14)


def test_point_box_minimum_matches_full_cloud_for_rotated_boxes():
    from cross_episode_sim.control.compiled_collision import point_box_minimum
    rng=np.random.default_rng(412)
    for _ in range(100):
        pts=rng.uniform(-1,1,(1000,3));center=rng.uniform(-1,1,3)
        axes=Rotation.random(random_state=rng).as_matrix();half=rng.uniform(.01,.4,3)
        delta=abs((pts-center)@axes)-half
        expected=(np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(1),0)-.004).min()
        actual=point_box_minimum(pts,center,axes,half)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
        assert (actual<.005)==(expected<.005)
    assert np.isinf(point_box_minimum(np.empty((0,3)),center,axes,half))
