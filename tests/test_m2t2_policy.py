import numpy as np
from cross_episode_sim.control.m2t2_policy import grasp_to_tcp


def test_grasp_contact_filter_rejects_neighbor_inside_centroid_radius():
    from cross_episode_sim.control.m2t2_policy import filter_observed_contacts
    points=np.array([[.4,0,.95],[.4,0,1.],[.4,.02,1.]])
    wanted={'contact':[.4,0,1.],'confidence':.5}
    neighbor={'contact':[.4,.12,1.],'confidence':.9}
    assert filter_observed_contacts([neighbor,wanted],points)==[wanted]
    assert filter_observed_contacts([],points)==[]


def test_grasp_convention_preserves_contact_center_and_approach():
    g=np.eye(4);g[:3,3]=[.5,.1,.9]
    tcp=grasp_to_tcp(g)
    np.testing.assert_allclose(tcp[:3,3],[.5,.1,1.0034])
    np.testing.assert_allclose(tcp[:3,2],g[:3,2])
    np.testing.assert_allclose(tcp[:3,1],g[:3,0])
    np.testing.assert_allclose(tcp[:3,:3].T@tcp[:3,:3],np.eye(3))
    assert np.isclose(np.linalg.det(tcp[:3,:3]),1.)
    np.testing.assert_allclose(g[:3,3],[.5,.1,.9])


def test_voxel_compression_keeps_holes_and_exact_coverage():
    from cross_episode_sim.control.observed_grasp_planner import merge_voxels
    cells=np.array([(x,y,z) for x in range(4) for y in range(3) for z in range(2) if (x,y,z)!=(1,1,1)])
    recovered=[]
    for lo,hi in merge_voxels(cells):
        recovered += [(x,y,z) for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1) for z in range(lo[2],hi[2]+1)]
    assert len(recovered)==len(cells)
    assert set(recovered)==set(map(tuple,cells))


def test_smooth_path_limits_and_endpoints():
    from cross_episode_sim.control.observed_grasp_planner import smooth_joint_path
    a=np.zeros(7);b=np.array([1.,-.5,.3,0,0,.4,.7]);q=smooth_joint_path(a,b)
    np.testing.assert_allclose(q[0],a);np.testing.assert_allclose(q[-1],b)
    assert np.max(np.abs(np.diff(q,axis=0)/.02))<=1.001
    assert np.max(np.abs(np.diff(q,n=2,axis=0)/.02**2))<=2.001


def test_observed_contact_component_preserves_counter_and_other_object():
    from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
    rng=np.random.default_rng(0)
    # Dense volumetric fixtures stand in for observed surfaces for connectivity testing.
    obj=rng.uniform([.58,-.06,.94],[.70,.06,1.02],(6000,3))
    other=rng.uniform([.82,-.04,.94],[.90,.04,1.02],(2000,3))
    counter=np.c_[rng.uniform(.4,1.,1000),rng.uniform(-.2,.2,1000),np.full(1000,.915)]
    points=np.concatenate([obj,other,counter]);mask=observed_target_mask(points,np.array([.68,0,.98]),.915)
    assert mask[:6000].mean()>.99
    assert not mask[6000:].any()


def test_observed_box_checker_matches_brute_force():
    from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner
    from scipy.spatial.transform import Rotation
    from scipy.spatial import cKDTree
    from types import SimpleNamespace
    rng=np.random.default_rng(7);checker=ObservedGraspPlanner.__new__(ObservedGraspPlanner)
    checker.centers=rng.uniform(-1,1,(30,3));checker.half=rng.uniform(.01,.2,(30,3))
    checker.rotations=Rotation.random(30,random_state=rng).as_matrix()
    checker.tree=cKDTree(checker.centers);checker.max_diagonal=np.linalg.norm(checker.half,axis=1).max()
    checker.boxes=[SimpleNamespace(name=str(i)) for i in range(30)]
    for _ in range(20):
        spheres=np.c_[rng.uniform(-2,2,(20,3)),rng.uniform(.01,.07,20)]
        closest=float('inf')
        for center,half,R in zip(checker.centers,checker.half,checker.rotations):
            d=np.abs((spheres[:,:3]-center)@R)-half
            gap=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(1),0)-spheres[:,3]
            closest=min(closest,gap.min())
        assert checker.world_clear(spheres)==(closest>=.005)


def test_private_contact_scoring_includes_nested_object_meshes():
    from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup
    parents=[0,0,1,2,0,4,0]
    np.testing.assert_array_equal(contact_root_lookup(parents,[1,4]),[-1,1,1,1,4,4,-1])


def test_stationary_base_floor_support_does_not_exempt_arm_or_obstacles():
    from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner
    from scipy.spatial import cKDTree
    from types import SimpleNamespace
    checker=ObservedGraspPlanner.__new__(ObservedGraspPlanner)
    checker.centers=np.array([[0.,0.,0.]])
    checker.half=np.array([[.1,.1,.01]])
    checker.rotations=np.eye(3)[None]
    checker.tree=cKDTree(checker.centers)
    checker.max_diagonal=np.linalg.norm(checker.half[0])
    checker.boxes=[SimpleNamespace(name='observed_floor')]
    checker.fixed_floor_support_spheres=np.array([[0.,0.,.03,.03]])
    assert checker.world_clear(np.array([[0.,0.,.03,.03]]))
    # A moving sphere at another position is not certified support.
    assert not checker.world_clear(np.array([[.01,0.,.03,.03]]))
    # Even the base sphere must avoid an obstacle extending above floor level.
    checker.half[0,2]=.06
    checker.max_diagonal=np.linalg.norm(checker.half[0])
    assert not checker.world_clear(np.array([[0.,0.,.03,.03]]))


def test_payload_box_does_not_use_oversized_bounding_sphere():
    from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner
    from cross_episode_sim.navigation.shaped_docking import RobotBox
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    p=ObservedGraspPlanner.__new__(ObservedGraspPlanner)
    p.centers=np.array([[0.,.15,0.]])
    p.half=np.array([[.02,.02,.02]])
    p.rotations=np.eye(3)[None]
    p.boxes=[SimpleNamespace(name='table_edge')]
    box=RobotBox('payload',np.zeros(3),np.eye(3),np.array([.2,.03,.03]))
    assert p.world_clear_box(box)  # Bounding sphere would falsely intersect.
    turned=RobotBox('payload',np.zeros(3),Rotation.from_euler('z',90,degrees=True).as_matrix(),box.half)
    assert not p.world_clear_box(turned)
    p.centers[0]=[.225,0.,0.]
    assert not p.world_clear_box(box,margin=.01)
