from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.control.observed_hinged_planner import ObservedHingedPlanner
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner
from cross_episode_sim.navigation.shaped_docking import RobotBox

class Indices:
    def cpu(self):return self
    def numpy(self):return np.array([0])

def planner(x):
    box=SimpleNamespace(name='obstacle',pose=[x,.2,.23,1,0,0,0],dims=[.01,.01,.01])
    p=ObservedHingedPlanner.__new__(ObservedHingedPlanner)
    ObservedGraspPlanner.__init__(p,None,None,[box])
    p.base_sphere_ids=Indices()
    p.base_boxes=[RobotBox('robot_0/base/collision',np.array([0.,0.,.1675]),np.eye(3),np.array([.2405,.2405,.1675]))]
    return p


def test_exact_base_removes_covering_sphere_false_positive():
    p=planner(.28);spheres=np.array([[[.224,.199,.234,.059]]])
    assert not ObservedGraspPlanner.world_clear(p,spheres)
    assert p.world_clear(spheres)


def test_actual_base_overlap_and_translated_base_still_rejected():
    spheres=np.array([[[.224,.199,.234,.059]]])
    assert not planner(.242).world_clear(spheres)
    assert not planner(.28).world_clear(spheres,base_offset=[.05,0,0])


def test_arm_spheres_are_not_exempted():
    spheres=np.array([[[.224,.199,.234,.059],[.28,.2,.23,.02]]])
    assert not planner(.28).world_clear(spheres)


@pytest.mark.parametrize('raises',[False,True])
def test_arm_graph_prechecks_fixed_base_and_restores_full_spheres(raises):
    import torch
    p=ObservedHingedPlanner.__new__(ObservedHingedPlanner)
    p.panel_points=np.empty((0,3));p.base_boxes=[object()]
    checked=[];p.world_clear_box=lambda box:checked.append(box) or True
    spheres=torch.tensor([[[0.,0.,0.,.1],[1.,0.,0.,.2]]])
    original=spheres.clone()
    cfg=SimpleNamespace(link_spheres=spheres,get_sphere_index_from_link_name=lambda n:torch.tensor([0]))
    cfg.disable_link_spheres=lambda n:spheres.__setitem__((slice(None),0,3),-100.)
    def search(*args,**kwargs):
        assert len(checked)==1 and spheres[0,0,3]<0
        assert torch.equal(spheres[:,1],original[:,1])
        if raises:raise RuntimeError('search failed')
        return 'proposal'
    p.arm=SimpleNamespace(planner=SimpleNamespace(plan_cspace=search,
        kinematics=SimpleNamespace(config=SimpleNamespace(kinematics_config=cfg))))
    if raises:
        with pytest.raises(RuntimeError,match='search failed'):p.cspace_search(None,None)
    else:assert p.cspace_search(None,None)=='proposal'
    assert torch.equal(spheres,original)
    p.world_clear_box=lambda box:False
    with pytest.raises(RuntimeError,match='Fixed base overlaps'):p.cspace_search(None,None)
    assert torch.equal(spheres,original)


def test_closed_finger_boxes_replace_only_finger_covering_spheres():
    p=planner(.28)
    finger=RobotBox('robot_0/gripper/left_pad1',np.array([.24,.2,.23]),np.eye(3),np.array([.01,.01,.01]))
    p.finger_geometry=SimpleNamespace(boxes=lambda q:[finger])
    p.finger_sphere_ids=np.array([1]);p._checked_q=np.zeros((1,7))
    spheres=np.array([[[.224,.199,.234,.059],[.24,.2,.23,.04]]])
    assert p.world_clear(spheres)
    finger.center[0]=.275
    assert not p.world_clear(spheres)


def test_rigid_panel_index_reused_without_changing_nearest_obstacle():
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    p=ObservedHingedPlanner.__new__(ObservedHingedPlanner)
    rng=np.random.default_rng(42)
    p.panel_points=rng.normal(size=(2000,3))
    queries=rng.normal(size=(100,3))
    tree=None
    for angle in np.linspace(-1.5,1.5,21):
        R=Rotation.from_euler('z',angle).as_matrix();center=np.array([.3,-.2,0.])
        moved=(p.panel_points-center)@R.T+center
        expected,indices=cKDTree(moved).query(queries)
        actual,found=p.panel_nearest(queries,(R,center))
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-14)
        np.testing.assert_array_equal(found,indices)
        if tree is not None:assert tree is p._panel_tree
        tree=p._panel_tree
    p.panel_points=p.panel_points+1
    p.panel_nearest(queries)
    assert p._panel_tree is not tree
