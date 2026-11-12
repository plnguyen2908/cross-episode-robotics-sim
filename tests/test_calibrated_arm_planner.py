from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
from cross_episode_sim.control.calibrated_arm_planner import CalibratedArmPlanner
from cross_episode_sim.control.observed_hinged_planner import ObservedHingedPlanner
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner


@dataclass
class Box:
    name: str
    center: np.ndarray
    axes: np.ndarray
    half: np.ndarray


def configured(monkeypatch, *, other_clear=True, box_clear=True):
    planner=object.__new__(CalibratedArmPlanner)
    planner._checked_q=np.zeros((1,7))
    planner.arm_box_spheres=np.array([0])
    planner.arm_box_names={'arbitrary_collision_mesh_name'}
    box=Box('arbitrary_collision_mesh_name',np.zeros(3),np.eye(3),np.ones(3)*.02)
    planner.geometry=SimpleNamespace(boxes=lambda q:[box])
    checks=[]
    def remainder(self,spheres,margin,base_offset,base_yaw):
        assert spheres[0,0,3]==-1
        assert spheres[0,1,3]==.03  # unrelated robot geometry remains checked
        assert margin==.005
        return other_clear
    monkeypatch.setattr(ObservedHingedPlanner,'world_clear',remainder)
    monkeypatch.setattr(ObservedGraspPlanner,'world_clear',lambda *args:False)
    def exact(box,margin):
        checks.append((box,margin));planner.last_rejection={'kind':'observed_payload_box'}
        return box_clear
    planner.world_clear_box=exact
    return planner,checks


def test_covering_sphere_overlap_uses_shape_box_without_reducing_margin(monkeypatch):
    planner,checks=configured(monkeypatch)
    spheres=np.array([[[0.,0.,0.,.05],[1.,0.,0.,.03]]])
    original=spheres.copy()
    assert planner.world_clear(spheres,base_offset=np.array([.1,0,0]))
    np.testing.assert_array_equal(spheres,original)
    np.testing.assert_allclose(checks[0][0].center,[.1,0,0])
    assert checks[0][1]==.005


def test_actual_arm_box_overlap_is_rejected(monkeypatch):
    planner,checks=configured(monkeypatch,box_clear=False)
    assert not planner.world_clear(np.array([[[0.,0.,0.,.05],[1.,0.,0.,.03]]]))
    assert checks and planner.last_rejection['narrow_phase']=='calibrated_arm_box'


def test_non_arm_obstacles_are_not_bypassed(monkeypatch):
    planner,checks=configured(monkeypatch,other_clear=False)
    assert not planner.world_clear(np.array([[[0.,0.,0.,.05],[1.,0.,0.,.03]]]))
    assert not checks
