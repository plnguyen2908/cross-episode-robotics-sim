"""An unreachable lift can retry a supported grasp, never drop an airborne pot."""
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.controller.base import FridgeTransfer


@pytest.mark.parametrize('supported', [True, False])
def test_lift_ik_retry_requires_support(monkeypatch, supported):
    import cross_episode_sim.controller.base as base
    import cross_episode_sim.control.timing as timing
    monkeypatch.setattr(base, 'wait_for_open_gripper', lambda *a: None)
    monkeypatch.setattr(timing, 'gripper_open_ready', lambda c: True)
    events = []
    lifting = [False]
    def move(stage, pose):
        events.append(stage)
        if stage == 'lift bread vertically':
            lifting[0] = True
            raise RuntimeError('No nearby placement IK: unreachable lift')
    c = SimpleNamespace(
        args=SimpleNamespace(native_object=True, grasp_y_offset=0., grasp_depth=0.),
        embodiment=SimpleNamespace(open_gripper=lambda c: events.append('open')),
        report={'max_unintended_robot_penetration_m': 0.},
        prepare_pickup=lambda: None, bread_pose=lambda: np.eye(4),
        bread_vertices=lambda: np.array([[0.,0.,0.],[.1,.1,.1]]),
        support_contacts=lambda **kw: supported if lifting[0] else True,
        record=lambda **kw: None, make_planner=lambda: SimpleNamespace(names=[]),
        actuator_ids=lambda names: [], load_world=lambda: None,
        select_annotated_grasp=lambda: (np.eye(4),np.eye(4)),
        move=move, close_on_loaf=lambda: None, contacts=lambda: ['left','right'],
        tcp=lambda: np.eye(4), selected_annotation_variant=3,
        tick=lambda t: None, pick_payload=lambda: 'retried',
    )
    if supported:
        assert FridgeTransfer.pick_payload(c) == 'retried'
        assert c.physically_rejected_annotation_variants == {3}
        assert 'withdraw after non-liftable annotated grasp' in events
    else:
        with pytest.raises(RuntimeError, match='No nearby placement IK'):
            FridgeTransfer.pick_payload(c)
        assert events[-1] == 'lift bread vertically'
        assert not hasattr(c, 'physically_rejected_annotation_variants')
