from types import SimpleNamespace
from PIL import Image
from cross_episode_sim.perception.target_language import visual_target_text
from cross_episode_sim.perception.molmo_pointing import MolmoPointer


def test_qualified_targets_keep_location_and_are_idempotent():
    text = visual_target_text('Find packaged snacks on the dining table')
    assert 'rectangular blocks' in text
    assert text.endswith('on the dining table')
    assert visual_target_text(text) == text
    for target in ('cup', 'bowl', 'microwave', 'purple basket', 'packaged drink'):
        assert visual_target_text(target) == target


def test_direct_redock_pointer_expands_before_remote_dispatch():
    calls = []
    pointer = MolmoPointer.__new__(MolmoPointer)
    pointer.remote = SimpleNamespace(call=lambda **kw: calls.append(kw) or {'points': []})
    pointer.point(Image.new('RGB', (40, 30)), 'packaged snack')
    assert 'plain solid-colored blocks' in calls[0]['target']
    assert 'without labels' in calls[0]['target']


def test_segmentation_verifier_retains_category_and_explains_visible_shape():
    from cross_episode_sim.perception.daaam_grounding import category_prompt
    prompt=category_prompt(('packaged snack',))
    assert '"packaged snack"' in prompt
    assert 'plain solid-colored blocks' in prompt
    assert '"other"' in prompt and '"unknown"' in prompt


def test_snack_point_on_calibrated_robot_is_not_a_navigation_target():
    import numpy as np
    from cross_episode_sim.perception.visual_targets import Detection, locate_detection
    camera=SimpleNamespace(depth_m=np.ones((20,20)),valid_depth=np.ones((20,20),bool),
        K=np.array([[100.,0.,10.],[0.,100.,10.],[0.,0.,1.]]),
        T_base_optical=np.eye(4),robot_mask=None,
        robot_bounds_base=((np.eye(4),np.array([0.,0.,1.]),np.array([.2,.2,.2])),))
    detection=Detection('packaged snack',1.,(5.,5.,15.,15.))
    assert locate_detection(camera,detection,np.eye(4)) is None
    camera.robot_bounds_base=((np.eye(4),np.array([2.,0.,1.]),np.array([.2,.2,.2])),)
    assert locate_detection(camera,detection,np.eye(4)) is not None
