from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.evaluation.object_targets import ObjectTargets, object_type


@pytest.mark.parametrize('key,expected', [
    ('molmo__Plate_24', 'Plate'), ('molmo__RoboTHOR_apple_ai2_1_v', 'Apple'),
    ('robocasa__lightwheel__salt_and_pepper_shaker__SaltShaker004', 'SaltShaker'),
    ('molmo__Vase_Medium_3', 'Vase'),
])
def test_scene_type_names(key, expected):
    assert object_type({'key': key}) == expected


def test_targets_include_distractors_without_changing_manifest():
    manifest = dict(bindings=[], scattered_objects=[dict(body='plate', key='molmo__Plate_24')])
    c = SimpleNamespace(manifest=manifest, base_pose=lambda: np.zeros(3),
                        data=SimpleNamespace(body=lambda _: SimpleNamespace(xpos=np.ones(3))))
    targets = ObjectTargets(c)
    assert targets.types() == ['Plate']
    assert targets.candidates('Plate') == ['plate']
    assert targets.candidates('Mug') == []
    assert manifest == dict(bindings=[], scattered_objects=[dict(body='plate', key='molmo__Plate_24')])


@pytest.mark.parametrize('carrying', [False, True])
def test_source_approach_preserves_payload_and_passes_collision_mode(carrying):
    from unittest.mock import Mock
    c = SimpleNamespace(
        manifest=dict(bindings=[dict(body='cup', asset='Mug_1', support='counter')]),
        holding_loaf=carrying, attached=carrying, object_name='held_box',
        data=SimpleNamespace(body=lambda _: SimpleNamespace(xpos=np.ones(3))),
        task_supports={'kitchen': 'counter'}, task_outward={}, table_bids={'counter': {1}},
        descendants=lambda _: {1}, tuck_for_navigation=Mock(), navigate_to_site=Mock())
    targets = ObjectTargets(c)
    targets._navigate('cup')
    assert c.navigate_to_site.call_args.args[2] == carrying
    assert c.object_name == 'held_box'
    assert c.holding_loaf == carrying
