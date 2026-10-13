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
