import numpy as np
import pytest
import yaml
from cross_episode_sim.generation.native_room_layout import build_three_room_layout
from cross_episode_sim.paths import robocasa_dir


@pytest.mark.parametrize('layout_id', (1, 2, 3, 4, 9))
def test_entry_apron_extends_shell_and_preserves_native_fixtures(layout_id):
    original = yaml.safe_load((robocasa_dir()/f'models/assets/scenes/kitchen_layouts/test/layout{layout_id:03d}.yaml').read_text())
    layout, meta = build_three_room_layout(np.random.default_rng(7), layout_id)
    clearance = meta['kitchen_entry_clearance_m']
    assert clearance == 1.8
    old_floor = original['room']['floor'][0]
    old_front = old_floor['pos'][1]-old_floor['size'][1]
    assert meta['room_bounds']['kitchen'][1] == pytest.approx(old_front-clearance)
    assert meta['doorway']['center'][1] == pytest.approx(old_front-clearance)
    for name, group in original.items():
        if name != 'room':
            assert layout[name] == group
    for floor, old in zip(layout['room']['floor'], original['room']['floor']):
        assert floor['pos'][1]+floor['size'][1] == pytest.approx(old['pos'][1]+old['size'][1])
