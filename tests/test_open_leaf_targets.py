"""Targets inside an appliance refuse points on its remembered open door (fix151, call 17)."""
from types import SimpleNamespace
from cross_episode_sim.code_policy.redocking import on_open_leaf_of_container

# fix151's own memory and points, run map frame
LEAF = {'microwave': dict(hinge=[0.062, -2.004], open_handle=[0.479, -1.798], top=1.403, bottom=0.803)}
RUNTIME = SimpleNamespace(open_leaves=LEAF)


def test_pointed_inner_face_of_the_open_door_is_refused():
    assert on_open_leaf_of_container(RUNTIME, 'food container inside microwave', [0.344, -1.993, 1.190])


def test_the_food_on_the_turntable_is_kept():
    assert not on_open_leaf_of_container(RUNTIME, 'food container inside microwave', [-0.238, -2.014, 0.990])


def test_other_targets_and_closed_appliances_are_untouched():
    assert not on_open_leaf_of_container(RUNTIME, 'handle on microwave', [0.344, -1.993, 1.190])
    assert not on_open_leaf_of_container(RUNTIME, 'cold drink inside fridge', [0.344, -1.993, 1.190])
    assert not on_open_leaf_of_container(SimpleNamespace(), 'food container inside microwave', [0.344, -1.993, 1.190])


def test_points_above_or_below_the_leaf_band_are_kept():
    assert not on_open_leaf_of_container(RUNTIME, 'food container inside microwave', [0.344, -1.993, 1.60])
