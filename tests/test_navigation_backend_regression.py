"""Guard the successful shared backend against exploratory policy replacements."""
import ast
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script,backend',[
    ('test_fixed_map_transit.py','FixedRoomWaypointFollower'),
    ('test_m2t2_carry_dock.py','FixedMapNavigator'),
])
def test_production_stages_keep_existing_graph_follower(script,backend):
    tree=ast.parse((ROOT/'scripts'/script).read_text())
    calls=[node.func for node in ast.walk(tree) if isinstance(node,ast.Call)]
    names={node.id for node in calls if isinstance(node,ast.Name)}
    attrs={node.attr for node in calls if isinstance(node,ast.Attribute)}
    assert backend in names and 'ObservedUndock' in names
    assert 'SuccessGraphTransit' not in names
    assert 'configure_frozen_localization' not in attrs


def test_adjacent_leaves_use_the_same_standard_door_approach():
    tree=ast.parse((ROOT/'scripts/demo_room_coverage_exploration.py').read_text())
    names=[node.func.id for node in ast.walk(tree)
           if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    assert names.count('PortalApproach')==2
    assert 'LocalDoorApproach' not in names
