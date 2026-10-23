from copy import deepcopy
from pathlib import Path
import pytest
from cross_episode_sim.generation.sampled_house import resolve_route


def graph():
    return {'connections':[dict(rooms=['kitchen','dining'],door='room_door_8'),
        dict(rooms=['kitchen','living_2'],door='room_door_3'),
        dict(rooms=['living_2','dining'],door='room_door_12')]}


def test_routes_use_graph_not_fixed_door_numbers():
    a=resolve_route(graph(),'open_shortcut');b=resolve_route(graph(),'living_room_detour')
    assert a['initial_states']==b['initial_states']
    assert a['doors_to_open']==['room_door_8']
    assert b['doors']==['room_door_3','room_door_12'] and not b['doors_to_open']


def test_no_fake_detour_on_tree():
    g=graph();g['connections'].pop()
    with pytest.raises(ValueError,match='cycle'):resolve_route(g,'living_room_detour')


def test_random_suite_is_reproducible_and_not_seven_fixed_cases():
    from cross_episode_sim.generation.variation_sampling import sample_file
    p=Path('configs/composite_sampling/randomized/set_table.yaml')
    a=sample_file(p);assert a==sample_file(p)
    assert len({c['scene']['base_layout']['seed'] for c in a})==7
    assert len({tuple(c['sampling']['active_controls']) for c in a})>1
    assert all(5<=c['scene']['base_layout']['room_count']<=7 for c in a)


def test_resolved_route_prefix_uses_atomic_calls_only():
    from cross_episode_sim.tasks.composites.room_route import route_plan
    c={'task':'set_table','scene':{'base_layout':{}},'resolved_route':resolve_route(graph(),'open_shortcut')}
    c['scene']['base_layout']={'seed':1}
    plan=[{'skill':'wait','arguments':{'seconds':1}}]
    result=route_plan(c,plan)
    assert [s['skill'] for s in result]==['navigate','open_door','navigate','open_door','wait']
    assert result[0]['arguments']['target']=='room_door_8_left'


def test_seed_count_overrides_and_continuous_rgb():
    import random
    from cross_episode_sim.generation.variation_sampling import sample_file,draw
    p=Path('configs/composite_sampling/randomized/set_table.yaml')
    a=sample_file(p,seed=123,count=2);b=sample_file(p,seed=124,count=2)
    assert len(a)==2 and a[0]['variant']=='sampled'
    assert a[0]['scene']['base_layout']['seed']!=b[0]['scene']['base_layout']['seed']
    rgb=draw({'rgb_uniform':[[.2,.3,.4],[.5,.6,.7]]},random.Random(7))
    assert all(lo<=v<=hi for v,lo,hi in zip(rgb,[.2,.3,.4],[.5,.6,.7]))
    with pytest.raises(ValueError):draw({'rgb_uniform':[[0,0,0],[2,1,1]]},random.Random(7))
