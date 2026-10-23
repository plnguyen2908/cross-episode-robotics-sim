from copy import deepcopy
import pytest
from cross_episode_sim.generation.base_layout import sample_layout, validate_partition, paths_between


def samples():
    result=[]
    for seed in range(120):
        try:result.append(sample_layout(seed,[0.,-4.8,5.5,0.],9))
        except ValueError:pass
    return result


def test_sampler_reproducible_diverse_and_before_variations():
    layouts=samples()
    assert len(layouts)>20
    assert len({tuple(tuple(r['bounds']) for r in l['rooms']) for l in layouts})==len(layouts)
    assert len({tuple(tuple(e['rooms']) for e in l['connections']) for l in layouts})>5
    for layout in layouts:
        assert layout==sample_layout(layout['seed'],[0.,-4.8,5.5,0.],9)
        assert layout['stage']=='base_layout_before_task_variations'
        validate_partition(layout)
        assert layout['footprint']['corner_count']>=8
        assert layout['footprint']['bounding_box_fill_ratio']<=.9
        roles=[r['room_type'] for r in layout['rooms']]
        assert roles.count('kitchen')==roles.count('dining')==1
        assert 1<=roles.count('bedroom')<=2
        living=[r for r in layout['rooms'] if r['room_type']=='living']
        assert len({(r['furnishing_variant'],r['furnishing_rotation_quarters']) for r in living})==len(living)
        assert all(e['initial_state']=='open' for e in layout['connections'])
        relevant={n for path in paths_between(layout['connections'],'kitchen','dining') for n in path}
        assert all(r['name'] in relevant or r['room_type']=='bedroom' for r in layout['rooms'])


def test_reject_invalid_partition_and_parameters():
    layout=deepcopy(samples()[0]);layout['rooms'][1]['bounds']=layout['rooms'][0]['bounds']
    with pytest.raises(ValueError,match='Overlapping'):validate_partition(layout)
    for kwargs in [dict(room_count=3),dict(min_room_side=2),dict(loop_probability=1.2)]:
        with pytest.raises(ValueError):sample_layout(1,[0.,-4.8,5.5,0.],**kwargs)
