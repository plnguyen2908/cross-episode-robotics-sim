from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.evaluation.any_vessel import AnyVesselInterface


def make_interface(tmp_path):
    objects = [dict(role='cup_05', body='near', asset='Mug_1'),
               dict(role='cup_01', body='far', asset='Mug_1'),
               dict(role='bowl_03', body='bowl', asset='Bowl_2')]
    c = SimpleNamespace(output=tmp_path, holding_loaf=False, attached=False, object_name='',
                        manifest=dict(bindings=objects, full_breakfast=dict(interchangeable_vessels=True)),
                        info=lambda r: next(b for b in objects if b['role'] == r),
                        base_pose=lambda: np.zeros(3), served=lambda r: False,
                        data=SimpleNamespace(body=lambda name: SimpleNamespace(xpos=np.array(
                            [1 if name == 'near' else 2, 0, 0]))))
    class Base:
        def __init__(self):
            self.calls = []; self.failures = {}
        def list_skills(self):
            return [dict(name=name, description=name, parameters=dict(type='object', additionalProperties=False,
                properties=dict(role=dict(type='string', enum=[b['role'] for b in objects]),
                                **{arg:dict(type='string',enum=values)}), required=['role', arg]))
                for name, arg, values in [('navigate','destination',['source','serving','spout']),
                                         ('pick','source',['surface','spout']),
                                         ('place','destination',['serving','spout'])]]
        def validate(self, name, args):
            assert args['role'] in [b['role'] for b in objects]
        def execute_skill(self, name, args):
            self.calls.append((name, deepcopy(args)))
            error = self.failures.get((name,args['role']))
            if error: return dict(success=False, error=dict(kind=error))
            if name == 'pick':
                c.holding_loaf=True; c.object_name=c.info(args['role'])['body']
            if name == 'place': c.holding_loaf=False
            return dict(success=True)
    base = Base()
    return AnyVesselInterface(c,base), base, c


def test_type_catalog_selection_and_sticky_instance(tmp_path):
    api, base, c = make_interface(tmp_path)
    catalog = api.list_skills()
    assert 'enum' not in catalog[0]['parameters']['properties']['asset_type']
    assert 'role' not in catalog[0]['parameters']['properties']
    assert 'role' in base.list_skills()[0]['parameters']['properties']
    assert api.execute_skill('pick',dict(asset_type='Mug',source='surface'))['error']['kind']=='source_navigation_required'
    for name,args in [('navigate',dict(destination='source')),('pick',dict(source='surface')),
                      ('navigate',dict(destination='spout')),('place',dict(destination='spout')),
                      ('navigate',dict(destination='spout')),('pick',dict(source='spout')),
                      ('navigate',dict(destination='serving')),('place',dict(destination='serving'))]:
        assert api.execute_skill(name, dict(asset_type='Mug',**args))['success']
    assert all(args['role']=='cup_05' for _,args in base.calls)
    assert api.execute_skill('navigate',dict(asset_type='Mug',destination='source'))['success']
    assert base.calls[-1][1]['role']=='cup_01'


def test_failed_candidate_fallback_and_exhaustion(tmp_path):
    api,base,c=make_interface(tmp_path)
    base.failures[('navigate','cup_05')]='execution_failed'
    assert api.execute_skill('navigate',dict(asset_type='Mug',destination='source'))['success']
    assert api.selected['Mug']=='cup_01'
    base.failures[('pick','cup_01')]='execution_failed'
    assert not api.execute_skill('pick',dict(asset_type='Mug',source='surface'))['success']
    assert api.execute_skill('navigate',dict(asset_type='Mug',destination='source'))['error']['kind']=='no_feasible_target'
    assert len(base.calls)==3


def test_ordering_failure_does_not_retire_candidates(tmp_path):
    api,base,c=make_interface(tmp_path)
    base.failures[('navigate','cup_05')]='precondition_failed'
    assert not api.execute_skill('navigate',dict(asset_type='Mug',destination='source'))['success']
    assert not api.retired and len(base.calls)==1
    with pytest.raises(ValueError):api.validate('pick',dict(role='cup_01',source='surface'))
    api.validate('pick',dict(asset_type='Plate',source='surface'))
    assert api.execute_skill('navigate',dict(asset_type='Plate'))['error']['kind']=='target_not_found'
    c.holding_loaf=True; c.object_name='bowl'
    assert api.execute_skill('navigate',dict(asset_type='Mug',destination='source'))['error']['kind']=='different_object_held'


def test_navigation_without_destination_approaches_source(tmp_path):
    api, base, _ = make_interface(tmp_path)
    nav = next(s for s in api.list_skills() if s['name'] == 'navigate')
    assert 'destination' not in nav['parameters']['required']
    assert nav['parameters']['properties']['destination']['default'] == 'source'
    assert 'asset_type only' in nav['description']
    assert api.execute_skill('navigate', {'asset_type': 'Mug'})['success']
    assert base.calls[-1] == ('navigate', {'role': 'cup_05', 'destination': 'source'})
    assert api.execute_skill('pick', {'asset_type': 'Mug', 'source': 'surface'})['success']
    assert base.calls[-1][1]['role'] == 'cup_05'


def test_pick_cannot_switch_type_after_navigation(tmp_path):
    api, base, c = make_interface(tmp_path)
    assert api.execute_skill('navigate', dict(asset_type='Bowl'))['success']
    assert api.execute_skill('navigate', dict(asset_type='Mug'))['success']
    before = len(base.calls)
    result = api.execute_skill('pick', dict(asset_type='Bowl', source='surface'))
    assert result['error']['kind'] == 'target_type_mismatch'
    assert len(base.calls) == before


def test_generic_dispatch_and_wrong_type(tmp_path, monkeypatch):
    api, base, c = make_interface(tmp_path)
    calls = []
    def execute(name, args):
        calls.append((name, args))
        return dict(success=name == 'navigate', error=None if name == 'navigate' else dict(kind='execution_failed'))
    monkeypatch.setattr(api.objects, 'execute', execute)
    assert api.execute_skill('navigate', dict(asset_type='Plate'))['success']
    assert api.execute_skill('pick', dict(asset_type='Mug', source='surface'))['error']['kind'] == 'target_type_mismatch'
    assert not api.execute_skill('pick', dict(asset_type='Plate', source='surface'))['success']
    assert [n for n, _ in calls] == ['navigate', 'pick']
    assert not base.calls
