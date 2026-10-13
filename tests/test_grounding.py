from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cross_episode_sim.evaluation.grounding import TargetRegistry


def registry():
    m = SimpleNamespace(geom_bodyid=[1, 2, 3], body_parentid=[0, 0, 1, 0],
                        body=lambda name: SimpleNamespace(id={'cup_body': 1, 'bowl_body': 3}[name]))
    return TargetRegistry(m, [dict(role='cup_01', body='cup_body'), dict(role='bowl_02', body='bowl_body')])


def observation(r, geom):
    image = np.zeros((40, 50, 3), np.uint8)
    seg = np.full((40, 50, 2), -1, dtype=int)
    if geom is not None:
        seg[15:25, 20:30, 0] = geom
        seg[15:25, 20:30, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    return r.update({'wrist': image}, {'wrist': seg})


def test_visible_ids_colors_masks_and_occlusion():
    r = registry()
    a = observation(r, 1)  # Child geom maps to the same target body.
    assert len(a['targets']) == 1
    first = a['targets'][0]
    assert first['category'] == 'cup' and first['visible']
    assert first['views']['wrist']['bbox_xyxy'] == [20, 15, 30, 25]
    assert a['target_masks']['wrist:obj_1'].sum() == 100
    assert a['images']['wrist'].any() and not a['raw_images']['wrist'].any()
    b = observation(r, 2)
    assert b['targets'][0]['visible'] is False and b['targets'][0]['views'] == {}
    assert b['targets'][1]['color_rgb'] != first['color_rgb']
    c = observation(r, 0)
    assert c['targets'][0]['target_id'] == first['target_id']
    assert c['targets'][0]['color_rgb'] == first['color_rgb']
    assert all('body' not in target and 'role' not in target for target in c['targets'])


def test_catalog_and_private_oracle_mapping():
    r = registry()
    catalog = [dict(name='pick', parameters=dict(properties={'role': dict(type='string', enum=['cup_01'])}, required=['role']))]
    public = r.catalog(catalog)
    assert 'role' not in public[0]['parameters']['properties']
    assert 'enum' not in public[0]['parameters']['properties']['target_id']
    assert public[0]['parameters']['required'] == ['target_id']
    seen = []
    interface = SimpleNamespace(validate=lambda name, args: seen.append((name, args)))
    with pytest.raises(ValueError, match='never been observed'):
        r.resolve('pick', {'target_id': 'obj_1'}, interface)
    observation(r, 0)
    assert r.resolve('pick', {'target_id': 'obj_1'}, interface) == {'role': 'cup_01'}
    assert seen[-1] == ('pick', {'role': 'cup_01'})
    with pytest.raises(ValueError, match='private role'):
        r.resolve('pick', {'role': 'cup_01'}, interface)


def test_vla_receives_grounding_and_measured_verification(tmp_path):
    from cross_episode_sim.evaluation import Budgets, evaluate
    from tests.test_evaluation import Session

    class GroundedSession(Session):
        def observe(self):
            result = super().observe()
            result.update(targets=[dict(target_id='obj_1', color_rgb=[255, 51, 51], visible=True)],
                          target_masks={'wrist:obj_1': np.ones((2, 2), bool)},
                          raw_images=result['images'], perception='oracle_instance_segmentation')
            return result

        def verify_skill(self, name, arguments):
            self.verified = (name, arguments)
            return False

    class Planner:
        def reset(self, context):
            pass

        def plan(self, observation, feedback):
            assert observation['targets'][0]['target_id'] == 'obj_1'
            assert 'state' not in observation and 'target_masks' not in observation
            if feedback is not None:
                assert feedback['status'] == 'failed'
                assert feedback['verification'] == 'measured_postcondition'
                return {'finish': True}
            return dict(skill='pick', arguments={'role': 'cup'})

    class VLA:
        def start_skill(self, *args):
            pass

        def act(self, observation):
            assert observation['target_masks']['wrist:obj_1'].all()
            assert 'raw_images' in observation and 'state' in observation
            return {'done': True}

    session = GroundedSession()
    result = evaluate(Planner(), session, tmp_path/'eval', backend='vla', vla=VLA(),
                      budgets=Budgets(tool_calls=10), context={'instruction': 'pick'})
    assert session.verified == ('pick', {'role': 'cup'})
    assert not session.oracle_calls and not result['success']
