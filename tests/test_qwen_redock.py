import pytest
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface, SENSOR_CATALOG, SkillOutcome
from cross_episode_sim.code_policy.skill_interface import QwenSkillInterface


@pytest.mark.parametrize('name',[n for n in SENSOR_CATALOG if n not in ('navigate','explore')])
def test_every_non_navigation_skill_redocks_before_handler(name):
    calls=[]
    def handler(progress,**args):
        calls.append(('skill',args));return SkillOutcome(True)
    def redock(**args):
        calls.append(('dock',args));return {'success':True}
    original=M2T2SkillInterface({name:handler})
    wrapper=QwenSkillInterface(original,redock)
    schema=SENSOR_CATALOG[name][1]
    args={key:(schema['properties'][key].get('enum',['bowl'])[0] if schema['properties'][key]['type']=='string' else 1)
          for key in schema.get('required',[])}
    result=wrapper.execute_skill(name,{**args,'redock':True,'redock_target':'kitchen counter'})
    assert result['success'] and [c[0] for c in calls]==['dock','skill']
    assert calls[-1][1]==args
    assert 'redock' not in next(r for r in original.list_skills() if r['name']==name)['parameters']['properties']


@pytest.mark.parametrize('raises', [False, True])
def test_failed_redock_attempts_manipulation_and_preserves_failure(raises):
    calls=[]
    original=M2T2SkillInterface({'pick':lambda **kw:calls.append(kw) or SkillOutcome(True)})
    def redock(**kw):
        if raises:raise RuntimeError('No local route')
        return {'success':False, 'reason':'No local route'}
    wrapper=QwenSkillInterface(original,redock)
    result=wrapper.execute_skill('pick',{'target':'bowl','redock':True})
    assert result['success'] and len(calls)==1
    assert not result['redock']['success']
    assert result['redock']['reason']=='No local route'


def test_unavailable_and_invalid_requests_do_not_move():
    calls=[]
    wrapper=QwenSkillInterface(M2T2SkillInterface({}),lambda **kw:calls.append(kw))
    assert not wrapper.execute_skill('pick',{'target':'bowl','redock':True})['success']
    with pytest.raises(ValueError):wrapper.execute_skill('navigate',{'target':'bowl','redock':True})
    assert not calls


def test_place_redocks_to_receptacle_not_object():
    calls=[]
    wrapper=QwenSkillInterface(M2T2SkillInterface({'place':lambda **kw:SkillOutcome(True)}),
        lambda **kw:calls.append(kw) or {'success':True})
    wrapper.execute_skill('place',{'receptacle':'table','redock':True})
    assert calls==[{'target':'table','purpose':'place','distance':None}]


def test_existing_navigation_receives_held_object_mode():
    from types import SimpleNamespace
    calls=[]
    operations=SimpleNamespace(dock=lambda **kw:calls.append(kw) or SkillOutcome(True))
    wrapper=QwenSkillInterface.from_navigation(
        M2T2SkillInterface({'place':lambda **kw:SkillOutcome(True)}),operations,
        SimpleNamespace(held_target='bowl'))
    assert wrapper.execute_skill('place',{'receptacle':'table','redock':True})['success']
    assert calls==[{'target':'table','purpose':'place','carrying':True}]


def test_recorded_context_never_uses_future_mapping(tmp_path):
    import json
    import numpy as np
    from types import SimpleNamespace
    from cross_episode_sim.code_policy.recorded_context import restore_start_context
    rows=[]
    for stamp in (1.,2.,3.):
        pose=np.eye(4);pose[0,3]=stamp*.1
        name=f'map_{stamp}.npz'
        np.savez(tmp_path/name,origin=np.array([-1.,-1.]),resolution=1.,floor=np.full((4,4),int(stamp),dtype=np.uint16))
        rows.append(dict(timestamp=stamp,accepted=True,pose=pose.tolist(),map=name))
    (tmp_path/'diagnostics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    grid=SimpleNamespace(floor_views=np.zeros((1,1),dtype=np.uint16),config=SimpleNamespace(resolution=.1),cell_to_world=lambda cells:cells*.1)
    anchor=restore_start_context(grid,tmp_path,2.)
    assert anchor[0,3]==.2 and grid.floor_views[0,0]==2
    assert len(grid.inherited_support_poses)==2


def test_redock_default_preserves_original_dispatch():
    calls=[]
    wrapper=QwenSkillInterface(M2T2SkillInterface({'pick':lambda **kw:SkillOutcome(True)}),lambda **kw:calls.append(kw))
    assert wrapper.execute_skill('pick',{'target':'bowl'})['success']
    assert not calls


def test_prior_support_history_is_time_filtered(tmp_path):
    import json
    import numpy as np
    from types import SimpleNamespace
    from cross_episode_sim.code_policy.recorded_context import restore_start_context
    np.savez(tmp_path/'floor.npz',origin=[-1.,-1.],resolution=1.,floor=np.ones((4,4),np.uint16))
    anchor=np.eye(4);anchor[0,3]=.2
    (tmp_path/'diagnostics.jsonl').write_text(json.dumps(dict(timestamp=2.,accepted=True,pose=anchor.tolist(),map='floor.npz')))
    history=[]
    for stamp in (1.,3.):
        pose=np.eye(4);pose[0,3]=stamp*.1
        history.append(dict(timestamp=stamp,accepted=True,pose=pose.tolist()))
    (tmp_path/'support_history.json').write_text(json.dumps(history))
    grid=SimpleNamespace(floor_views=np.zeros((1,1),np.uint16),config=SimpleNamespace(resolution=.1),cell_to_world=lambda cells:cells*.1)
    restore_start_context(grid,tmp_path,2.)
    assert len(grid.inherited_support_poses)==2
    assert np.allclose(grid.inherited_support_poses[:,0],[0.,-.1])


def test_redock_backend_error_fails_the_dock_not_the_recipe():
    calls=[]
    original=M2T2SkillInterface({'pick':lambda **kw:calls.append(kw) or SkillOutcome(False,'no fresh target')})
    def redock(**kw):raise FileNotFoundError('failed_call_005/report.json')
    result=QwenSkillInterface(original,redock).execute_skill('pick',{'target':'bowl','redock':True})
    assert len(calls)==1 and not result['success']
    assert not result['redock']['success'] and 'FileNotFoundError' in result['redock']['reason']
