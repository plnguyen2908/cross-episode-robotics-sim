import pytest
from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface,SensorNavigation,NavigationState,SkillOutcome


class Operations:
    def __init__(self, fail=None):self.calls=[];self.fail=fail;self.position=0
    def call(self,name,**kwargs):
        self.calls.append((name,kwargs));self.position+=1
        return SkillOutcome(name!=self.fail,'blocked' if name==self.fail else '')
    def check_coverage(self,**kw):return self.call('check_coverage',**kw)
    def transit(self,**kw):return self.call('transit',**kw)
    def approach(self,**kw):return self.call('approach',**kw)
    def dock(self,**kw):return self.call('dock',**kw)


def interface(ops,state=None,**kwargs):
    nav=SensorNavigation(ops,state or NavigationState())
    return M2T2SkillInterface({'navigate':nav.execute},progress=lambda *a:None,**kwargs)


def test_find_route_and_docking_share_one_public_call(tmp_path):
    ops=Operations();api=interface(ops,output=tmp_path)
    r=api.execute_skill('navigate',{'target':'bowl','search_at':'kitchen'})
    assert r['success']
    assert [name for name,_ in ops.calls]==['check_coverage','transit','approach','dock']
    assert ops.calls[1][1]['target']=='kitchen' and ops.calls[-1][1]['target']=='bowl'
    assert len(api.history)==1
    assert (tmp_path/'sensor_skill_calls.json').exists()


def test_carry_uses_destination_and_place_dock():
    ops=Operations();api=interface(ops,NavigationState(held_target='bowl'))
    assert api.execute_skill('navigate',{'target':'bowl','destination':'dining table'})['success']
    assert all(kw['carrying'] for name,kw in ops.calls if name!='check_coverage')
    assert ops.calls[-1][1]==dict(target='dining table',purpose='place',carrying=True)


def test_failed_stage_keeps_state_and_does_not_continue():
    ops=Operations(fail='transit');state=NavigationState(station='old')
    r=interface(ops,state).execute_skill('navigate',{'target':'cup','search_at':'kitchen'})
    assert not r['success'] and ops.position==2 and state.station is None
    assert [name for name,_ in ops.calls]==['check_coverage','transit']


@pytest.mark.parametrize('arguments',[
    {'target':'bowl','x':1.}, {'target':'bowl','mode':'teleport'},
    {'target':'bowl','dock':'3'}, {'target':''}, {'target':'bowl','mode':None},
])
def test_rejects_unknown_or_oracle_arguments(arguments):
    ops=Operations();assert not interface(ops).execute_skill('navigate',arguments)['success'];assert not ops.calls


def test_dock_does_not_repeat_visual_approach_or_transit():
    ops=Operations();r=interface(ops).execute_skill('navigate',{'target':'drawer','mode':'dock','purpose':'handle'})
    assert r['success'] and [n for n,_ in ops.calls]==['check_coverage','dock']


def test_no_silent_oracle_fallback():
    api=interface(Operations());r=api.execute_skill('open_drawer',{'target':'drawer'})
    assert r['reason']=='backend_unavailable'
    assert not next(s for s in api.list_skills() if s['name']=='open_drawer')['available']


def test_no_nested_calls_or_private_backend_results():
    def run(progress,**kw):
        nested=api.execute_skill('pick',{'target':'cup'})
        assert nested['reason']=='skill_already_running'
        raise RuntimeError('private_fixture_id=secret')
    api=M2T2SkillInterface({'pick':run},progress=lambda *a:None)
    r=api.execute_skill('pick',{'target':'cup'})
    assert r['reason']=='backend_error' and 'secret' not in str(r)
    assert len(api.history)==1


def test_held_target_mismatch_rejected_before_motion():
    ops=Operations();api=interface(ops,NavigationState(held_target='cup'))
    assert not api.execute_skill('navigate',{'target':'bowl','destination':'table'})['success']
    assert not ops.calls


def test_missing_coverage_requests_explicit_exploration_without_motion():
    ops=Operations(fail='check_coverage');r=interface(ops).execute_skill('navigate',{'target':'bowl'})
    assert r['reason']=='exploration_required'
    assert [n for n,_ in ops.calls]==['check_coverage']


def test_exploration_is_separately_registered_and_logged():
    from cross_episode_sim.skills.sensor_interface import SensorExploration
    class Explorer:
        def __init__(self):self.targets=[]
        def explore(self,**kw):self.targets.append(kw['target']);return SkillOutcome(True)
    ops=Explorer();skill=SensorExploration(ops,NavigationState())
    api=M2T2SkillInterface({'explore':skill.execute},progress=lambda *a:None)
    assert next(row for row in api.list_skills() if row['name']=='explore')['available']
    assert api.execute_skill('explore',{})['success']
    assert api.execute_skill('explore',{'target':'kitchen'})['success']
    assert ops.targets==[None,'kitchen']
    assert [e['skill'] for e in api.history]==['explore','explore']


def test_internal_coverage_request_survives_sanitization():
    from cross_episode_sim.skills.sensor_interface import ExplorationRequired
    def handler(**kwargs):raise ExplorationRequired('private map details')
    api=M2T2SkillInterface({'navigate':handler},progress=lambda *a:None)
    r=api.execute_skill('navigate',{'target':'cup'})
    assert r['reason']=='exploration_required' and 'private map' not in str(r)


def test_m2t2_name_keeps_sensor_import_compatible_and_oracle_separate():
    from cross_episode_sim.skills.sensor_interface import SensorSkillInterface
    from cross_episode_sim.skills.interface import SkillInterface
    from cross_episode_sim.skills.fixed_interface import FixedSkillInterface
    assert SensorSkillInterface is M2T2SkillInterface
    assert SkillInterface is not M2T2SkillInterface
    assert FixedSkillInterface is not M2T2SkillInterface
