from types import SimpleNamespace
import pytest
from cross_episode_sim.code_policy.household_session import HouseholdSession


def session(tmp_path, held=None):
    return HouseholdSession(SimpleNamespace(), None, tmp_path, 'bowl', held=held)


def test_real_household_handlers_and_schema_feedback(tmp_path):
    s=session(tmp_path)
    rows=s.list_skills()
    registered={v['name'] for v in rows if v['available']}
    assert {'place','open_door','open_drawer','rotate_knob','pick','navigate'} <= registered
    assert s.calls[-1]['tool']=='list_skills'
    assert {v['name'] for v in s.calls[-1]['result']['available']} == registered


def test_loaded_navigation_requires_destination_and_preserves_category(tmp_path):
    s=session(tmp_path, 'cup');calls=[]
    s.move=lambda *args:calls.append(args) or {'success':True}
    with pytest.raises(ValueError):s.navigate('cup')
    with pytest.raises(ValueError):s.navigate('bowl',destination='kitchen drawer')
    s.navigate('cup',destination='kitchen drawer')
    assert calls==[('kitchen drawer','approach',None,'navigate')]


def test_hand_is_not_released_to_open_a_fixture(tmp_path):
    s=session(tmp_path,'bowl')
    assert not s.fixture('open_door',target='fridge').success
    assert not s.appliance('rotate_knob',target='front left stove burner knob',state='on').success
    assert s.held=='bowl'


def test_failed_placement_consumes_plan_but_not_held_state(tmp_path):
    s=session(tmp_path,'bowl')
    s.prepared=dict(root=tmp_path,dock=tmp_path,script='backend',extra=[],
                    receptacle='dining table',region='default',description='dining table',epoch=0)
    def fail(*args):raise RuntimeError('release failed')
    s.stage=fail
    with pytest.raises(RuntimeError):s.place('dining table')
    assert s.prepared is None and s.held=='bowl'


def test_loaded_redock_never_uses_empty_arm_motion(tmp_path):
    s=session(tmp_path,'cup');calls=[]
    s.invalidate=lambda:None
    s.prepare_place=lambda *args:calls.append(args)
    assert s.redock('kitchen drawer',purpose='place')['success']
    assert calls==[('kitchen drawer','interior')]
    assert s._redocking is None


def test_qwen_can_choose_cabinet_dock_without_prior_failed_place(tmp_path):
    s=session(tmp_path,'bowl');calls=[]
    s.invalidate=lambda:None
    s.prepare_place=lambda *args,**kwargs:calls.append((args,kwargs))
    assert s.redock('kitchen storage cabinet',purpose='place',distance=.45,lateral_offset=-.12)['success']
    assert calls==[(('kitchen storage cabinet','interior'),dict(cabinet_distance=.45,cabinet_offset=-.12))]
    assert s.held=='bowl' and s._redocking is None


def test_invalid_loaded_dock_options_never_move(tmp_path):
    s=session(tmp_path,'bowl')
    s.invalidate=lambda:pytest.fail('Invalid request must not move or invalidate state')
    with pytest.raises(ValueError):s.redock('kitchen storage cabinet',purpose='place',distance=float('nan'))
    with pytest.raises(ValueError):s.redock('kitchen storage cabinet',purpose='place',lateral_offset=1.)
    with pytest.raises(ValueError):s.redock('dining table',purpose='place',lateral_offset=.1)


def test_place_without_redock_requests_current_pose_only(tmp_path):
    s=session(tmp_path,'bowl');calls=[]
    def prepare(target,region,**kw):
        calls.append(kw)
        s.prepared=dict(root=tmp_path,dock=tmp_path,script='place',description=target,extra=[])
    s.prepare_place=prepare
    s.stage=lambda *args:{'execution_success':True}
    s.invalidate=lambda:None
    assert s.place('dining table').success
    assert calls==[{'dock':False}]
    assert s.held is None


def test_failed_public_call_is_counted_and_delivered(tmp_path):
    s=session(tmp_path)
    result=s.functions()['navigate']('bowl',destination='table')
    assert not result['success'] and len(s.calls)==1
    assert 'Pick before' in result['reason']


def test_prompt_does_not_repeat_large_collision_payloads():
    from scripts.test_qwen_household import prompt_history
    history=[dict(answer='redock()',feedback={},tool_results=[dict(tool='redock',arguments={},
        result=dict(success=False,reason='blocked',planning_events=['large']*10000))])]*20
    compact=prompt_history(history)
    assert len(compact)==4
    assert compact[-1]['tools'][0]['result']=={'success':False,'reason':'blocked'}


def test_direct_atomic_aliases_use_same_dispatcher(tmp_path):
    s=session(tmp_path);calls=[]
    s.execute_skill=lambda name,args:calls.append((name,args)) or {'success':True}
    tools=s.functions()
    tools['place']({'receptacle':'dining table','redock':True})
    tools['open_door'](target='fridge',redock=True)
    assert calls==[('place',{'receptacle':'dining table','redock':True}),
                   ('open_door',{'target':'fridge','redock':True})]


def test_continuation_accepts_qwen_dispatcher_spelling(tmp_path):
    s=session(tmp_path);calls=[]
    s.continue_door=lambda **kwargs:calls.append(kwargs) or {'success':True}
    result=s.execute_skill('continue_door',{'target':'fridge','motion':'base_push'})
    assert result['success']
    assert calls==[{'target':'fridge','motion':'base_push'}]


def test_private_support_accepts_native_stack_shelf_not_ground():
    import mujoco
    from cross_episode_sim.evaluation.qwen_component_score import ComponentScore
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="stack_4_main_group_1" pos="0 0 .4">
        <geom name="stack_4_main_group_1_floor" type="box" size=".3 .3 .02"/>
        <body pos=".3 .3 .2"><joint name="stack_4_main_group_1_doorhinge" type="hinge" range="0 90"/>
          <geom type="box" size=".01 .2 .2"/></body>
      </body>
      <body name="bowl" pos="0 0 .45"><freejoint/><geom type="sphere" size=".03" mass=".1"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    for _ in range(200):mujoco.mj_step(model,data)
    evaluator=ComponentScore(model,data);evaluator.initially_held={model.body('bowl').id}
    assert evaluator.placement(data,'bowl_in_cabinet')['success']
    data.qpos[model.jnt_qposadr[model.body('bowl').jntadr[0]]]=1.
    for _ in range(400):mujoco.mj_step(model,data)
    assert not evaluator.placement(data,'bowl_in_cabinet')['success']


def test_backend_feedback_preserves_recovery_reason_without_private_fields():
    report={'error':'No freshly observed support surface for inside drawer',
            'private_object_pose':[99,98,97], 'success':False}
    reason=HouseholdSession.backend_reason(report)
    assert 'support_not_visible' in reason and 'redock' in reason
    assert '99' not in reason and 'private' not in reason
    assert 'placement_plan_unreachable' in HouseholdSession.backend_reason({'planning_rejections':[{}]})


@pytest.mark.parametrize('fails',[False,True])
def test_alignment_keeps_payload_and_consumes_plan(tmp_path,fails):
    s=session(tmp_path,'bowl');seen=[]
    def prepare(target,region,**kwargs):
        assert kwargs=={'dock':False}
        assert s.approach_options.approach_mode=='front_entry'
        s.prepared=dict(root=tmp_path,dock=tmp_path,script='backend',description=target,extra=[])
    def stage(script,args,root,key):
        seen.append((args,key))
        if fails:raise RuntimeError('No executable alignment')
        return {'alignment_completed':True}
    s.prepare_place=prepare;s.stage=stage
    if fails:
        with pytest.raises(RuntimeError):s.align('kitchen storage cabinet')
    else:
        result=s.align('kitchen storage cabinet')
        assert result['success'] and result['release_executed'] is False
    assert s.held=='bowl' and s.prepared is None and s.approach_options is None
    assert '--align-only' in seen[0][0] and seen[0][1]=='alignment_completed'


def test_default_place_reuses_prepared_redock_target(tmp_path):
    s=session(tmp_path,'bowl')
    prepared=dict(approach_options=None,dock='sensor-selected-dock')
    s.prepared=prepared
    with s.manipulation_context('place',{},{}):
        assert s.prepared is prepared
    assert s.prepared is prepared
    # An actual approach change must still discard the cached plan.
    with s.manipulation_context('place',{},dict(approach_mode='front_entry')):
        assert s.prepared is None


def test_confined_pick_reobserves_after_alignment_and_restores_mode(tmp_path):
    s=session(tmp_path);events=[]
    def propose(target):
        events.append('observe');return {'candidates':[{'id':0}]}
    def execute(ids):
        align=getattr(s.runtime,'pick_align_only',False)
        events.append('align' if align else 'grasp')
        return {'alignment_completed':align,'execution_success':not align}
    s.propose=propose;s.execute=execute
    result=s.execute_skill('pick',dict(target='bottle',approach_mode='front_entry'))
    assert result['success']
    assert events==['observe','align','observe','grasp']
    assert not s.runtime.pick_align_only


def test_failed_alignment_does_not_attempt_grasp_or_leave_mode_enabled(tmp_path):
    s=session(tmp_path);events=[]
    s.propose=lambda target:{'candidates':[{'id':0}]}
    def execute(ids):events.append('align');return {'alignment_completed':False}
    s.execute=execute
    result=s.execute_skill('pick',dict(target='bottle',approach_mode='front_entry'))
    assert not result['success'] and result['reason']=='front_alignment_failed'
    assert events==['align'] and not s.runtime.pick_align_only


def test_top_down_pick_has_no_confined_alignment_cycle(tmp_path):
    s=session(tmp_path);events=[]
    s.propose=lambda target:{'candidates':[{'id':0}]}
    s.execute=lambda ids:events.append('grasp') or {'execution_success':True}
    assert s.execute_skill('pick',dict(target='bowl',approach_mode='top_down'))['success']
    assert events==['grasp']


def test_basket_place_routes_to_container_backend_without_changing_table(tmp_path,monkeypatch):
    import cross_episode_sim.control.held_view_posture as view
    from cross_episode_sim.control.manipulation_approach import ApproachOptions
    s=session(tmp_path,'drink bottle');motions=[]
    s.checkpoint=lambda root:root;s.aperture=lambda *a,**k:None;s.held_points=lambda:object()
    monkeypatch.setattr(view,'raise_held_view',lambda *args:motions.append(args[2]))
    s.approach_options=ApproachOptions(approach_mode='top_down')
    s.prepare_place('purple basket','interior',dock=False)
    assert s.prepared['script']=='scripts/test_basket_place.py'
    assert s.prepared['extra']==['--current-pose'] and len(motions)==1
    s.prepare_place('dining table','default',dock=False)
    assert s.prepared['script']=='scripts/test_m2t2_place.py'
    assert len(motions)==1


def test_stove_place_keeps_native_candidates_and_explicit_redock(tmp_path,monkeypatch):
    import cross_episode_sim.control.held_view_posture as view
    monkeypatch.setattr(view,'raise_held_view',lambda *args:None)
    s=session(tmp_path,'cookware')
    s.held_points=lambda:object()
    s.checkpoint=lambda root:root
    s.aperture=lambda *a,**k:None
    s.prepare_place('front left stove burner','default',dock=False)
    assert s.prepared['script']=='scripts/test_burner_place.py'
    assert s.prepared['extra']==['--current-pose']
    assert s.prepared['receptacle']=='front left stove burner'


def test_loaded_approach_reuses_opening_memory(tmp_path,monkeypatch):
    import cross_episode_sim.navigation.aperture_dock as aperture_dock
    import cross_episode_sim.navigation.receptacle_approach as approach
    s=session(tmp_path,'food container');seen=[]
    measured={'closed_handle':[1.,0.,1.], 'hinge':[1.,.5,1.], 'closed_normal':[-1.,0.,0.]}
    s.aperture=lambda target,**kwargs:measured
    s.held_points=lambda:object()
    monkeypatch.setattr(aperture_dock,'dock_to_aperture',lambda runtime,opening,*a,**k:seen.append(opening) or {'success':True})
    monkeypatch.setattr(approach,'approach_receptacle',lambda *a,**k:pytest.fail('Do not rediscover an already observed opening'))
    assert s.navigate('food container',destination='microwave')['success']
    assert seen==[measured] and s.held=='food container'
    assert s.infer_region('microwave')=='interior'


def test_explicit_learned_wrist_is_passed_to_pick_not_place(tmp_path):
    s=session(tmp_path);seen=[]
    s.propose=lambda target:{'candidates':[{'id':0}]}
    def execute(ids):
        seen.append((s.approach_options.front_wrist,getattr(s.runtime,'pick_align_only',False)))
        return {'alignment_completed':True,'execution_success':True}
    s.execute=execute
    result=s.execute_skill('pick',dict(target='bowl',approach_mode='front_entry',front_wrist='learned'))
    assert result['success'] and seen==[('learned',False)]
    assert s.approach_options is None
    with pytest.raises(ValueError,match='applies to pick'):
        s.execute_skill('place',dict(receptacle='microwave',front_wrist='learned'))


@pytest.mark.parametrize('offset',[0.,-.12,.1])
def test_empty_hand_lateral_dock_forwarded_without_changing_loaded_defaults(tmp_path,monkeypatch,offset):
    from cross_episode_sim.code_policy.pick_session import PickSession
    s=session(tmp_path);calls=[]
    monkeypatch.setattr(PickSession,'move',lambda self,*args,**kw:calls.append((args,kw)) or {'success':True})
    assert s.redock('food container inside microwave',distance=.65,lateral_offset=offset)['success']
    assert calls==[(('food container inside microwave','pick',.65,'redock'),{'lateral_offset':offset})]


def test_loaded_microwave_dock_accepts_observed_opening_offsets(tmp_path):
    s=session(tmp_path,'food container');calls=[]
    s.invalidate=lambda:None
    s.prepare_place=lambda *args,**kwargs:calls.append((args,kwargs))
    assert s.redock('microwave',purpose='place',distance=.6,lateral_offset=0.)['success']
    assert calls==[(('microwave','interior'),dict(cabinet_distance=.6,cabinet_offset=0.))]
    assert s.held=='food container'


def test_standalone_microwave_redock_uses_remembered_aperture(tmp_path,monkeypatch):
    import cross_episode_sim.runtime.household_placement as backend
    from contextlib import nullcontext
    from cross_episode_sim.control.manipulation_approach import ApproachOptions
    from cross_episode_sim.control import held_view_posture
    from cross_episode_sim.runtime import artifacts
    s=session(tmp_path,'food container');calls=[]
    s.aperture=lambda *args,**kwargs:{'hinge':[1,0,1],'closed_handle':[1,1,1],'closed_normal':[-1,0,0]}
    s.checkpoint=lambda path:path
    s.held_points=lambda:[]
    s.stage=lambda *args:(_ for _ in ()).throw(AssertionError('Must not call generic table docking'))
    monkeypatch.setattr(backend,'prepare_support_view',lambda *args,**kw:calls.append((args[3],kw)) or [])
    monkeypatch.setattr(held_view_posture,'raise_held_view',lambda *args,**kw:None)
    monkeypatch.setattr(artifacts,'write_text',lambda *args:None)
    s.prepare_place('microwave','interior',cabinet_distance=.6,cabinet_offset=0.)
    assert calls[0][0]=='cabinet'
    assert calls[0][1]['cabinet_distances']==(.6,)
    assert s.prepared['script']=='scripts/test_aperture_place.py'
