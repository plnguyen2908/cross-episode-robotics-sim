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
