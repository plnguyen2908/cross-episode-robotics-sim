"""Phase transitions, human-event guards, count contract and replay timing."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from cross_episode_sim.tasks.breakfast.gather import (
    DEFAULT_CONFIG, GatherBreakfast, content_state_at, validate_gather_config, vessel_roles)
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode


def config():return json.loads(DEFAULT_CONFIG.read_text())


@pytest.mark.parametrize('people,expected',[(1,2),(2,4)])
def test_exact_requested_vessel_counts(people,expected):
    assert len(vessel_roles(people))==expected
    assert sum(r.startswith('bowl') for r in vessel_roles(people))==people
    assert 'sweet_condiment' not in vessel_roles(people)


def test_no_extra_or_missing_vessels_allowed():
    c=config();validate_gather_config(c)
    c['initial_sources']['extra_cup']='kitchen'
    with pytest.raises(ValueError,match='exactly'):validate_gather_config(c)


def controller(tmp_path):
    c=GatherBreakfast.__new__(GatherBreakfast)
    c.manifest={'bindings':[dict(role='cup_one',body='cup',source='dining',destination='kitchen',
        filling_position=[0.,0.,1.],serving_position=[2.,2.,1.],serving_room='dining',
        vessel_type='cup',content_geoms=['fill'])], 'storage':{},'gathering_config':config()}
    c.object_info={'cup':c.manifest['bindings'][0]}
    c.content_states={'cup_one':'empty'};c.content_events=[];c.phase='GATHER'
    c.attached=False;c.report={'human_events':[]};c.output=tmp_path
    c.data=SimpleNamespace(time=0.,body=lambda name:SimpleNamespace(xpos=np.array([0.,0.,1.])))
    c.events=[];c.record=lambda **kw:c.events.append(kw)
    return c


def test_empty_vessel_at_office_is_retrieved_not_skipped(tmp_path):
    c=controller(tmp_path);c.current_room=lambda info:'dining'
    with patch.object(BreakfastEpisode,'transfer_role') as transfer:
        c.transfer_role('cup_one');transfer.assert_called_once_with(c,'cup_one')


def test_supported_counter_vessel_does_not_get_transferred(tmp_path):
    c=controller(tmp_path);c.current_room=lambda info:'kitchen'
    with patch.object(BreakfastEpisode,'transfer_role') as transfer:
        c.transfer_role('cup_one');transfer.assert_not_called()
    assert c.events[-1]['skipped_transfer']


@pytest.mark.parametrize('contents',['empty','spilled'])
def test_serving_requires_filled_contents(tmp_path,contents):
    c=controller(tmp_path);c.current_room=lambda info:'kitchen';c.phase='SERVE'
    c.content_states['cup_one']=contents
    with pytest.raises(RuntimeError,match='Cannot serve'):c.transfer_role('cup_one')
    assert not c.verify_role('cup_one')


def test_phase_change_restores_original_serving_targets(tmp_path):
    c=controller(tmp_path);c.current_room=lambda info:'kitchen'
    c.configure_phase('GATHER');c.manifest['bindings'][0]['destination_position']=[.5,.5,1.]
    c.report['executed_placement_targets']={'cup_one':[.5,.5,1.]}
    c.configure_phase('SERVE')
    assert c.manifest['bindings'][0]['destination_position']==[2.,2.,1.]
    assert c.report['executed_placement_targets']=={}


def test_human_cannot_fill_missing_or_held_vessels(tmp_path):
    c=controller(tmp_path);c.all_at_counter=lambda:False
    with pytest.raises(RuntimeError,match='all vessels'):c.human_fill()
    assert not c.content_events and c.content_states['cup_one']=='empty'


def test_human_event_occurs_after_wait_and_is_replayable(tmp_path):
    c=controller(tmp_path);c.all_at_counter=lambda:True
    c.tuck_for_navigation=lambda:None;c.base_pose=lambda:np.zeros(3)
    c.task_navigate=lambda *a,**kw:None;c.in_default_travel_posture=lambda loaded:True
    c.contents_visible=lambda states:None
    c.tick=lambda seconds:setattr(c.data,'time',c.data.time+seconds)
    c.human_fill()
    assert c.filled()
    event=c.content_events[0]
    assert event['time']==config()['human_wait_seconds']
    assert event['actor']=='simulated_human'
    assert content_state_at(c.content_events,event['time']-.01,['cup_one'])=={'cup_one':'empty'}
    assert content_state_at(c.content_events,event['time'],['cup_one'])=={'cup_one':'filled'}
    assert (tmp_path/'human_events.json').is_file()


def test_replay_preserves_spill_after_fill():
    events=[dict(time=5.,states={'cup':'filled'}),dict(time=9.,states={'cup':'spilled'})]
    assert content_state_at(events,0.,['cup'])=={'cup':'empty'}
    assert content_state_at(events,7.,['cup'])=={'cup':'filled'}
    assert content_state_at(events,10.,['cup'])=={'cup':'spilled'}


def test_full_plan_has_fill_barrier_between_gather_and_serve(tmp_path):
    c=controller(tmp_path)
    with patch('cross_episode_sim.tasks.breakfast.gather.CompositeEpisode') as runner:
        runner.return_value.run.return_value=True
        assert c.run_test()==0
        steps=runner.return_value.run.call_args.args[1]
    assert [s['operation'] for s in steps]==['validate_population','phase','transfer','human_fill','phase','transfer']
    assert steps[1]['arguments']=={'phase':'GATHER'} and steps[4]['arguments']=={'phase':'SERVE'}


@pytest.mark.parametrize('kind',['cabinet','drawer'])
@pytest.mark.parametrize('fail',[False,True])
def test_storage_adapter_dispatches_live_actions_and_restores_context(tmp_path,kind,fail):
    from cross_episode_sim.tasks.breakfast.storage import retrieve_from_storage
    from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
    from cross_episode_sim.fixtures.drawer_loop import DrawerLoop
    c=controller(tmp_path);c.model=MagicMock();c.model.joint.return_value.id=0
    c.model.jnt_qposadr=np.array([0]);c.model.geom.return_value.id=0
    c.descendants=lambda name:{0};c.args=SimpleNamespace(motion_slowdown=1.,grip_force=10.)
    c.successful_grasp_annotations={'Bowl_2':{1}}
    c.initial_lift_height=.02;c.active_family='top';c.args.annotation_source='qualified_registry'
    facade=object();c.robot_actions=facade
    info=c.manifest['bindings'][0];info['source']=kind;events=[]
    def transfer(owner,role):
        assert owner.__dict__ is c.__dict__
        assert owner.robot_actions is None
        assert owner.dining==c.task_supports['kitchen']
        events.append(('transfer',role))
        owner.successful_grasp_annotations['Bowl_2']={9}
        owner.initial_lift_height=.025;owner.active_family='any'
        owner.args.annotation_source='rotated_rim_hypotheses'
        if fail:raise RuntimeError('test failed grasp')
    cls,method=(CabinetTransfer,'door_cycle') if kind=='cabinet' else (DrawerLoop,'drawer_cycle')
    with patch.object(cls,method,lambda self,opening:events.append(('access',opening))), \
            patch.object(BreakfastEpisode,'transfer_role',transfer), \
            patch.object(BreakfastEpisode,'verify_role',return_value=True):
        if fail:
            with pytest.raises(RuntimeError,match='test failed grasp'):retrieve_from_storage(c,info)
        else:retrieve_from_storage(c,info)
    assert c.robot_actions is facade and not hasattr(c,'door_joint')
    assert c.initial_lift_height==.02 and c.active_family=='top'
    assert c.args.annotation_source=='qualified_registry'
    assert c.successful_grasp_annotations['Bowl_2']==({1} if kind=='cabinet' else {9})
    assert not hasattr(c,'dining')
    assert events==[('access',True),('transfer','cup_one')]+([] if fail else [('access',False)])


def test_drawer_reuses_achieved_opening_stance_before_redocking():
    from cross_episode_sim.tasks.breakfast.storage import approach_drawer_pickup
    start=np.array([.5,-1.5,np.pi/2]);pose=np.eye(4);pose[:3,3]=[.35,-.854,.794]
    c=SimpleNamespace(base_pose=lambda:start,bread_pose=lambda:pose,record=MagicMock(),
                      plan_route=MagicMock(return_value=[start]),task_navigate=MagicMock())
    approach_drawer_pickup(c)
    np.testing.assert_array_equal(c.plan_route.call_args.args[0],start[:2])
    assert c.record.call_args.kwargs['reused_opening_stance']
    assert c.task_navigate.call_args.args[1] is False


def test_drawer_rejected_dock_tries_alternatives_without_executing_failure():
    from cross_episode_sim.tasks.breakfast.storage import approach_drawer_pickup
    start=np.array([.5,-1.5,np.pi/2]);alternate=np.array([.35,-1.57,np.pi/2])
    c=SimpleNamespace(base_pose=lambda:start,record=MagicMock(),task_navigate=MagicMock(),
        plan_route=MagicMock(side_effect=[RuntimeError('turn touches wall'),[start,alternate]]))
    with patch('cross_episode_sim.tasks.breakfast.storage.drawer_pickup_stances',return_value=iter([start,alternate])):
        approach_drawer_pickup(c)
    c.task_navigate.assert_called_once()
    np.testing.assert_array_equal(c.task_navigate.call_args.args[0],alternate[:2])
    assert len(c.record.call_args.kwargs['rejected_drawer_docks'])==1


def test_drawer_alternate_docks_follow_open_drawer_geometry():
    from cross_episode_sim.tasks.breakfast.storage import drawer_pickup_stances
    pose=np.eye(4);pose[:3,3]=[.35,-.85,.79]
    c=SimpleNamespace(base_pose=lambda:np.array([.5,-1.5,np.pi/2]),bread_pose=lambda:pose,model=None,data=None)
    with patch('cross_episode_sim.tasks.breakfast.storage.support_bounds',return_value=(np.array([.28,-.92,.71]),np.array([.72,-.38,.74]))):
        docks=list(drawer_pickup_stances(c))
    assert len(docks)==6
    assert np.isclose(docks[1][1],-.92-.55)
    assert all(not np.isclose(d[1],-1.35) for d in docks)


def departure_controller():
    def contact(a,b,depth):
        return SimpleNamespace(geom1=a,geom2=b,dist=-depth)
    pose=np.eye(4);pose[2,3]=.8
    c=SimpleNamespace(holding_loaf=True,tcp=lambda:pose.copy(),bread_pose=lambda:pose.copy(),
        bread_bids={1},descendants=lambda name:{2},object_name='cup',
        model=SimpleNamespace(geom_bodyid=np.array([1,2,3,4,2])),
        data=SimpleNamespace(contact=[contact(0,1,.000064),contact(0,4,.0002),
                                      contact(0,2,.00003),contact(3,1,.00003)]))
    target=pose.copy();target[2,3]+=.17
    return c,target,contact


def test_drawer_departure_only_budgets_shallow_existing_payload_fixture_contacts():
    from cross_episode_sim.tasks.breakfast.storage import drawer_departure_budget
    c,target,_=departure_controller()
    assert drawer_departure_budget(c,'lift bread',target)=={
        'depths':{(0,1):.000064},'start_z':.8}


@pytest.mark.parametrize('change',['stage','unloaded','down','short','sideways','rotate'])
def test_drawer_departure_requires_loaded_vertical_lift(change):
    from cross_episode_sim.tasks.breakfast.storage import drawer_departure_budget
    c,target,_=departure_controller();stage='lift bread'
    if change=='stage':stage='lower object'
    if change=='unloaded':c.holding_loaf=False
    if change=='down':target[2,3]=.7
    if change=='short':target[2,3]=.81
    if change=='sideways':target[0,3]=.01
    if change=='rotate':target[:3,:3]=np.diag([-1.,-1.,1.])
    assert drawer_departure_budget(c,stage,target) is None


def test_departure_separates_but_rejects_new_and_worsening_contacts():
    from cross_episode_sim.tasks.breakfast.storage import (
        drawer_departure_budget,drawer_departure_contact_view)
    c,target,contact=departure_controller()
    c._drawer_departure_budget=drawer_departure_budget(c,'lift bread',target)
    existing=contact(1,0,.000025);new=contact(0,2,.000015)
    data=SimpleNamespace(contact=[existing,new],time=9.,
        body=lambda name:SimpleNamespace(xpos=np.array([0.,0.,.805])))
    view=drawer_departure_contact_view(c,data,True)
    assert view.contact==[new] and view.time==9.
    assert data.contact==[existing,new]  # Scratch and physical contacts are unchanged.
    existing.dist=-.00004  # Below initial depth, but increasing since last sample.
    assert drawer_departure_contact_view(c,data,True).contact==[existing,new]
    data.contact=[]
    assert drawer_departure_contact_view(c,data,True).contact==[]
    data.contact=[existing]  # Re-entry after separating is also rejected.
    assert drawer_departure_contact_view(c,data,True).contact==[existing]


@pytest.mark.parametrize('loaded,dz,active',[(False,.005,True),(True,-.005,True),
    (True,.021,True),(True,.005,False)])
def test_departure_exception_does_not_apply_outside_scoped_lift(loaded,dz,active):
    from cross_episode_sim.tasks.breakfast.storage import (
        drawer_departure_budget,drawer_departure_contact_view)
    c,target,contact=departure_controller()
    c._drawer_departure_budget=drawer_departure_budget(c,'lift bread',target) if active else None
    data=SimpleNamespace(contact=[contact(0,1,.000025)],
        body=lambda name:SimpleNamespace(xpos=np.array([0.,0.,.8+dz])))
    assert drawer_departure_contact_view(c,data,loaded) is data


@pytest.mark.parametrize('pitch',[25.,30.,40.])
def test_front_shelf_allows_diagonal_entry_without_changing_legacy_policy(pitch):
    from cross_episode_sim.fixtures.cabinet_transfer import front_shelf_approach
    a=np.radians(pitch)
    z=np.array([0.,np.cos(a),-np.sin(a)]);y=np.array([1.,0.,0.])
    pose=np.column_stack([np.cross(y,z),y,z])
    assert front_shelf_approach(pose,diagonal=True)
    assert not front_shelf_approach(pose,diagonal=False)


@pytest.mark.parametrize('approach',[[0.,0.,-1.],[0.,-1.,0.],[1.,0.,0.],[0.,.7,.7]])
def test_shelf_rejects_entry_through_top_back_side_or_from_below(approach):
    from cross_episode_sim.fixtures.cabinet_transfer import front_shelf_approach
    pose=np.eye(3);pose[:,2]=np.array(approach)/np.linalg.norm(approach)
    assert not front_shelf_approach(pose,diagonal=True)


def test_cabinet_pickup_dock_tracks_object_and_checks_route(tmp_path):
    from cross_episode_sim.tasks.breakfast.storage import retrieve_from_storage
    from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer
    c=controller(tmp_path);c.model=MagicMock();c.model.joint.return_value.id=0
    c.model.jnt_qposadr=np.array([0]);c.model.geom.return_value.id=0
    c.descendants=lambda name:{0};c.args=SimpleNamespace(motion_slowdown=1.,grip_force=10.)
    c.robot_actions=None;info=c.manifest['bindings'][0];info['source']='cabinet'
    pose=np.eye(4);pose[:3,3]=[2.58,-.44,.69]
    c.bread_pose=lambda:pose.copy();c.base_pose=lambda:np.array([2.1,-1.2,np.pi/2])
    c.plan_route=MagicMock(side_effect=[RuntimeError('blocked door'),[np.zeros(3)]])
    c.task_navigate=MagicMock()
    def cycle(proxy,opening):
        if opening:proxy.cabinet_dock(False)
    with patch.object(CabinetTransfer,'door_cycle',cycle), \
            patch.object(BreakfastEpisode,'transfer_role'), \
            patch.object(BreakfastEpisode,'verify_role',return_value=True), \
            patch('cross_episode_sim.tasks.breakfast.storage.support_bounds',
                return_value=(np.array([2.43,-.56,.60]),np.array([2.72,-.03,.62]))):
        retrieve_from_storage(c,info)
    assert c.plan_route.call_count==2
    c.task_navigate.assert_called_once()
    np.testing.assert_allclose(c.task_navigate.call_args.args[0],[2.30,-1.11])


def test_shelf_exit_clears_front_before_full_lift():
    from cross_episode_sim.tasks.breakfast.storage import shelf_exit_waypoints
    start=np.eye(4);start[:3,3]=[2.58,-.50,.74]
    first,clear,extracted,raised=shelf_exit_waypoints(start)
    np.testing.assert_allclose(first[:3,3]-start[:3,3],[0.,-.045,.003])
    np.testing.assert_allclose(clear[:3,3]-start[:3,3],[0.,-.09,.025])
    assert extracted[1,3]<=-.82 and extracted[2,3]==clear[2,3]
    assert raised[1,3]==extracted[1,3] and np.isclose(raised[2,3]-start[2,3],.14)
    for pose in (first,clear,extracted,raised):
        np.testing.assert_array_equal(pose[:3,:3],start[:3,:3])
    np.testing.assert_array_equal(start[:3,3],[2.58,-.50,.74])


@pytest.mark.parametrize('distance,other_room,expected',[
    (.5,False,'local'),(1.5,False,'navigate'),(.5,True,'navigate')])
def test_cabinet_counter_uses_local_dock_only_when_near_and_same_room(
        tmp_path,distance,other_room,expected):
    from cross_episode_sim.tasks.breakfast.storage import retrieve_from_storage
    from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer,SHELF
    c=controller(tmp_path);c.model=MagicMock();c.model.joint.return_value.id=0
    c.model.jnt_qposadr=np.array([0]);c.model.geom.return_value.id=0
    c.descendants=lambda name:{0};c.args=SimpleNamespace(motion_slowdown=1.,grip_force=10.)
    c.profile=SimpleNamespace(manipulation_offsets=((.6,0.),))
    c.base_pose=lambda:np.zeros(3)
    c.room_id=lambda xy: 6 if other_room and xy[0]>0 else 4
    c.object_name='cup';info=c.manifest['bindings'][0]
    info.update(source='cabinet',destination_position=[distance,0.,1.])
    events=[];c.prepare_destination=lambda:events.append('local')
    def transfer(proxy,role):
        proxy.source=SHELF
        proxy.transport_payload()
    with patch.object(CabinetTransfer,'door_cycle',lambda self,opening:None), \
            patch.object(BreakfastEpisode,'transfer_role',transfer), \
            patch.object(BreakfastEpisode,'transport_payload',lambda self:events.append('navigate')), \
            patch.object(BreakfastEpisode,'verify_role',return_value=True):
        retrieve_from_storage(c,info)
    assert events==[expected]
