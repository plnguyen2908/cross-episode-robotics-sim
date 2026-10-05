"""Surface discovery and gather-to-serve support bookkeeping."""
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from cross_episode_sim.manipulation.kitchen_surfaces import (
    discover_filling_surfaces, stove_is_off, filling_candidates, allocate_filling_targets)
from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode


def scene():
    xml='''<mujoco><worldbody>
    <body name="floor_room_main" pos="0 0 -.02"><geom type="box" size="4 3 .02"/></body>
    <body name="counter_custom_main" pos="-2 0 .45"><geom type="box" size=".7 .4 .45"/></body>
    <body name="island_new_main" pos="0 0 .45" euler="0 0 90"><geom type="box" size=".7 .4 .45"/></body>
    <body name="table_kitchen_main" pos="2 0 .4"><geom type="box" size=".7 .4 .4"/></body>
    <body name="table_other_room" pos="0 7 .4"><geom type="box" size=".7 .4 .4"/></body>
    <body name="chair_main" pos="1 1 .4"><geom type="box" size=".3 .3 .4"/></body>
    <body name="microwave_main" pos="2 1 .45"><geom type="box" size=".3 .3 .45"/></body>
    <body name="counter_movable" pos="-1 1 .45"><freejoint/><geom type="box" size=".3 .3 .45"/></body>
    <body name="stove_different_main" pos="3 1 .45"><geom type="box" size=".3 .3 .45"/>
      <geom type="box" size=".3 .3 .015" pos="0 0 .465"/>
      <body name="stove_different_knob_front_left"><joint name="knob" type="hinge"/>
        <geom type="sphere" size=".02"/></body>
    </body>
    </worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    return m,d


def test_discovers_new_layout_surfaces_but_not_other_rooms_or_appliance_tops():
    m,d=scene();sites=discover_filling_surfaces(m,d,'counter_custom_main')
    assert {s['body'] for s in sites.values()}=={'counter_custom_main','island_new_main',
        'table_kitchen_main','stove_different_main'}
    assert sites['kitchen']['body']=='counter_custom_main'
    np.testing.assert_allclose(sites['kitchen_surface_island_new_main']['outward'],[1.,0.])


def test_stove_eligibility_uses_live_knob_state_and_rejects_unknown_state():
    m,d=scene()
    assert stove_is_off(m,d,'stove_different_main')
    d.joint('knob').qpos[0]=1.
    assert not stove_is_off(m,d,'stove_different_main')
    assert not stove_is_off(m,d,'counter_custom_main')


def test_frozen_stove_knob_angle_is_checked():
    m,d=scene();bid=m.body('stove_different_knob_front_left').id
    m.body_jntnum[bid]=0
    m.body_quat[bid]=[np.cos(.5),0.,0.,np.sin(.5)]
    assert not stove_is_off(m,d,'stove_different_main')


def gather():
    c=GatherBreakfast.__new__(GatherBreakfast)
    info=dict(role='cup',body='cup_body',source='living',destination='kitchen',
              filling_position=[0.,0.,1.],serving_position=[0.,7.,1.],serving_room='dining')
    c.manifest={'bindings':[info],'storage':{}}
    c.filling_sites=['kitchen','island'];c.filling_surface_info={'island':{'kind':'island'}}
    c.content_states={'cup':'empty'};c.report={};c.phase='GATHER';c.attached=False
    c.record=lambda **kw:None;c.current_room=lambda info:'island'
    c.data=SimpleNamespace(body=lambda name:SimpleNamespace(xpos=np.array([1.,0.,1.])))
    return c,info


def test_alternate_surface_counts_for_fill_and_is_retained_as_serving_source():
    c,info=gather()
    c.transfer_role('cup')
    assert info['destination']==info['filling_site']=='island'
    assert c.all_at_counter()
    c.content_states['cup']='filled';c.configure_phase('SERVE')
    assert info['source']=='island' and info['destination']=='dining'
    with patch.object(BreakfastEpisode,'transfer_role') as transfer:
        c.verify_role=lambda role:False
        c.transfer_role('cup')
        transfer.assert_called_once_with(c,'cup')


def test_active_stove_cannot_satisfy_human_fill_barrier():
    c,info=gather();c.filling_surface_info={'island':{'kind':'stove','body':'stove'}}
    c.model=None
    with patch('cross_episode_sim.manipulation.kitchen_surfaces.stove_is_off',return_value=False):
        assert not c.all_at_counter()


def test_full_counter_falls_back_to_another_surface_and_updates_filling_target():
    c=BreakfastEpisode.__new__(BreakfastEpisode)
    c.object_name='cup';info={'destination':'kitchen','destination_position':[0.,0.,1.]}
    c.object_info={'cup':info};c.destination='counter'
    c.task_supports={'kitchen':'counter','island':'island_body'}
    c.task_outward={'kitchen':[0.,-1.],'island':[0.,-1.]}
    c.filling_sites=['kitchen','island'];c._redock_index=1;c.model=c.data=None
    c.record=lambda **kw:None;c.navigation=[]
    def prepare():
        if info['destination']=='kitchen':
            raise RuntimeError('No supported upright placement region is clear')
        c.destination_pose=np.eye(4);c.destination_pose[:3,3]=[2.,-.4,1.]
        c.placement_pose_options=[]
    c.prepare_destination=prepare
    c.navigate_to_site=lambda site,point,loaded:c.navigation.append(site)
    with patch('cross_episode_sim.tasks.breakfast.episode.support_bounds',
               return_value=(np.array([1.,-.5,.9]),np.array([3.,.5,1.]))):
        c.relocate_filling_placement()
    assert c.navigation==['island']
    assert c.destination=='island_body'
    assert info['destination']==info['filling_site']=='island'
    assert info['filling_position']==[2.,-.4,1.]
    assert not c._search_whole_placement_surface


def open_counter_scene():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="floor_room_main" pos="0 0 -.02"><geom type="box" size="4 3 .02"/></body>
      <body name="counter_narrow_main" pos="-1 0 .45"><geom type="box" size=".6 .35 .45"/></body>
      <body name="coffee_machine" pos="-1 -.04 1.05"><geom type="box" size=".5 .26 .15"/></body>
      <body name="counter_open_main" pos="1 0 .45"><geom type="box" size=".7 .35 .45"/></body>
      <body name="cup" pos="0 -2 .96"><freejoint/><geom type="box" size=".045 .045 .06"/></body>
      <body name="cup_two" pos="0 -2.3 .96"><freejoint/><geom type="box" size=".045 .045 .06"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    surfaces=discover_filling_surfaces(m,d,'counter_narrow_main')
    return m,d,surfaces


def test_open_surface_beats_narrow_appliance_gap_and_restores_collision_groups():
    m,d,surfaces=open_counter_scene();original=m.geom_group.copy()
    candidates=filling_candidates(m,d,'cup',surfaces)
    assert candidates and candidates[0]['site']=='kitchen_surface_counter_open_main'
    assert all(c['hand_clearance_m']>=.10 and c['approach_clearance_m']>=.40 for c in candidates)
    np.testing.assert_array_equal(m.geom_group,original)


def test_gather_targets_reserve_room_for_both_cups_without_changing_spawn_or_serving():
    m,d,surfaces=open_counter_scene();before=d.qpos.copy()
    manifest={'bindings':[dict(role=name,body=name,initial_source='living',serving_position=[0,5,1])
                          for name in ('cup','cup_two')]}
    choices=allocate_filling_targets(m,d,manifest,surfaces)
    assert len(choices)==2
    assert all(c['site']=='kitchen_surface_counter_open_main' for c in choices.values())
    a,b=[c['footprint'] for c in choices.values()]
    assert abs(choices['cup']['position'][0]-choices['cup_two']['position'][0])>=.19
    np.testing.assert_array_equal(d.qpos,before)
    assert all(i['serving_position']==[0,5,1] for i in manifest['bindings'])
