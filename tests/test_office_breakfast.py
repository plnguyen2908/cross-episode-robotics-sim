"""Controlled scene randomness and office goal dispatch regressions."""
import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

from cross_episode_sim.tasks.breakfast.office import (
    DEFAULT_CONFIG, OfficeBreakfastEpisode, factor_rng, validate_config, instruction, source_candidates)
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode


def config():return json.loads(DEFAULT_CONFIG.read_text())


def test_reproducible_factor_streams_and_two_day_persistence():
    c=config();v=copy.deepcopy(c['variants'][2]);v['day']=2
    a=factor_rng(c,v,'furniture').uniform(size=8)
    np.testing.assert_array_equal(a,factor_rng(c,v,'furniture').uniform(size=8))
    v['day']=3
    np.testing.assert_array_equal(a,factor_rng(c,v,'furniture').uniform(size=8))
    v['day']=4
    assert not np.array_equal(a,factor_rng(c,v,'furniture').uniform(size=8))


def test_object_stream_independent_of_clutter_sampling():
    c=config();v=c['variants'][1]
    expected=factor_rng(c,v,'objects').uniform(size=8)
    factor_rng(c,v,'clutter').uniform(size=1000)
    c['clutter_count_per_room']=[0,0]
    np.testing.assert_array_equal(expected,factor_rng(c,v,'objects').uniform(size=8))


def test_sources_cover_the_strip_reproducibly_in_two_dimensions():
    c=config();v=c['variants'][0]
    a=np.asarray(source_candidates(c,factor_rng(c,v,'objects')))
    np.testing.assert_array_equal(a,source_candidates(c,factor_rng(c,v,'objects')))
    assert np.ptp(a[:,0])>.8 and np.ptp(a[:,1])>.065
    assert np.all((a[:,0]>=.08)&(a[:,0]<=.92))
    assert np.all((a[:,1]>=.025)&(a[:,1]<=.10))
    next_day=dict(v,day=v['day']+1)
    assert not np.array_equal(a,source_candidates(c,factor_rng(c,next_day,'objects')))
    assert all(v['objects'] for v in c['variants'])


def test_allocator_uses_continuous_candidates_without_fixed_slot_fallback():
    import mujoco
    from cross_episode_sim.tasks.breakfast.scene import allocate_sites, SUPPORTS
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="{SUPPORTS['kitchen']}"><geom type="box" pos="0 0 .7" size="1 .4 .04"/></body>
      <body name="item" pos="0 0 1"><geom type="box" size=".04 .04 .04"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    info={};sample={}
    request=dict(body='item',room='kitchen',field='position',info=info,
                 candidates=[(.123456,.076543)],sample_record=sample)
    reserved=allocate_sites(model,data,[request])
    assert sample==dict(spawn_fraction=.123456,spawn_inset_m=.076543)
    np.testing.assert_allclose(info['position'][:2],[-1+2*.123456,-.4+.076543+.04])
    with pytest.raises(ValueError,match='No separated supported accessible site'):
        allocate_sites(model,data,[request],reserved)


@pytest.mark.parametrize('key,value',[('people',3),('desk_translation_m',[4,0]),
    ('object_yaw_jitter_deg',90),('clutter_count_per_room',[0,9])])
def test_unvalidated_ranges_rejected(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):validate_config(c)


def test_goal_already_satisfied_skips_manipulation():
    c=OfficeBreakfastEpisode.__new__(OfficeBreakfastEpisode)
    c.verify_role=lambda role:True;events=[];c.record=lambda **kw:events.append(kw)
    with patch.object(BreakfastEpisode,'transfer_role',side_effect=AssertionError('unnecessary transfer')):
        c.transfer_role('cup_one')
    assert events==[dict(task_role='cup_one',goal_already_satisfied=True,skipped_transfer=True)]


def test_missing_goal_uses_existing_transfer():
    c=OfficeBreakfastEpisode.__new__(OfficeBreakfastEpisode);c.verify_role=lambda role:False
    with patch.object(BreakfastEpisode,'transfer_role',return_value='done') as transfer:
        assert c.transfer_role('bowl_one')=='done'
        transfer.assert_called_once_with('bowl_one')


def test_equipment_obstacles_restored_after_failed_placement_planning():
    c=OfficeBreakfastEpisode.__new__(OfficeBreakfastEpisode)
    original=[{'body':'clutter'}];c.manifest={'fixed_props':['monitor'],'background':original}
    def check(owner):
        assert owner.manifest['background']==original+[{'body':'monitor','task_object':False}]
        raise RuntimeError('planning failure')
    with patch.object(BreakfastEpisode,'prepare_destination',check):
        with pytest.raises(RuntimeError):c.prepare_destination()
    assert c.manifest['background'] is original


def test_instruction_refers_to_office_and_satisfied_goals():
    text=instruction(config())
    assert 'office desk' in text and 'fetch only missing' in text
    assert 'book' not in text


def test_replay_caption_follows_active_object_not_last_annotation():
    c=OfficeBreakfastEpisode.__new__(OfficeBreakfastEpisode)
    c.manifest={'bindings':[dict(body='cup',role='cup_one',asset='Mug_1'),
                           dict(body='syrup',role='sweet_condiment',asset='bottles/Syrup004')]}
    c.annotation_asset='Mug_1'
    c.object_name='syrup'
    assert c.object_label()=='sweet_condiment (Syrup004)'
    c.object_name='cup'
    assert c.object_label()=='cup_one (Mug_1)'


def test_office_equipment_visible_and_collidable():
    import mujoco
    import xml.etree.ElementTree as ET
    from cross_episode_sim.tasks.breakfast.office import add_office_equipment
    root=ET.fromstring('<mujoco><worldbody/></mujoco>')
    names=add_office_equipment(root,np.array([0.,0.,.7]),np.array([2.,1.,.8]))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert len(names)==3
    assert np.all(model.geom_group==1)
    assert np.all(model.geom_contype==1)


def test_initial_goal_check_before_first_grasp():
    from types import SimpleNamespace
    from cross_episode_sim.tasks.breakfast.scene import SUPPORTS
    c=OfficeBreakfastEpisode.__new__(OfficeBreakfastEpisode)
    c.manifest={'bindings':[dict(role='cup_one',body='cup',initial_quaternion=[1,0,0,0],
        destination='dining',destination_position=[1.,2.,.8])]}
    c.report={};c.attached=False
    c.assignment=lambda:{'cup':SUPPORTS['dining']}
    c.data=SimpleNamespace(body=lambda name:SimpleNamespace(xpos=np.array([1.,2.,.8]),xmat=np.eye(3)))
    assert c.verify_role('cup_one')
    c.holding_loaf=True
    assert not c.verify_role('cup_one')


def test_all_goals_initially_satisfied_can_finish_without_any_grasp():
    c=OfficeBreakfastEpisode.__new__(OfficeBreakfastEpisode)
    c.manifest={'bindings':[{'role':'cup_one'}]};c.attached=False
    c.validate_population=lambda:None;c.verify_role=lambda role:True
    with patch('cross_episode_sim.tasks.breakfast.office.CompositeEpisode') as episode:
        episode.return_value.run.side_effect=lambda *args:episode.call_args.args[2]()=={'cup_one':True,'empty_hand':True}
        assert c.run_test()==0


@pytest.mark.parametrize('wall_x,blocked',[(0.5,True),(2.,False)])
def test_shifted_furniture_cannot_intersect_walls(wall_x,blocked):
    import mujoco
    from cross_episode_sim.tasks.breakfast.office import validate_furniture
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="desk"><geom name="table" type="box" pos="0 0 .8" size=".6 .4 .04"/></body>
      <body name="wall_office"><geom name="wall" type="box" pos="{wall_x} 0 1" size=".05 2 1"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    if blocked:
        with pytest.raises(ValueError,match='intersects structure'):validate_furniture(model,data,['desk'])
    else:validate_furniture(model,data,['desk'])
