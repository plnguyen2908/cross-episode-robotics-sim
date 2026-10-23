import yaml
import pytest
from cross_episode_sim.generation.variation_sampling import sample_file

GROUPS = {
 'execution_order': {'procedure': 'by_object_category'},
 'kitchen_layout': {'scene.kitchen_layout': 3},
 'object_instances': {'scene.asset_instance_set': 'same_category_alternatives'},
 'furniture_translation': {'scene.furniture.sofa.dx': {'uniform': [.1,.2]}, 'scene.furniture.sofa.dy': .2},
 'furniture_rotation': {'scene.furniture.sofa.yaw_deg': 8},
 'door_route': {'room_route': 'living_room_detour'},
}


def write(tmp_path, **kwargs):
 p=tmp_path/'sampling.yaml'
 s=dict(task='set_table',seed=42,count=20,controls=GROUPS)
 s.update(kwargs);p.write_text(yaml.safe_dump(s));return p


@pytest.mark.parametrize('n',[0,1,6])
def test_subset_sizes_and_repeatability(tmp_path,n):
 p=write(tmp_path,randomize_controls=dict(min_active=n,max_active=n))
 a=sample_file(p)
 assert a==sample_file(p) and len(a)==20
 for c in a:
  selected=c['sampling']['active_controls']
  assert len(selected)==n
  assert (c['procedure']=='by_object_category') == ('execution_order' in selected)
  assert (c['scene']['kitchen_layout']==3)==('kitchen_layout' in selected)
  assert ('room_doors' in c['scene'])==('door_route' in selected)
  if 'furniture_translation' in selected:
   assert c['scene']['furniture']['sofa']['dy']==.2
 if n==0:
  assert all(c['scene']['furniture']=={} for c in a)


def test_mixed_subsets(tmp_path):
 a=sample_file(write(tmp_path,randomize_controls=dict(min_active=0,max_active=6)))
 assert len({len(c['sampling']['active_controls']) for c in a})>3


@pytest.mark.parametrize('kw',[
 {'randomize_controls':dict(min_active=0,max_active=7)},
 {'controls':{'original':{'procedure':'by_place_setting'}}},
 {'controls':{'execution_order':{'scene.kitchen_layout':3}}},
 {'mode':'one_at_a_time'},
])
def test_invalid(tmp_path,kw):
 with pytest.raises(ValueError):sample_file(write(tmp_path,**kw))


def test_all_templates():
 from pathlib import Path
 for p in Path('configs/composite_sampling').glob('*.yaml'):
  a=sample_file(p)
  assert len(a)==7
  spec=a[0]['sampling']['specification']
  assert len(spec['controls'])==(5 if spec['task']=='warm_two_pots' else 6)


def test_optional_appearance_groups(tmp_path):
 controls=dict(GROUPS,wall_color={'scene.appearance.wall_rgb':{'choices':[[.2,.3,.4]]}},
               object_color={'scene.appearance.object_rgb':[.4,.3,.2]},
               furniture_instances={'scene.furniture_set':'compact_chairs'})
 a=sample_file(write(tmp_path,controls=controls,randomize_controls=dict(min_active=9,max_active=9)))
 assert a[0]['scene']['appearance']['wall_rgb']==[.2,.3,.4]
 assert a[0]['scene']['furniture_set']=='compact_chairs'


def test_room_count_combines_with_route(tmp_path):
 controls={'room_layout':{'scene.room_count':6,'scene.room_arrangement':'branch'},
           'door_route':{'room_route':'open_shortcut','scene.room_doors.room_door_5':'closed'}}
 a=sample_file(write(tmp_path,controls=controls,randomize_controls=dict(min_active=2,max_active=2)))
 assert a[0]['scene']['room_count']==6
 assert a[0]['scene']['room_doors']['room_door_5']=='closed'
 from cross_episode_sim.generation.room_routes import room_door_states
 assert len(room_door_states(a[0]['scene']))==6
