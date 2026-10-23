from copy import deepcopy
import json
import pytest
from cross_episode_sim.generation import household_scene
from cross_episode_sim.generation.composite_scenes import warm_pots


def test_pots_do_not_require_dining_destination():
    spec=warm_pots({'object_counts':{'pot':2}, 'minimum_heating_seconds':30})
    assert spec['reserve_dining_destinations'] is False
    assert all(o['support']=='kitchen_counter' for o in spec['objects'].values())


def test_retry_preserves_controls_and_records_rejection(tmp_path, monkeypatch):
    params={'seed':7,'base_layout':{'seed':123,'room_count':6},'furniture':{'test':1}}
    original=deepcopy(params); seen=[]
    def build(output, current, spec):
        seen.append(deepcopy(current))
        (output/'scene.xml').write_text('<mujoco/>')
        if len(seen)==1:
            raise ValueError('No separated supported accessible site: pot/kitchen')
        return {'success':True}
    monkeypatch.setattr(household_scene,'_build_scene_once',build)
    assert household_scene.build_scene(tmp_path,params,{})=={'success':True}
    assert params==original
    assert seen[0]['base_layout']['seed']!=seen[1]['base_layout']['seed']
    assert seen[0]['seed']==seen[1]['seed']==7
    assert seen[0]['furniture']==seen[1]['furniture']
    assert (tmp_path/'rejected_setups/01/scene.xml').is_file()
    records=json.loads((tmp_path/'scene_setup_attempts.json').read_text())
    assert [r['success'] for r in records]==[False,True]


def test_unknown_error_does_not_retry(tmp_path,monkeypatch):
    seen=[]
    def build(*args):
        seen.append(1);raise ValueError('Unknown asset category')
    monkeypatch.setattr(household_scene,'_build_scene_once',build)
    with pytest.raises(ValueError,match='Unknown asset'):
        household_scene.build_scene(tmp_path,{'base_layout':{'seed':1}}, {})
    assert len(seen)==1


def test_retry_is_bounded(tmp_path,monkeypatch):
    seen=[]
    def build(*args):
        seen.append(1);raise ValueError('No valid sampled house after 100 attempts')
    monkeypatch.setattr(household_scene,'_build_scene_once',build)
    with pytest.raises(ValueError,match='No valid sampled house'):
        household_scene.build_scene(tmp_path,{'base_layout':{'seed':1,'setup_attempts':2}}, {})
    assert len(seen)==2


def test_rotated_table_searches_supported_interior():
    import mujoco
    from cross_episode_sim.tasks.breakfast.scene import allocate_sites, SUPPORTS
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="{SUPPORTS['dining']}" pos="0 0 .7" euler="0 0 10">
        <geom type="box" size=".7 .4 .03"/>
      </body>
      <body name="object" pos="2 2 1"><freejoint/>
        <geom type="box" size=".15 .15 .1"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    def request():return dict(body='object',room='dining',field='position',info={},preferred=.5)
    edge=request();edge['inset']=.025
    with pytest.raises(ValueError,match='No separated supported'):
        allocate_sites(m,d,[edge])
    interior=request()
    allocate_sites(m,d,[interior])
    assert len(interior['info']['position'])==3
