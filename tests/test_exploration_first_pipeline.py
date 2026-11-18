import json
from pathlib import Path
import pytest
from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline


@pytest.fixture(autouse=True)
def articulated_sources(monkeypatch):
    # These stub sources have no compiled model; the frozen-joint check needs one.
    monkeypatch.setattr('cross_episode_sim.evaluation.fixture_articulation.require_articulated',
                        lambda source:dict(fixture_joints_checked=True))


def source(tmp_path):
    p=tmp_path/'source';p.mkdir()
    (p/'config.json').write_text(json.dumps({'task':'set_table','scene':{'people':1}}))
    (p/'room_doors.json').write_text(json.dumps({'door':'open'}))
    return p


def memory(p, confirmed=True):
    p.mkdir(parents=True,exist_ok=True)
    (p/'semantic_memory.json').write_text(json.dumps({'tables':[{'label':'dining table','observations':2 if confirmed else 1}], 'furniture':[{'label':label,'observations':2,'viewpoints':[[0,0,0]]} for label in ('kitchen','dining table')]}))
    for name in ('navigation_map.npz','slam/estimated_base_poses.npy','private_checkpoint/state.npy'):
        f=p/name;f.parent.mkdir(exist_ok=True);f.touch()


def test_exploration_failure_blocks_manipulation(tmp_path):
    class Runner(SensorSetTablePipeline):
        def stage(self,name,*args,**kwargs):
            assert name=='explore'
            raise RuntimeError('No observed connecting route')
    result=Runner(source(tmp_path),tmp_path/'run',persistent_runtime=False).run()
    assert not result['success']
    assert result['error']=='Exploration failed: backend_error'
    assert not list((tmp_path/'run').glob('transfer_*'))


def test_navigation_continues_exploration_state_before_grasp(tmp_path):
    commands=[]
    class Runner(SensorSetTablePipeline):
        def stage(self,name,command,report,success_key='success'):
            commands.append((name,command))
            if name.endswith('_pick'):raise RuntimeError('test stops before physics')
            memory(Path(report).parent)
            return {success_key:True}
    runner=Runner(source(tmp_path),tmp_path/'run',persistent_runtime=False);runner.run()
    assert [n for n,c in commands]==['explore','1_transit_kitchen','1_navigate_0','1_pick']
    explore=commands[0][1];transit=commands[1][1];nav=commands[2][1]
    assert '--record-room-memory' not in explore
    assert '--finish-station-approach' not in explore
    assert '--fixed-map' in nav
    assert transit[transit.index('--start')+1]==str(tmp_path/'run/exploration')
    assert nav[nav.index('--continuation')+1]==str(tmp_path/'run/transfer_01_bowl/kitchen_transit')
    assert nav[nav.index('--goal-room')+1]==''


def test_unconfirmed_memory_rejected(tmp_path):
    memory(tmp_path/'survey',confirmed=False)
    with pytest.raises(RuntimeError,match='confirmed dining table'):
        SensorSetTablePipeline.require_destination_memory(tmp_path/'survey')


def test_exploration_only_never_launches_pickup(tmp_path):
    names=[]
    class Runner(SensorSetTablePipeline):
        def stage(self,name,command,report,success_key='success'):
            names.append(name)
            memory(Path(report).parent)
            return {success_key:True}
    result=Runner(source(tmp_path),tmp_path/'run',persistent_runtime=False).run(exploration_only=True)
    assert names==['explore']
    assert result['scope']=='exploration only'
    assert result['success'] and not result['full_task_success']
    assert not list((tmp_path/'run').glob('transfer_*'))


def test_furniture_labels_are_collected_once_and_reused_for_transit(tmp_path):
    commands=[]
    class Runner(SensorSetTablePipeline):
        def stage(self,name,command,report,success_key='success'):
            commands.append(command)
            if name.endswith('_navigate_0'):raise RuntimeError('stop at handoff')
            memory(Path(report).parent)
            return {success_key:True}
    Runner(source(tmp_path),tmp_path/'run',furniture_device='cuda:2',persistent_runtime=False).run()
    assert commands[0][commands[0].index('--record-furniture-device')+1]=='cuda:2'
    assert commands[1][commands[1].index('--receptacle')+1]=='kitchen'
    assert '--record-furniture-device' not in commands[1]


def test_missing_kitchen_station_blocks_pipeline(tmp_path):
    p=tmp_path/'survey';memory(p)
    value=json.loads((p/'semantic_memory.json').read_text())
    value['furniture']=[x for x in value['furniture'] if x['label']!='kitchen']
    (p/'semantic_memory.json').write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='confirmed kitchen station'):
        SensorSetTablePipeline.require_destination_memory(p)


def test_pipeline_navigation_is_journaled_as_public_skills(tmp_path):
    calls=[]
    class Runner(SensorSetTablePipeline):
        def explore(self,python):return self.source
        @staticmethod
        def require_destination_memory(directory):pass
        def stage(self,name,*args,**kwargs):
            calls.append(name)
            if name=='2_place':raise RuntimeError('stop before private final scoring')
            return {'success':True}
    output=tmp_path/'run'
    report=Runner(source(tmp_path),output)._run()
    assert not report['success']
    journal=json.loads((output/'sensor_skill_calls.json').read_text())
    assert len(journal)==9 and all(e['success'] for e in journal[:-1])
    assert journal[-1]['skill']=='place' and not journal[-1]['success']
    assert [e['skill'] for e in journal]==['explore','navigate','pick','navigate','place','navigate','pick','navigate','place']
    assert journal[0]['skill']=='explore'
    journal=[e for e in journal if e['skill']=='navigate']
    assert [e['arguments'] for e in journal]==[
        dict(target='bowl',search_at='kitchen'),dict(target='bowl',destination='dining table'),
        dict(target='cup',search_at='kitchen'),dict(target='cup',destination='dining table')]
    assert calls.index('1_navigate_0')<calls.index('1_pick')<calls.index('1_carry')<calls.index('1_place_dock')<calls.index('1_place')


def test_interface_routes_each_transfer_to_its_actual_backend_and_inputs(tmp_path):
    commands=[]
    class Runner(SensorSetTablePipeline):
        def explore(self,python):return self.source
        @staticmethod
        def require_destination_memory(directory):pass
        def stage(self,name,command,report,success_key='success'):
            commands.append((name,command,success_key))
            if name=='2_place':raise RuntimeError('stop before final scoring')
            return {success_key:True}
    output=tmp_path/'run';Runner(source(tmp_path),output)._run()
    calls={name:(cmd,key) for name,cmd,key in commands}
    for index,category in ((1,'bowl'),(2,'cup')):
        root=output/f'transfer_{index:02d}_{category}'
        carry,key=calls[f'{index}_carry']
        assert carry[1]=='scripts/test_m2t2_carry_dock.py'
        assert carry[carry.index('--pick')+1]==str(root/'pick/physics')
        assert '--known-map-only' in carry and key=='transit_success'
        dock,key=calls[f'{index}_place_dock']
        assert dock[1]=='scripts/test_m2t2_place_redock_astar.py'
        assert dock[dock.index('--start')+1]==str(root/'carry')
        assert dock[dock.index('--perception')+1]==str(root/'table_perception')
        assert dock[dock.index('--continuation-map')+1]==str(root/'carry')
    transit,_=calls['2_transit_kitchen']
    assert transit[transit.index('--start')+1]==str(output/'transfer_01_bowl/return')


@pytest.mark.parametrize('initial_doors',['saved','open','closed'])
def test_open_doors_keep_qualified_station_backend_without_door_metadata(tmp_path,initial_doors):
    commands=[]
    class Runner(SensorSetTablePipeline):
        def stage(self,name,command,report,success_key='success'):
            commands.append((name,command,success_key))
            memory(Path(report).parent)
            return {success_key:True}
    src=source(tmp_path)
    (src/'room_doors.json').unlink()
    result=Runner(src,tmp_path/'run',initial_doors=initial_doors,persistent_runtime=False).run(exploration_only=True)
    assert result['success']
    assert [x[0] for x in commands]==['explore']
    command=commands[0][1]
    if initial_doors=='closed':
        assert command[1]=='scripts/demo_room_coverage_exploration.py'
        assert command[command.index('--initial-doors')+1]=='closed'
    else:
        assert command[1]=='scripts/demo_semantic_navigation.py'
        assert '--station-exploration' in command and '--preserve-initial-state' in command
        assert command[command.index('--target')+1]=='dining table'
        assert commands[0][2]=='sensor_navigation_success'
        assert '--require-stations' not in command
    assert '--finish-station-approach' not in command
    assert '--stop-after-crossing' not in command
    assert '--stop-after-opening' not in command
    assert '--handle-then-push' not in command
    assert '--single-left-leaf' not in command
    assert result['exploration']==str(tmp_path/'run/exploration')
