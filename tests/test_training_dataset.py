import json
from pathlib import Path
import pytest
from cross_episode_sim.generation.training_dataset import episode_config, needs_more, summarize, TASKS
from cross_episode_sim.generation.two_room_layout import sample_two_room_layout
from cross_episode_sim.generation.sampled_house import resolve_route
from cross_episode_sim.tasks.composites.room_route import route_plan


def test_two_room_closed_route_uses_physical_open_calls():
    layout = sample_two_room_layout(7, [-3, -2, 3, 3])
    assert [r['room_type'] for r in layout['rooms']] == ['kitchen', 'dining']
    route = resolve_route(layout, 'closed_direct')
    assert route['doors_to_open'] == [layout['connections'][0]['door']]
    c = episode_config('train', 'set_table', 0, 123)
    c['resolved_route'] = route
    assert [x['skill'] for x in route_plan(c, [])] == ['navigate','open_door','navigate','open_door']
    with pytest.raises(ValueError, match='cycle'): resolve_route(layout,'living_room_detour')


def test_profiles_and_schedule_are_reproducible():
    for task in TASKS:
        a = episode_config('train',task,0,123)
        b = episode_config('train',task,1,123)
        assert a == episode_config('train',task,0,123)
        assert a['scene']['base_layout']['room_count'] == 2
        assert a['procedure'] != b['procedure']
        if task != 'warm_two_pots':
            assert a['room_route']=='closed_direct' and episode_config('train',task,2,123)['room_route']=='default'
        test = episode_config('test_multi_room',task,0,123)
        assert 5<=test['scene']['base_layout']['room_count']<=7
        assert test['scene']['kitchen_layout'] != a['scene']['kitchen_layout']
        assert test['scene']['base_layout']['seed'] != a['scene']['base_layout']['seed']


def test_duration_alone_cannot_satisfy_training():
    task='set_table'
    s={'per_task_seconds':{task:3600},'coverage':{task:{'procedures':TASKS[task]['procedures'],'opened_room_door':False,'routes':['default']}}}
    assert needs_more(s,task,3600,'train')
    s['coverage'][task]['opened_room_door']=True
    assert not needs_more(s,task,3600,'train')


def test_failed_and_metadata_incomplete_episodes_do_not_count(tmp_path, monkeypatch):
    c=episode_config('train','set_table',0,123)
    batch=tmp_path/'batches'/'round_0000'; folder=batch/c['episode_id'];folder.mkdir(parents=True)
    (batch/'batch_configs.json').write_text(json.dumps([c]))
    (folder/'report.json').write_text(json.dumps({'success':False,'video':'composite.mp4'}))
    assert summarize(tmp_path)['successful_seconds']==0
    (folder/'report.json').write_text(json.dumps({'success':True,'video':'composite.mp4'}))
    (folder/'composite.mp4').write_bytes(b'video')
    (folder/'skill_calls.json').write_text(json.dumps([{'skill':'open_door','arguments':{'target':'room_door_0_left'},'success':True}]))
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'10.0')
    assert summarize(tmp_path)['successful_seconds']==0
    (folder/'house_overview.png').write_bytes(b'image')
    (folder/'asset_selection.json').write_text(json.dumps({'selected':{'cup_01':'Mug_1'}}))
    (folder/'initial_snapshot.json').write_text(json.dumps({'objects':{'cup_01':{'category':'cup'}}}))
    result=summarize(tmp_path)
    assert result['successful_seconds']==10
    assert result['episodes'][0]['objects'][0]['asset_id']=='Mug_1'
    assert list((tmp_path/'layouts').glob('*.png'))


def test_unbounded_attempt_seeds_do_not_cross_task_or_split_domains():
    seeds = set()
    for split in ('train', 'test_two_rooms', 'test_multi_room'):
        for task in TASKS:
            for index in (0, 9999, 10000, 10001, 1000000):
                c = episode_config(split, task, index, 123)
                seed = c['dataset_sampling']['master_seed']
                assert seed not in seeds
                seeds.add(seed)
    # Existing configs retain their original seed convention.
    assert episode_config('train', 'set_table', 30, 123)['dataset_sampling']['master_seed'] == 153


def test_queue_fills_eight_slots_with_five_tasks_and_resumes_ids(tmp_path):
    from cross_episode_sim.generation.training_dataset import queue_configs
    first = queue_configs('train',list(TASKS),8,123,tmp_path)
    assert len(first)==8 and len({c['episode_id'] for c in first})==8
    folder=tmp_path/'round_0000';folder.mkdir()
    (folder/'batch_configs.json').write_text(json.dumps(first))
    second = queue_configs('train',list(TASKS),8,123,tmp_path)
    assert not {c['episode_id'] for c in first} & {c['episode_id'] for c in second}
    assert len(queue_configs('train',['set_table'],8,123,tmp_path))==8


def test_runtime_sets_asset_root_and_rejects_bad_install(monkeypatch, tmp_path):
    from cross_episode_sim.generation.training_dataset import configure_runtime
    monkeypatch.delenv('ROBOCASA_DIR',raising=False)
    configure_runtime()
    import os
    assert Path(os.environ['ROBOCASA_DIR'],'models/objects/kitchen_objects.py').is_file()
    monkeypatch.setenv('ROBOCASA_DIR',str(tmp_path))
    with pytest.raises(RuntimeError,match='Invalid ROBOCASA_DIR'):configure_runtime()



def test_hour_is_per_task_not_shared():
    task='set_table'
    s={'per_task_seconds':{task:3599},'coverage':{task:{'procedures':TASKS[task]['procedures'],'opened_room_door':True,'routes':['default']}}}
    assert needs_more(s,task,3600,'train')
    s['per_task_seconds'][task]=3600
    assert not needs_more(s,task,3600,'train')
    s['per_task_seconds'][task]=899
    assert needs_more(s,task,900,'test_two_rooms')


def test_per_task_manifests_exclude_other_tasks_and_rebase_paths(tmp_path):
    from cross_episode_sim.generation.training_dataset import write_task_manifests
    root=tmp_path/'train';root.mkdir()
    rows=[]
    for task in TASKS:
        rows.append(dict(id=task,task=task,**{k:'batches/'+task+'/'+k for k in ('folder','layout_image','config','prompt','calls','video')}))
    summary=dict(episodes=rows,per_task_seconds={t:3600 for t in TASKS},coverage={t:{'procedures':TASKS[t]['procedures'],'opened_room_door':True,'routes':['default']} for t in TASKS})
    write_task_manifests(root,summary,{'mug':{'asset_id':'mug','categories':['cup'],'episodes':['set_table']}})
    dest=tmp_path/'tasks/set_table/train'
    doc=json.loads((dest/'manifest.json').read_text())
    assert doc['target_seconds']==3600 and doc['complete']
    assert [r['id'] for r in doc['episodes']]==['set_table']
    assert (dest/doc['episodes'][0]['video']).resolve()==(root/'batches/set_table/video').resolve()
    assert json.loads((tmp_path/'tasks/warm_two_pots/train/objects.json').read_text())==[]
