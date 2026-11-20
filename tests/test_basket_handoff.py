"""Exercise the metadata contract between observation, posture, dock and place."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.runtime.basket_placement import place_in_basket
from cross_episode_sim.runtime.basket_placement import refine_basket_dock


@pytest.fixture(autouse=True)
def isolate_existing_handoff_tests(monkeypatch):
    monkeypatch.setattr('cross_episode_sim.runtime.basket_placement.refine_basket_dock',
                        lambda pipeline,root,source,*args:source)


@pytest.mark.parametrize('distance,redock',[(.82,True),(.60,False)])
def test_fresh_interior_refinement_before_arm_planning(tmp_path,distance,redock):
    calls=[]
    x,y=np.meshgrid(np.linspace(distance-.08,distance+.08,20),np.linspace(-.08,.08,20))
    floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,.93)]
    class Pipeline:
        def stage(self,name,command,report,success_key):
            calls.append(command)
            report.parent.mkdir()
            report.write_text(json.dumps({success_key:True}))
            if '--observe-only' in command:
                np.savez(report.parent/'input.npz',table_surface=floor,points=floor)
    source=tmp_path/'arrival'
    result=refine_basket_dock(Pipeline(),tmp_path,source,'purple basket','inside bottom','kitchen',4)
    assert '--observe-only' in calls[0]
    assert len(calls)==(2 if redock else 1)
    if redock:
        assert calls[1][1]=='scripts/test_basket_place_redock.py'
        assert calls[1][calls[1].index('--start')+1]==str(source)
        assert result.name=='basket_refined_dock'
    else:assert result==source


def test_post_raise_checkpoint_retains_arrival_metadata(tmp_path,monkeypatch):
    monkeypatch.setattr('cross_episode_sim.navigation.receptacle_approach.approach_receptacle',lambda *a,**k:pytest.fail('Visible basket must go directly to Qwen docking'))
    monkeypatch.setattr('cross_episode_sim.control.raised_payload_posture.raise_payload_clearance',lambda *a,**k:None)
    carry=tmp_path/'carry';carry.mkdir();np.save(carry/'observed_payload_tool.npy',np.zeros((8,3)))
    stages=[]
    class Pipeline:
        runtime=SimpleNamespace()
        def checkpoint(self,path,previous):
            path.mkdir();return path  # Checkpoint does not create a stage report.
        def stage(self,name,command,report,success_key):
            stages.append(name)
            if '--start' in command:
                arrival=Path(command[command.index('--start')+1])/'report.json'
                assert json.loads(arrival.read_text())['initialization_valid'] is True
            report.parent.mkdir(exist_ok=True)
            report.write_text(json.dumps({success_key:True,'placement_footprint_available':False}))
    placed,dock=place_in_basket(Pipeline(),tmp_path,carry,'purple basket',
        'inside bottom of the purple basket','kitchen',2,lambda *a:None)
    assert stages==['2_observe_receptacle','2_place_dock','2_place']
    assert placed.name=='place' and dock.name=='loaded_tuck_handoff'
    assert json.loads((dock/'report.json').read_text())['docking_success']


@pytest.mark.parametrize('recovered',[False,True])
def test_missing_floor_reacquired_before_docking(tmp_path,monkeypatch,recovered):
    monkeypatch.setattr('cross_episode_sim.navigation.receptacle_approach.approach_receptacle',lambda *a,**k:None)
    raises=[]
    monkeypatch.setattr('cross_episode_sim.control.raised_payload_posture.raise_payload_clearance',lambda *a,**k:raises.append(a[2]))
    carry=tmp_path/'carry';carry.mkdir();np.save(carry/'observed_payload_tool.npy',np.zeros((8,3)))
    stages=[]
    class Pipeline:
        runtime=SimpleNamespace()
        def checkpoint(self,path,previous):path.mkdir();return path
        def stage(self,name,command,report,success_key):
            stages.append(name)
            success=(recovered if name.endswith('observe_receptacle_retry') else False) if 'observe_receptacle' in name else True
            value={success_key:success}
            report.parent.mkdir(exist_ok=True);report.write_text(json.dumps(value))
            if '--perception' in command:
                directory=Path(command[command.index('--perception')+1])
                assert directory.name=='receptacle_perception_retry'
                assert json.loads((directory/'report.json').read_text())['observation_success']
    call=lambda:place_in_basket(Pipeline(),tmp_path,carry,'orange basket',
        'inside bottom of the orange basket','kitchen',6,lambda *a:None)
    if recovered:call();assert '6_place_dock' in stages
    else:
        with pytest.raises(RuntimeError,match='still unobserved'):call()
        assert '6_place_dock' not in stages
    if not recovered:assert raises[0].name=='basket_view_recovery'
    else:assert raises[0].name=='basket_clearance'


def test_failed_dock_preserves_reason_without_invalid_placement_call(tmp_path,monkeypatch):
    monkeypatch.setattr('cross_episode_sim.control.raised_payload_posture.raise_payload_clearance',lambda *a,**k:None)
    carry=tmp_path/'carry';carry.mkdir();np.save(carry/'observed_payload_tool.npy',np.zeros((8,3)))
    stages=[]
    class Pipeline:
        runtime=SimpleNamespace()
        def checkpoint(self,path,previous):path.mkdir();return path
        def stage(self,name,command,report,success_key):
            stages.append(name)
            report.parent.mkdir(exist_ok=True)
            report.write_text(json.dumps({success_key:success_key!='docking_success',
                'error':'No observed route'}))
    placed,dock=place_in_basket(Pipeline(),tmp_path,carry,'purple basket',
        'inside bottom of the purple basket','kitchen',2,lambda *a:None)
    report=json.loads((placed/'report.json').read_text())
    assert report['error']=='No observed route'
    assert not report['execution_success'] and not report['manipulation_attempted']
    assert stages==['2_observe_receptacle','2_place_dock']


def test_hidden_floor_recovery_only_after_real_dock_and_fresh_failure(tmp_path,monkeypatch):
    events=[]
    monkeypatch.setattr('cross_episode_sim.control.raised_payload_posture.raise_payload_clearance',
        lambda *a,**kw:events.append(('posture',kw.get('lateral_offset'))))
    carry=tmp_path/'carry';carry.mkdir();np.save(carry/'observed_payload_tool.npy',np.zeros((8,3)))
    class Pipeline:
        runtime=SimpleNamespace()
        stages=[]
        def checkpoint(self,path,previous):path.mkdir();return path
        def stage(self,name,command,report,success_key):
            events.append(('stage',name))
            success=name!='2_place'
            self.stages.append(dict(name=name,success=success))
            report.parent.mkdir(exist_ok=True)
            report.write_text(json.dumps({success_key:success,'phase':'observe'}))
    pipeline=Pipeline()
    placed,_=place_in_basket(pipeline,tmp_path,carry,'purple basket','inside bottom','kitchen',2,lambda *a:None)
    assert events.index(('posture',-.25))>events.index(('stage','2_place'))>events.index(('stage','2_place_dock'))
    # The place is tried first from the raised arm that just showed the interior.
    assert placed.name=='place_revealed_0'
    assert pipeline.stages[2]['recovered_by']=='2_place_revealed_0'


def test_recovery_redocks_from_revealed_interior_before_replanning(tmp_path,monkeypatch):
    events=[]
    monkeypatch.setattr('cross_episode_sim.control.raised_payload_posture.raise_payload_clearance',
        lambda *a,**kw:events.append(('arm',kw.get('lateral_offset'))))
    def refine(pipeline,root,source,*args):
        events.append(('refine',root.name));return source
    monkeypatch.setattr('cross_episode_sim.runtime.basket_placement.refine_basket_dock',refine)
    carry=tmp_path/'carry';carry.mkdir();np.save(carry/'observed_payload_tool.npy',np.zeros((8,3)))
    class Pipeline:
        runtime=SimpleNamespace();stages=[]
        def checkpoint(self,path,previous):path.mkdir();return path
        def stage(self,name,command,report,success_key):
            events.append(('stage',name));report.parent.mkdir(exist_ok=True)
            success=name!='2_place';self.stages.append(dict(name=name,success=success))
            report.write_text(json.dumps({success_key:success,'phase':'observe'}))
    place_in_basket(Pipeline(),tmp_path,carry,'purple basket','inside bottom','kitchen',2,lambda *a:None)
    assert events.index(('arm',-.25)) < events.index(('refine','basket_recovery_0')) < events.index(('stage','2_place_revealed_0'))
