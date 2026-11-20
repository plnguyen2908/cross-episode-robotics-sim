"""Fix 32/34: in-room search is a fixed turn by default and needs no model; One-Ring only on an explicit address."""
import json
from types import SimpleNamespace
from cross_episode_sim.navigation.onering import OneRingProcess,BASE_ACTIONS


def test_scan_policy_turns_without_a_model(tmp_path,monkeypatch):
    monkeypatch.setenv('SENSOR_MODEL_ENDPOINTS',json.dumps({'onering':'scan'}))
    with OneRingProcess(None,tmp_path/'onering') as policy:
        assert policy.reset() is None
        actions=[policy.predict(SimpleNamespace(timestamp=float(i)),'Find a cup') for i in range(3)]
    assert actions==['r','r','r'] and BASE_ACTIONS['r'][0]==0.
    rows=[json.loads(line) for line in (tmp_path/'onering/predictions.jsonl').read_text().splitlines()]
    assert [row['transport'] for row in rows]==['scan']*3


def test_without_the_scan_setting_the_remote_client_is_used(tmp_path,monkeypatch):
    monkeypatch.setenv('SENSOR_MODEL_ENDPOINTS',json.dumps({'onering':'127.0.0.1:1'}))
    made=[]
    class Client:
        def __init__(self,address,kind=None):made.append((address,kind))
        def call(self,**kw):return None
        def close(self):pass
    monkeypatch.setattr('cross_episode_sim.perception.rpc.ModelClient',Client)
    with OneRingProcess(None,tmp_path/'onering') as policy:assert not getattr(policy,'scan',False)
    assert made==[('127.0.0.1:1','onering')]


def test_scan_is_the_default_without_any_onering_endpoint(tmp_path,monkeypatch):
    monkeypatch.setenv('SENSOR_MODEL_ENDPOINTS',json.dumps({'qwen':'127.0.0.1:1'}))
    with OneRingProcess(None,tmp_path/'onering') as policy:
        assert policy.scan and policy.predict(SimpleNamespace(timestamp=0.),'Find a bowl')=='r'
