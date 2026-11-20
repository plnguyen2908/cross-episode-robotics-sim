"""Private progress reporting with no simulator or paid model requests."""
import json
from types import SimpleNamespace

from cross_episode_sim.agent_policy import batch


def test_progress_summary_excludes_service_errors(tmp_path):
    for i,(termination,rate) in enumerate([('call_limit',.25),('success',1.),(batch.INVALID,.9)]):
        folder=tmp_path/f'{i:04d}_episode';folder.mkdir()
        batch.write_json(folder/'result.json',dict(termination=termination,success=termination=='success',
            task='set_table',room_count=2,task_progress=dict(done=int(rate*4),total=4,rate=rate)))
    summary=batch.summarize(tmp_path,3)
    assert summary['mean_task_progress']==.625
    assert summary['progress_episodes']==2
    assert summary['by_task']['set_table']['mean_task_progress']==.625
    assert summary['success_rate']==.5


def test_episode_records_progress_without_giving_it_to_policy(tmp_path,monkeypatch):
    from scripts.evaluate_astra import run_episode
    from cross_episode_sim.agent_policy import environment
    class FakeEnvironment:
        def __init__(self,*args):
            self.data=SimpleNamespace(time=0.);self.start=0.;self.steps=0;self.stable=0
            self.initial={};self.prompt='Bring two bowls and two cups to the table'
            self.scorer=SimpleNamespace(progress=lambda initial,data:dict(done=int(data.time>0),total=4,rate=.25 if data.time>0 else 0.))
        def observe(self,turn):return {'simulation_seconds':self.data.time},[]
        def goals(self):return {'complete':False}
        def act(self,name,args,deadline):
            self.data.time+=args['seconds'];self.steps+=10
            return {'ok':True,'executed_seconds':args['seconds']}
        def close(self):pass
    monkeypatch.setattr(environment,'Environment',FakeEnvironment)
    source=tmp_path/'source';source.mkdir();batch.write_json(source/'config.json',{'task':'set_table'})
    output=tmp_path/'output';output.mkdir()
    args=SimpleNamespace(framework='openai-api',model='gpt-6.2-sol',effort='high',memory='none',
        max_calls=2,seconds=3600,wall_seconds=14400,request_timeout=600,log_interval=15,
        controller_config=None,mock=True)
    result=run_episode(args,source,output,[],2)
    assert result['task_progress']['rate']==.25 and not result['success']
    trace=[json.loads(line) for line in (output/'progress_trace.jsonl').read_text().splitlines()]
    assert trace[0]['phase']=='initial' and trace[0]['task_progress']['rate']==0
    assert trace[-1]['phase']=='final' and trace[-1]['control_steps']==10
    assert result['first_success'] is None
    for line in (output/'exchanges.jsonl').read_text().splitlines():
        record=json.loads(line)
        assert 'task_progress' not in record['observation']
        assert 'goal_evidence' not in record['observation']
        assert 'task_progress' not in record['feedback']
