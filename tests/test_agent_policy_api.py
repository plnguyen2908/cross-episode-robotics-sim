"""Direct OpenAI Responses API backend and shared-output bookkeeping, with a fake client (no network)."""
import json
import time
from types import SimpleNamespace
import pytest
from cross_episode_sim.agent_policy import batch
from cross_episode_sim.agent_policy.interface import TOOLS,function_tools,problem
from cross_episode_sim.agent_policy.openai_api import (API_INSTRUCTIONS,ModelServiceError,RequestTimeout,
    ResponsesBackend,WallTimeLimit,check_model)

openai=pytest.importorskip('openai')


def error(kind,code=None):
    exc=kind.__new__(kind);exc.args=(kind.__name__,);exc.code=code
    return exc


def reply(n,name='wait',arguments='{"seconds": 1}',calls=1):
    output=[SimpleNamespace(type='reasoning',id=f'rs{n}')]
    output+=[SimpleNamespace(type='function_call',name=name,arguments=arguments,call_id=f'call{n}_{i}') for i in range(calls)]
    usage=SimpleNamespace(input_tokens=100,output_tokens=10,total_tokens=110,
        input_tokens_details=SimpleNamespace(cached_tokens=60),output_tokens_details=SimpleNamespace(reasoning_tokens=4))
    return SimpleNamespace(id=f'resp{n}',model='served-model',status='completed',output=output,usage=usage)


class Client:
    def __init__(self,script):
        self.script=list(script);self.requests=[];self.responses=SimpleNamespace(create=self.create)
    def create(self,**request):
        self.requests.append(request);step=self.script.pop(0)
        if isinstance(step,Exception):raise step
        return step


def backend(script,**options):
    client=Client(script);waits=[]
    return ResponsesBackend('m','high','none',client=client,sleep=waits.append,**options),client,waits


def image(tmp_path):
    path=tmp_path/'0000_navigation.png';path.write_bytes(b'png');return path


def test_tools_match_the_shared_contract():
    assert [t['name'] for t in function_tools('all')]==list(TOOLS)
    assert 'history_frame' not in [t['name'] for t in function_tools('none')]
    assert 'exactly one call' in API_INSTRUCTIONS and 'JSON schema' not in API_INSTRUCTIONS


def test_thread_chains_stored_responses(tmp_path):
    b,client,_=backend([reply(0),reply(1,'finish','{}')])
    assert b.decide({'task':'t','last_tool_result':{'ok':True}},[image(tmp_path)])=={'name':'wait','arguments':{'seconds':1}}
    assert b.decide({'task':'t','last_tool_result':{'ok':True,'executed_seconds':1}},[])=={'name':'finish','arguments':{}}
    first,second=client.requests
    assert 'previous_response_id' not in first and first['tool_choice']=='required' and first['parallel_tool_calls'] is False
    assert first['reasoning']=={'effort':'high'} and first['timeout']==240
    assert [part['type'] for part in first['input'][0]['content']]==['input_text','input_text','input_image']
    assert second['previous_response_id']=='resp0'
    assert second['input'][0]=={'type':'function_call_output','call_id':'call0_0','output':json.dumps({'ok':True,'executed_seconds':1})}
    assert b.last_usage=={'input_tokens':100,'cached_input_tokens':60,'output_tokens':10,'reasoning_tokens':4,'total_tokens':110,'requests':1}
    assert b.last_response=={'id':'resp1','model':'served-model'}


def test_stateless_resends_the_thread():
    replies=[reply(0),reply(1)]
    for item in replies[0].output+replies[1].output:item.model_dump=lambda exclude_none,item=item:{'type':item.type}
    b,client,_=backend(replies,stateless=True)
    b.decide({'last_tool_result':None},[]);b.decide({'last_tool_result':{'ok':True}},[])
    second=client.requests[1]
    assert second['store'] is False and 'previous_response_id' not in second
    assert [x.get('type',x.get('role')) for x in second['input']]==['user','reasoning','function_call','function_call_output','user']


def test_extra_parallel_calls_are_answered_but_not_executed():
    b,client,_=backend([reply(0,calls=2),reply(1)])
    b.decide({'last_tool_result':None},[]);b.decide({'last_tool_result':{'ok':True}},[])
    outputs=[x for x in client.requests[1]['input'] if x.get('type')=='function_call_output']
    assert [x['call_id'] for x in outputs]==['call0_0','call0_1'] and 'Ignored' in outputs[1]['output']


def test_unparsable_arguments_reach_validation():
    b,_,_=backend([reply(0,arguments='{oops')])
    decision=b.decide({'last_tool_result':None},[])
    assert decision=={'name':'wait','arguments':'{oops'} and 'arguments' in problem(decision,'none')


def test_invalid_argument_is_named():
    assert problem({'name':'wait','arguments':{'seconds':50}},'none').startswith('seconds: 50 is greater than')
    assert problem({'name':'pick','arguments':{}},'none').startswith('Unknown tool')
    assert problem({'name':'history_frame','arguments':{}},'none').startswith('Unknown tool')


def test_transient_errors_are_retried_then_reported_as_service_errors():
    b,client,waits=backend([error(openai.RateLimitError),error(openai.InternalServerError),reply(0)])
    assert b.decide({'last_tool_result':None},[])['name']=='wait'
    assert waits==[2.,4.] and b.retry_wait==6.
    b,client,waits=backend([error(openai.APIConnectionError)]*3,retries=2)
    with pytest.raises(ModelServiceError):b.decide({'last_tool_result':None},[])
    assert len(client.requests)==3


@pytest.mark.parametrize('exc',[error(openai.RateLimitError,'insufficient_quota'),error(openai.AuthenticationError),
    error(openai.NotFoundError),error(openai.BadRequestError)])
def test_key_quota_and_model_errors_are_not_retried(exc):
    b,client,waits=backend([exc,reply(0)])
    with pytest.raises(ModelServiceError):b.decide({'last_tool_result':None},[])
    assert len(client.requests)==1 and not waits


def test_timeouts_and_wall_limit():
    b,_,_=backend([error(openai.APITimeoutError)])
    with pytest.raises(RequestTimeout):b.decide({'last_tool_result':None},[])
    b,client,_=backend([error(openai.APITimeoutError)])
    with pytest.raises(WallTimeLimit):b.decide({'last_tool_result':None},[],deadline=time.monotonic()+30)
    assert 0<client.requests[0]['timeout']<=30
    b,client,_=backend([reply(0)])
    with pytest.raises(WallTimeLimit):b.decide({'last_tool_result':None},[],deadline=time.monotonic()-1)
    assert not client.requests


def test_missing_tool_call_is_requested_once_more():
    empty=reply(0);empty.output=empty.output[:1]
    b,client,_=backend([empty,reply(1)])
    assert b.decide({'last_tool_result':None},[])['name']=='wait'
    assert b.last_usage['requests']==2 and b.last_usage['input_tokens']==200
    b,_,_=backend([empty,empty])
    with pytest.raises(RuntimeError,match='no tool call'):b.decide({'last_tool_result':None},[])


def test_check_model_lists_similar_ids():
    def retrieve(model):
        if model!='gpt-6.1-sol-preview':raise error(openai.NotFoundError)
        return SimpleNamespace(id=model)
    client=SimpleNamespace(models=SimpleNamespace(retrieve=retrieve,list=lambda:[SimpleNamespace(id=i) for i in ('gpt-6.1-sol-preview','gpt-6-astra','other')]))
    assert check_model('gpt-6.1-sol-preview',client)=='gpt-6.1-sol-preview'
    with pytest.raises(ModelServiceError,match='gpt-6.1-sol-preview'):check_model('gpt-6.1-sol',client)


def result(folder,termination,success=False,task='set_table',rooms=2):
    batch.write_json(folder/'result.json',{'success':success,'termination':termination,'task':task,'room_count':rooms,
        'usage':{'input_tokens':10}})


def test_claims_split_one_output_between_workers(tmp_path):
    done,failed,dead,held=(tmp_path/f'000{i}_x' for i in range(4))
    assert batch.claim(done) and batch.claim(failed) and batch.claim(dead) and batch.claim(held)
    with pytest.raises(FileExistsError):batch.claim(done)
    result(done,'call_limit');result(failed,batch.INVALID)
    owner=json.loads((dead/'claim.json').read_text())
    batch.write_json(dead/'claim.json',{**owner,'pid':2**22+12345})
    assert not batch.claim(done,resume=True)       # scored result is kept
    assert not batch.claim(held,resume=True)       # live owner
    assert batch.claim(failed,resume=True) and not (failed/'result.json').exists()   # service error is rerun
    assert batch.claim(dead,resume=True)           # owner died
    assert len(list((tmp_path/'_stale').iterdir()))==2


def test_summary_excludes_service_errors(tmp_path):
    for i,(termination,success,task,rooms) in enumerate([('success',True,'set_table',2),('call_limit',False,'set_table',3),
            ('wall_time_limit',False,'warm_two_pots',2),(batch.INVALID,False,'warm_two_pots',2)]):
        folder=tmp_path/f'{i:04d}_x';folder.mkdir();result(folder,termination,success,task,rooms)
    s=batch.summarize(tmp_path,6)
    assert (s['episodes'],s['successes'],s['pending'],s['service_errors'])==(3,1,3,1)
    assert s['by_task']['set_table']=={'episodes':2,'successes':1,'success_rate':.5}
    assert s['by_room_count']['2']['episodes']==2 and s['usage']=={'input_tokens':40}
    assert json.loads((tmp_path/'summary.json').read_text())['terminations'][batch.INVALID]==1


def codex_backend(outcome,timeout=240):
    """The gpt-policy-eval backend around a fake Codex client (no app server)."""
    import sys
    from cross_episode_sim.agent_policy import backends
    src=str(backends.ROOT/'external/gpt-policy-eval/src')
    if src not in sys.path:sys.path.insert(0,src)
    pytest.importorskip('gpt_policy.harness.errors')
    b=backends.Backend.__new__(backends.Backend)
    b.framework,b.model,b.effort,b.memory,b.timeout,b.output,b.last_usage='gpt-policy-eval','m','high','none',timeout,None,None
    def decide(text,shape,content=()):
        from gpt_policy.harness.waiting import receive_event
        if outcome=='hang':receive_event(__import__('queue').Queue())
        if isinstance(outcome,Exception):raise outcome
        return outcome
    b.client=SimpleNamespace(decide=decide)
    return b


def test_codex_backend_reports_key_and_quota_failures_as_service_errors(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    decision={'name':'wait','arguments':{'seconds':1}}
    assert codex_backend({'name':'act','arguments':{'decision':decision}}).decide({'task':'t'},[])==decision
    from gpt_policy.harness.codex_errors import codex_failure
    rejected=codex_failure({'message':'unexpected status 401 Unauthorized','codexErrorInfo':{'httpConnectionFailed':{'httpStatusCode':401}}})
    quota=codex_failure({'message':'limit','codexErrorInfo':'usageLimitExceeded'})
    for exc in (rejected,quota):
        with pytest.raises(ModelServiceError):codex_backend(exc).decide({'task':'t'},[])
    with pytest.raises(RuntimeError) as other:codex_backend(RuntimeError('Codex turn failed: bad schema')).decide({'task':'t'},[])
    assert not isinstance(other.value,ModelServiceError)


def test_codex_backend_stops_waiting_at_the_wall_limit(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WallTimeLimit):codex_backend('hang').decide({'task':'t'},[],deadline=time.monotonic()+.3)
    with pytest.raises(WallTimeLimit):codex_backend('hang').decide({'task':'t'},[],deadline=time.monotonic()-1)
    with pytest.raises(RequestTimeout):codex_backend('hang',timeout=.3).decide({'task':'t'},[],deadline=time.monotonic()+60)
