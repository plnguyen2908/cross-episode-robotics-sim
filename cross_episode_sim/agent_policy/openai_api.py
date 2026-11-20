"""GPT policy over the OpenAI Responses API with an API key; no Codex app server.

One persistent thread per episode and one tool call per decision, with the same
tools, prompt and observations as the Codex clients. This is a cross-sim
adaptation: the upstream GPT-Policy-Eval harness is not in the loop.
"""
import base64
import difflib
import json
import os
import time
from pathlib import Path
from .interface import INSTRUCTIONS, function_tools

# The Codex clients ask for a JSON decision; here the tool call is the decision.
API_INSTRUCTIONS=INSTRUCTIONS.replace('Return the requested JSON schema.','Answer every turn with exactly one call to one of the provided tools.')
IGNORED=json.dumps({'ok':False,'error':'Ignored: only the first tool call of a decision is executed'})


class ModelServiceError(RuntimeError):
    """The API could not serve the episode (key, quota, model ID, outage); not a policy result."""


class RequestTimeout(RuntimeError):
    """One model request exceeded the per-request limit."""


class WallTimeLimit(RuntimeError):
    """The episode's wall-clock budget ended before the model answered."""


def make_client():
    from openai import OpenAI
    if not os.environ.get('OPENAI_API_KEY'):raise ModelServiceError('OPENAI_API_KEY is not set')
    return OpenAI(max_retries=0)


def check_model(model,client=None):
    """Fail before any simulation when the key or the model ID is wrong. Returns the ID, or None if the key may not read models."""
    import openai
    client=client or make_client()
    try:return client.models.retrieve(model).id
    except openai.PermissionDeniedError:return None
    except openai.NotFoundError:
        try:ids=sorted(m.id for m in client.models.list())
        except openai.OpenAIError:ids=[]
        words=[w for w in model.replace('.','-').split('-') if w.isalpha() and w!='gpt']
        near=sorted(set(difflib.get_close_matches(model,ids,n=8,cutoff=.5))|{i for i in ids if any(w in i for w in words)})
        raise ModelServiceError(f'Model {model!r} is not available to this API key. Similar IDs: {", ".join(near) or "none found"}') from None
    except openai.OpenAIError as exc:raise ModelServiceError(f'{type(exc).__name__}: {exc}') from exc


def usage_dict(usage):
    def number(owner,key):return int(getattr(owner,key,None) or 0)
    return {'input_tokens':number(usage,'input_tokens'),
            'cached_input_tokens':number(getattr(usage,'input_tokens_details',None),'cached_tokens'),
            'output_tokens':number(usage,'output_tokens'),
            'reasoning_tokens':number(getattr(usage,'output_tokens_details',None),'reasoning_tokens'),
            'total_tokens':number(usage,'total_tokens'),'requests':1}


class ResponsesBackend:
    def __init__(self,model,effort,memory,timeout=240,stateless=False,retries=6,client=None,sleep=time.sleep):
        self.model,self.effort,self.timeout,self.stateless,self.retries=model,effort,timeout,stateless,retries
        self.client=client or make_client()
        self.tools=function_tools(memory)
        self.sleep=sleep
        self.previous=None   # last response ID: the thread lives on the server
        self.pending=[]      # call IDs still owed a tool result
        self.thread=[]       # whole input so far; only kept when stateless
        self.last_usage=None;self.last_response=None;self.retry_wait=0.

    def decide(self,observation,images,deadline=None):
        content=[{'type':'input_text','text':json.dumps(observation,allow_nan=False)}]
        for p in images:
            content.append({'type':'input_text','text':'Image '+Path(p).name})
            content.append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()})
        feedback=json.dumps(observation.get('last_tool_result'),allow_nan=False)
        items=[{'type':'function_call_output','call_id':call,'output':IGNORED if i else feedback} for i,call in enumerate(self.pending)]
        items.append({'role':'user','content':content})
        payload={'model':self.model,'instructions':API_INSTRUCTIONS,'tools':self.tools,'tool_choice':'required',
                 'parallel_tool_calls':False,'truncation':'auto'}
        if self.effort not in ('','default'):payload['reasoning']={'effort':self.effort}
        if self.stateless:payload.update(input=self.thread+items,store=False,include=['reasoning.encrypted_content'])
        else:
            payload['input']=items
            if self.previous:payload['previous_response_id']=self.previous
        self.last_usage=None
        for _ in range(2):  # a response without a tool call is asked for once more
            response=self.request(payload,deadline)
            used=usage_dict(getattr(response,'usage',None))
            self.last_usage=used if self.last_usage is None else {k:self.last_usage[k]+v for k,v in used.items()}
            calls=[x for x in response.output if x.type=='function_call']
            if calls:break
        else:
            raise RuntimeError(f'Model returned no tool call (status={getattr(response,"status",None)}, incomplete={getattr(response,"incomplete_details",None)})')
        self.previous=response.id;self.pending=[x.call_id for x in calls]
        self.last_response={'id':response.id,'model':getattr(response,'model',None)}
        if self.stateless:self.thread+=items+[x.model_dump(exclude_none=True) for x in response.output]
        try:arguments=json.loads(calls[0].arguments)
        except json.JSONDecodeError:arguments=calls[0].arguments
        return {'name':calls[0].name,'arguments':arguments}

    def request(self,payload,deadline):
        import openai
        attempt=0
        while True:
            budget=self.timeout if deadline is None else min(self.timeout,deadline-time.monotonic())
            if budget<=0:raise WallTimeLimit('Wall-time limit reached before the model request')
            try:return self.client.responses.create(**payload,timeout=budget)
            except openai.APITimeoutError as exc:
                if budget<self.timeout:raise WallTimeLimit('Wall-time limit reached while waiting for the model') from exc
                raise RequestTimeout(f'No model response within {budget:g}s') from exc
            except (openai.RateLimitError,openai.APIConnectionError,openai.InternalServerError,openai.ConflictError) as exc:
                attempt+=1
                if getattr(exc,'code',None)=='insufficient_quota' or attempt>self.retries:
                    raise ModelServiceError(f'{type(exc).__name__}: {exc}') from exc
                pause=min(60.,2.**attempt)
                if deadline is not None:pause=max(0.,min(pause,deadline-time.monotonic()))
                self.retry_wait+=pause;self.sleep(pause)
            except openai.APIError as exc:
                raise ModelServiceError(f'{type(exc).__name__}: {exc}') from exc

    def close(self):
        if hasattr(self.client,'close'):self.client.close()
