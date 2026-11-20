"""Use the pinned upstream model clients with a shared cross-sim tool contract.

These are cross-sim adaptations, not reproductions of the upstream benchmarks.
RoboICL's original dual-arm tensor/reference sampler is deliberately replaced.
"""
import base64
import json
import re
import sys
import time
from pathlib import Path
from .interface import INSTRUCTIONS, schema
from .openai_api import ModelServiceError, RequestTimeout, WallTimeLimit

ROOT=Path(__file__).resolve().parents[2]

def service_failure(exc):
    """Codex could not serve the turn (key, quota, overload); not a policy result."""
    from gpt_policy.harness.errors import AgentOverloadedError,AgentRecoverableError,AgentUsageLimitError
    if isinstance(exc,(AgentUsageLimitError,AgentOverloadedError,AgentRecoverableError)):return True
    status=re.search(r"'httpStatusCode': (\d+)",str(exc))
    return str(exc).startswith('Codex ') and ("'unauthorized'" in str(exc) or bool(status and int(status.group(1)) in (401,402,403,404,429)))


class Backend:
    def __init__(self,framework,model,effort,memory,timeout=240,output=None):
        self.framework,self.model,self.effort,self.memory=framework,model,effort,memory
        self.timeout=timeout
        self.messages=[]
        self.client=None
        self.last_usage=None
        self.output=Path(output) if output else None
        if framework=='gpt-policy-eval':
            sys.path.insert(0,str(ROOT/'external/gpt-policy-eval/src'))
            from gpt_policy.harness.codex import CodexAppServer
            self.client=CodexAppServer(model,effort=effort,decision_only=True)
            self.client.start_thread(INSTRUCTIONS)
        elif framework=='roboicl':
            sys.path.insert(0,str(ROOT/'external/roboicl'))
            from roboicl.policy.codex_transport import request
            self.request=request
        else:raise ValueError('Unknown framework')

    def decide(self,observation,images,deadline=None):
        # deadline (episode wall limit) bounds the Codex app-server wait; the RoboICL transport only has the request timeout.
        text=json.dumps(observation,allow_nan=False)
        if self.client:
            from gpt_policy.input.manifest import ImagePart
            from gpt_policy.harness.errors import AgentTimeoutError
            from gpt_policy.harness.waiting import monitor_health
            limit=time.monotonic()+self.timeout;walled=deadline is not None and deadline<limit
            if walled and deadline<=time.monotonic():raise WallTimeLimit('Wall-time limit reached before the model request')
            # Upstream decide expects name/arguments at the root, so request an
            # act wrapper around our discriminated decision schema.
            shape={'type':'object','properties':{'name':{'type':'string','enum':['act']},
                'arguments':schema(self.memory)},'required':['name','arguments'],'additionalProperties':False}
            from gpt_policy.harness.usage import collect_usage
            with collect_usage() as ledger:
                ledger.attach(self.output / "usage" if self.output else Path("runs/astra_usage"))
                try:
                    with monitor_health(lambda:None,deadline=deadline if walled else limit):
                        result=self.client.decide(text,shape,content=tuple(ImagePart(Path(p),label=Path(p).name) for p in images))
                except AgentTimeoutError as exc:
                    if walled:raise WallTimeLimit('Wall-time limit reached while waiting for the model') from exc
                    raise RequestTimeout(f'No model response within {self.timeout:g}s') from exc
                except RuntimeError as exc:
                    if service_failure(exc):raise ModelServiceError(str(exc)) from exc
                    raise
                self.last_usage=ledger.calls[-1]["usage"] if ledger.calls else None
            return result['arguments']['decision']
        content=[{'type':'input_text','text':text}]
        for p in images:
            content.append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()})
        self.messages.append({'role':'user','content':content})
        response=self.request({'model':self.model,'instructions':INSTRUCTIONS,
            'reasoning':{'effort':self.effort},'input':self.messages,
            'tools':[{'type':'function','name':'act','parameters':schema(self.memory)}]},timeout=self.timeout)
        args=response['output'][0]['arguments']
        self.messages.append({'role':'assistant','content':[{'type':'output_text','text':args}]})
        self.last_usage=response['usage']
        return json.loads(args)['decision']

    def close(self):
        if self.client:self.client.close()
