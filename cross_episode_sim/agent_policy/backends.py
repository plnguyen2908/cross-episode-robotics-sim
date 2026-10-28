"""Use the pinned upstream model clients with a shared cross-sim tool contract.

These are cross-sim adaptations, not reproductions of the upstream benchmarks.
RoboICL's original dual-arm tensor/reference sampler is deliberately replaced.
"""
import base64
import json
import sys
from pathlib import Path
from .interface import INSTRUCTIONS, schema

ROOT=Path(__file__).resolve().parents[2]

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

    def decide(self,observation,images):
        text=json.dumps(observation,allow_nan=False)
        if self.client:
            from gpt_policy.input.manifest import ImagePart
            from gpt_policy.harness.waiting import monitor_health
            # Upstream decide expects name/arguments at the root, so request an
            # act wrapper around our discriminated decision schema.
            shape={'type':'object','properties':{'name':{'type':'string','enum':['act']},
                'arguments':schema(self.memory)},'required':['name','arguments'],'additionalProperties':False}
            import time
            from gpt_policy.harness.usage import collect_usage
            with collect_usage() as ledger:
                ledger.attach(self.output / "usage" if self.output else Path("runs/astra_usage"))
                with monitor_health(lambda:None,deadline=time.monotonic()+self.timeout):
                    result=self.client.decide(text,shape,content=tuple(ImagePart(Path(p),label=Path(p).name) for p in images))
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
