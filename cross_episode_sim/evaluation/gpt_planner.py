"""GPT as the zero-shot whole-plan / replan model (--planner gpt_zero_shot / gpt_replan).

Drop-in for the Qwen planner client used by qwen_zero_shot_plan.py: the same
prompts (instruction + skill catalog + planning-contract rules), the same gray
placeholder image, the same catalog and contract checks with up to 2 repair
requests, and the same replan feedback. Only the model differs.

OpenAI Responses API, one stateless request per plan or replan (store=False).
The key is read from OPENAI_API_KEY and never written anywhere. Settings:
PLANNER_MODEL (default gpt-6.1-sol), PLANNER_EFFORT (reasoning effort, default
high), PLANNER_TIMEOUT seconds per request (default 600). Each request's token
usage is kept on `last_usage` and recorded in the episode's qwen_plan.json, so
the batch's API cost can be totalled (scripts/planner_usage.py).
"""
import base64
import io
import os
import time

DEFAULT_MODEL = 'gpt-6.1-sol'


class PlannerServiceError(RuntimeError):
    """Bad key, unknown model, exhausted quota or repeated server errors: not a policy result."""


def usage_dict(usage):
    def number(owner, key):return int(getattr(owner, key, None) or 0)
    return {'input_tokens': number(usage, 'input_tokens'),
            'cached_input_tokens': number(getattr(usage, 'input_tokens_details', None), 'cached_tokens'),
            'output_tokens': number(usage, 'output_tokens'),
            'reasoning_tokens': number(getattr(usage, 'output_tokens_details', None), 'reasoning_tokens'),
            'total_tokens': number(usage, 'total_tokens')}


def png_data_url(image):
    from PIL import Image
    import numpy as np
    buffer = io.BytesIO();Image.fromarray(np.asarray(image, dtype='uint8')).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


class GPTPlannerClient:
    """`call(images=..., prompt=..., max_new_tokens=...) -> text`, like the Qwen gRPC client."""
    def __init__(self, model=None, effort=None, timeout=None, retries=6, client=None, sleep=time.sleep):
        self.model = model or os.environ.get('PLANNER_MODEL') or DEFAULT_MODEL
        self.effort = effort if effort is not None else os.environ.get('PLANNER_EFFORT', 'high')
        self.timeout = float(timeout or os.environ.get('PLANNER_TIMEOUT') or 600)
        self.retries, self.sleep = retries, sleep
        if client is None:
            if not os.environ.get('OPENAI_API_KEY'):raise PlannerServiceError('OPENAI_API_KEY is not set')
            from openai import OpenAI
            client = OpenAI(max_retries=0)
        self.client = client
        self.last_usage = None;self.retry_wait = 0.

    def call(self, images=(), prompt='', max_new_tokens=None, **_):
        import openai
        content = [{'type': 'input_text', 'text': prompt}]
        content += [{'type': 'input_image', 'image_url': png_data_url(image)} for image in images]
        payload = dict(model=self.model, input=[{'role': 'user', 'content': content}], store=False)
        if self.effort not in ('', 'default'):payload['reasoning'] = {'effort': self.effort}
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                response = self.client.responses.create(**payload, timeout=self.timeout)
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError,
                    openai.ConflictError, openai.APITimeoutError) as exc:
                attempt += 1
                if getattr(exc, 'code', None) == 'insufficient_quota' or attempt > self.retries:
                    raise PlannerServiceError(f'{type(exc).__name__}: {exc}') from exc
                pause = min(60., 2. ** attempt);self.retry_wait += pause;self.sleep(pause)
            except openai.APIError as exc:
                raise PlannerServiceError(f'{type(exc).__name__}: {exc}') from exc
        self.last_usage = dict(usage_dict(getattr(response, 'usage', None)), model=getattr(response, 'model', self.model),
                               effort=self.effort, wall_seconds=round(time.monotonic() - started, 2))
        text = getattr(response, 'output_text', None)
        if text is None:
            text = ''.join(part.text for item in response.output if getattr(item, 'type', '') == 'message'
                           for part in item.content if getattr(part, 'type', '') == 'output_text')
        return text

    def close(self):
        if hasattr(self.client, 'close'):self.client.close()


def check_planner_model(model=None):
    """Fail before any simulation when the key or the model ID is wrong."""
    from ..agent_policy.openai_api import check_model, ModelServiceError
    try:return check_model(model or os.environ.get('PLANNER_MODEL') or DEFAULT_MODEL)
    except ModelServiceError as exc:raise PlannerServiceError(str(exc)) from exc
