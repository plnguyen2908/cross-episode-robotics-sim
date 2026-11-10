import pytest
from cross_episode_sim.code_policy.capx import extract_code, validate_code, make_executor


@pytest.mark.parametrize('code', ['import os', 'env.model', 'open("file")',
    'act.__globals__', '__import__("os")', 'while True: pass'])
def test_rejects_unexposed_access(code):
    with pytest.raises(ValueError):validate_code(code, {'act':lambda:None})


def test_real_executor_preserves_state_after_failure():
    calls=[]
    def act(value):
        calls.append(value)
        if value==2:raise ValueError('Motion failed')
    executor=make_executor({'act':act},lambda:{})
    assert executor.step('x=1\nact(x)')[-1]['sandbox_rc']==0
    assert executor.step('act(x+1)')[-1]['sandbox_rc']==1
    assert executor.step('act(x+2)')[-1]['sandbox_rc']==0
    assert calls==[1,2,3]


def test_requires_fenced_code():
    assert extract_code('```python\nprint(1)\n```').strip()=='print(1)'
    with pytest.raises(ValueError):extract_code('I would pick up the bowl.')
