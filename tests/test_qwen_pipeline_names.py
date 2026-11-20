"""The opening-view fallback binds numpy before using it (fix152 call 18: UnboundLocalError on np).

QwenPipelineRedock.__call__ imports numpy locally on some paths, which makes `np` a local
name for the whole method; every block that uses it must import it first.
"""
import inspect
import cross_episode_sim.runtime.qwen_pipeline as qp


def test_opening_view_block_imports_numpy_before_use():
    lines = inspect.getsource(qp).splitlines()
    start = next(i for i, l in enumerate(lines) if "leaf.get('opening_centre') is not None" in l)
    block = lines[start + 1:start + 12]
    first_use = next(i for i, l in enumerate(block) if 'np.' in l)
    assert any('import numpy as np' in l for l in block[:first_use])
