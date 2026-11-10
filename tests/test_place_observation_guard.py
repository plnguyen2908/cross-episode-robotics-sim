"""Exercise the script's authorization guard without loading GPU models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('observe,transit,docked,rejected',[
    (True,True,False,False),
    (False,True,False,True),
    (True,False,False,True),
    (False,False,True,False),
])
def test_transit_authorizes_observation_only(observe,transit,docked,rejected):
    script=Path(__file__).resolve().parents[1]/'scripts/test_m2t2_place.py'
    tree=ast.parse(script.read_text())
    guard=next(node for node in tree.body if isinstance(node,ast.If))
    condition=compile(ast.Expression(guard.test),str(script),'eval')
    args=SimpleNamespace(observe_only=observe,isolated_initial_state=False)
    report=dict(transit_success=transit,docking_success=docked)
    assert bool(eval(condition,{'a':args,'r':report}))==rejected
