"""CaP-X execution over an observation-only API, without simulator access.

Imports/attribute introspection are intentionally unavailable to policy cells.
This constrained Python subset is an API boundary, not a general Python sandbox.
"""
import ast
import os
import re
from pathlib import Path
import sys


def load_capx():
    root=Path(__file__).resolve().parents[2]/'external/cap-x'
    os.environ['CAPX_EXTERNAL_ONLY']='1'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from capx.envs.tasks.base import CodeExecutionEnvBase,CodeExecEnvConfig
    return CodeExecutionEnvBase,CodeExecEnvConfig


def extract_code(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    if not blocks:raise ValueError('Return one fenced Python block')
    return '\n'.join(blocks)


def validate_code(code, functions):
    tree=ast.parse(code)
    allowed_calls=set(functions)|{'print','len','range','min','max','abs','round','enumerate','sorted','list','dict','float','int','bool','str'}
    forbidden=(ast.Import,ast.ImportFrom,ast.Attribute,ast.While,ast.FunctionDef,
               ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda,ast.With,ast.AsyncWith,
               ast.Global,ast.Nonlocal)
    for node in ast.walk(tree):
        if isinstance(node,forbidden):raise ValueError(f'Unsupported policy syntax: {type(node).__name__}')
        if isinstance(node,ast.Name) and (node.id.startswith('_') or node.id in {'env','APIS','INPUTS','obs'}):
            raise ValueError('Use only the documented sensor/action functions')
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in allowed_calls):
            raise ValueError('Only documented API calls and basic Python builtins are allowed')
    return tree


def make_executor(functions, observation):
    base,config=load_capx()
    class PublicView:
        def get_observation(self):return observation()
        def compute_reward(self):return 0.  # Private scorer never enters model feedback.
        def task_completed(self):return False
    class PublicAPI:
        def functions(self):return functions
    class SensorCodeEnv(base):
        def __init__(self):
            super().__init__(config(low_level=PublicView(),apis=[],prompt='Sensor-only robot control',privileged=False,enable_render=False))
            self._apis={'robot':PublicAPI()}
            self._init_exec_globals()
        def _exec_user_code(self,code):
            validate_code(code,functions)
            return super()._exec_user_code(code)
    return SensorCodeEnv()


def validate_feedback_cell(code, functions):
    """One physical action per observation cycle for calibration experiments."""
    tree=validate_code(code,functions)
    read_only={'list_skills','get_robot_state','door_state','propose_grasps'}
    actions=set(functions)-read_only
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in actions]
    loops=(ast.For,ast.While,ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)
    if len(calls)>1 or any(isinstance(n,loops) for n in ast.walk(tree)):
        raise ValueError('Use exactly one action per cell, then inspect fresh images and feedback. Do not combine pick/redock/finish or loop over attempts.')
    return code
