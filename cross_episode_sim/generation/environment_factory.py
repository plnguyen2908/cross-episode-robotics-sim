"""Datagen factory and inexpensive prerequisite checks for the shared scene."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from cross_episode_sim.paths import GRASP_SWEEP_DIR


class CompositeEnvironment(Protocol):
    interface: object
    def configure(self, scene_params, task_specification): ...
    def snapshot(self) -> dict: ...
    def evaluation_snapshot(self) -> dict: ...
    def verify_setup(self, config) -> dict: ...
    def save_initial_state(self, path): ...
    def render(self, output, speedup=1.0): ...
    def close(self): ...


def native_python():
    configured = os.environ.get('CES_ROBOCASA_PYTHON')
    if configured:
        return configured
    if importlib.util.find_spec('robosuite') is not None:
        return sys.executable
    sibling = Path(sys.executable).parent.parent.parent / 'robocasa/bin/python'
    return str(sibling) if sibling.is_file() else sys.executable


CAPABILITIES = {
    'multi_object_transfer', 'setting_regions', 'cabinet_interior_placement',
    'drawer_interior_placement', 'shared_fixture_manipulation', 'fridge_retrieval',
    'microwave_loading', 'microwave_controls', 'private_duration_tracking',
    'basket_containment', 'loaded_basket_transport', 'pot_grasps', 'independent_burner_controls',
}


def preflight(config):
    missing = [r for r in config['requirements'] if r not in CAPABILITIES]
    if not (GRASP_SWEEP_DIR/'current_results.json').is_file():
        missing.append('installed qualified object data: '+str(GRASP_SWEEP_DIR))
    if not shutil.which('ffprobe'):
        missing.append('ffprobe')
    try:
        result = subprocess.run([native_python(), '-c',
            'import importlib.util; assert importlib.util.find_spec("robosuite"); assert importlib.util.find_spec("robocasa")'],
            capture_output=True, timeout=10)
        if result.returncode:
            missing.append('RoboCasa Python; set CES_ROBOCASA_PYTHON')
    except (OSError, subprocess.TimeoutExpired):
        missing.append('RoboCasa Python; set CES_ROBOCASA_PYTHON')
    return {'ready': not missing, 'task_class_implemented': True, 'missing': missing,
            'physical_validation': 'must be established by rollout', 'native_python': native_python()}


def create_environment(config, output):
    status = preflight(config)
    if not status['ready']:
        raise RuntimeError('Physical preflight failed: '+', '.join(status['missing']))
    from cross_episode_sim.env.household import PhysicalCompositeEnvironment
    return PhysicalCompositeEnvironment(config, output)
