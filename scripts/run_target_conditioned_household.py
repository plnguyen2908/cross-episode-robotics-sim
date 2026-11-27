"""Run any household recipe through the Qwen skill facade from initial state.

Same runner as scripts/run_sensor_household.py (identical arguments, including the memory flags) with the
target-conditioned pipeline: set-table bowl picks use scripts/execute_target_pick.py, everything else unchanged.
"""
import importlib.util
from pathlib import Path

from cross_episode_sim.evaluation.target_conditioned_household import TargetConditionedHouseholdPipeline

spec = importlib.util.spec_from_file_location('run_sensor_household', Path(__file__).with_name('run_sensor_household.py'))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.SensorHouseholdPipeline = TargetConditionedHouseholdPipeline

if __name__ == '__main__':raise SystemExit(base.main())
