"""Compatibility imports for the observation-based M2T2 backend.

The oracle interfaces remain in interface.py and fixed_interface.py.
"""
from .m2t2_interface import (
    M2T2SkillInterface, NavigationState, NavigationOperations, SensorNavigation,
    SensorExploration, ExplorationRequired, SkillOutcome, NAVIGATION_SCHEMA,
    EXPLORATION_SCHEMA, SENSOR_CATALOG,
)

SensorSkillInterface = M2T2SkillInterface
