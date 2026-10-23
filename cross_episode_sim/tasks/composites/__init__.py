"""Concrete composite tasks; no external task-backend module is needed."""
from .base import HouseholdCompositeTask as HouseholdCompositeTask
from .clear_and_store import ClearAndStore
from .prepare_meal import PrepareAndServeMeal
from .set_table import SetTable
from .sort_groceries import UnpackAndSortGroceries
from .warm_pots import WarmTwoPots

TASK_CLASSES = {cls.task_name: cls for cls in (
    SetTable, ClearAndStore, PrepareAndServeMeal, UnpackAndSortGroceries, WarmTwoPots)}


def create_task(config, output=None, interface=None):
    return TASK_CLASSES[config['task']](config, output, interface)
