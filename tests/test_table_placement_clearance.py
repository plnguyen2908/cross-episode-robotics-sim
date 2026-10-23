from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cross_episode_sim.controller.manipulation import TableReorder


@pytest.mark.parametrize('height,move_count', [(.012, 1), (.12, 2)])
def test_placement_never_raises_above_validated_staging(height, move_count):
    c = SimpleNamespace(move=Mock(), mesh_contact_move=Mock())
    target = np.eye(4)
    target[2, 3] = 1.
    staging = target.copy()
    staging[2, 3] += height
    TableReorder.approach_table_placement(c, target, staging)
    assert c.move.call_count == move_count
    assert all(call.args[1][2, 3] <= staging[2, 3] for call in c.move.call_args_list)
    np.testing.assert_allclose(c.mesh_contact_move.call_args.args[1], target)


def test_blocked_staging_never_starts_contact_descent():
    c = SimpleNamespace(move=Mock(side_effect=RuntimeError('blocked')),
                        mesh_contact_move=Mock())
    with pytest.raises(RuntimeError, match='blocked'):
        TableReorder.approach_table_placement(c, np.eye(4), np.eye(4))
    c.mesh_contact_move.assert_not_called()
