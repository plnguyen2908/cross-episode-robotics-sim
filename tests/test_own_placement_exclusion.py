"""A search ignores detections where this robot itself placed an object of the same description (fix168)."""
import json
import os
from types import SimpleNamespace

import numpy as np

from cross_episode_sim.perception.molmo_pointing import PointingSupervisor


def supervisor(label, positions):
    s = PointingSupervisor(None, None, label)
    s.targets.targets = [SimpleNamespace(position=np.asarray(p, float)) for p in positions]
    return s


def test_drops_target_at_own_placement_only(monkeypatch):
    label = 'cookware on kitchen counter'
    # fix168: release recorded at (1.723, 1.012); the search then found (1.77, 1.06).
    monkeypatch.setenv('HOUSEHOLD_PLACED_EXCLUSIONS', json.dumps({label: [[1.723, 1.012, .905]]}))
    s = supervisor(label, [[1.77, 1.06, .92], [0.30, 2.40, .93]])
    dropped = s.drop_own_placements()
    assert len(dropped) == 1 and np.allclose(dropped[0][:2], [1.77, 1.06])
    assert [list(t.position[:2]) for t in s.targets.targets] == [[0.30, 2.40]]


def test_other_descriptions_and_no_memory_unchanged(monkeypatch):
    monkeypatch.setenv('HOUSEHOLD_PLACED_EXCLUSIONS', json.dumps({'cookware on kitchen counter': [[1.7, 1.0, .9]]}))
    s = supervisor('cookware on front left stove burner', [[1.72, 1.01, .92]])
    assert s.drop_own_placements() == [] and len(s.targets.targets) == 1
    monkeypatch.delenv('HOUSEHOLD_PLACED_EXCLUSIONS')
    s = supervisor('cookware on kitchen counter', [[1.72, 1.01, .92]])
    assert s.drop_own_placements() == [] and len(s.targets.targets) == 1
