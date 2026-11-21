import numpy as np
from cross_episode_sim.navigation import route_memory as rm


def poses(points):
    out = []
    for x, y in points:
        T = np.eye(4);T[0, 3], T[1, 3] = x, y;out.append(T)
    return out


def memory_with_objects():
    m = rm.RouteMemory()
    # observed counter / table from the 5-room grocery re-run (grocery-5r-000021)
    m.station_objects = {'kitchen': np.array([-0.91, -2.85]), 'dining table': np.array([-5.23, 13.52])}
    return m


def test_qwen_planner_drops_a_kitchen_label_at_the_dining_table(monkeypatch):
    monkeypatch.setattr(rm, 'labels_checked', lambda: True)
    m = memory_with_objects()
    # a carry that "arrived in the kitchen" while still at the dining table
    m.record(poses([(-5.83, 12.47), (-4.02, 12.62)]), np.eye(4), 'dining table', 'kitchen')
    assert 'kitchen' not in m.stations and len(m.stations['dining table']) == 1
    assert m.rejected_labels[0]['label'] == 'kitchen' and m.routes[-1][1] is None
    # a real trip keeps both labels
    m.record(poses([(-5.8, 12.4), (-1.5, -1.6)]), np.eye(4), 'dining table', 'kitchen')
    assert len(m.stations['kitchen']) == 1
    # a third-room endpoint is not a station
    m.record(poses([(-1.5, -1.6), (6.0, 4.0)]), np.eye(4), 'kitchen', 'dining table')
    assert len(m.stations['dining table']) == 2  # only the earlier two starts
    # fixture docks are unaffected
    m.record_dock(poses([(-1.0, -1.0), (-1.1, -1.0)]), np.eye(4), 'fridge')
    assert m.stations['fridge'][-1]['kind'] == 'successful_dock'


def test_recipe_keeps_every_label(monkeypatch):
    monkeypatch.setattr(rm, 'labels_checked', lambda: False)
    m = memory_with_objects()
    m.record(poses([(-5.83, 12.47), (-4.02, 12.62)]), np.eye(4), 'dining table', 'kitchen')
    assert 'kitchen' in m.stations and not m.rejected_labels
