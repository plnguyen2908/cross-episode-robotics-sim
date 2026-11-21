import numpy as np
from cross_episode_sim.navigation import station_locator as sl


def located(monkeypatch, costs):
    monkeypatch.setattr(sl, "room_station", lambda grid, xy, memory: (None, {}))
    monkeypatch.setattr(sl, 'station_path_costs', lambda policy, xy, memory: dict(costs))
    class P:grid = None
    return sl.current_station(P(), np.zeros(2), {})[0]


def test_at_a_station_only_when_close_along_the_map(monkeypatch):
    assert located(monkeypatch, {'kitchen': 0.4, 'dining table': 7.0}) == 'kitchen'
    assert located(monkeypatch, {'kitchen': 7.2, 'dining table': 1.1}) == 'dining table'
    # third room / hallway: nearer to the kitchen, but not at it
    assert located(monkeypatch, {'kitchen': 4.5, 'dining table': 9.0}) is None
    # ambiguous middle
    assert located(monkeypatch, {'kitchen': 2.5, 'dining table': 3.0}) is None


def test_straight_line_only_without_any_map_path(monkeypatch):
    monkeypatch.setattr(sl, 'station_path_costs', lambda *a: {'kitchen': np.inf, 'dining table': np.inf})
    monkeypatch.setattr(sl, 'straight_costs', lambda xy, memory: {'kitchen': 1.0, 'dining table': 6.5})
    station, evidence = sl.current_station(None, np.zeros(2), {})
    assert station == 'kitchen' and evidence['basis'] == 'straight_line_no_map_path'


def test_room_footprint_decides_first(monkeypatch):
    monkeypatch.setattr(sl, "USE_ROOM_FOOTPRINTS", True)
    monkeypatch.setattr(sl, 'room_station', lambda grid, xy, memory: ('kitchen', {'kitchen': 48.8, 'dining table': 21.5}))
    monkeypatch.setattr(sl, 'station_path_costs', lambda *a: {'kitchen': 9.0, 'dining table': 9.0})
    class P:grid = None
    station, evidence = sl.current_station(P(), np.zeros(2), {})
    assert station == 'kitchen' and evidence['basis'] == 'observed_room_footprint'
    monkeypatch.setattr(sl, 'room_station', lambda grid, xy, memory: (None, {}))
    assert sl.current_station(P(), np.zeros(2), {})[0] is None   # not in a room, far from both: unknown
