"""An aperture dock that leaves the arm start blocked is excluded and the base re-docks."""
import json
import numpy as np
import pytest
from cross_episode_sim.navigation import aperture_dock


def fake_docks(monkeypatch, goals):
    calls = []
    def once(runtime, aperture, tool_points, output, seconds, distances, multi_goal, offsets, support, excluded, first=True, side_docks=None):
        if not first:
            output.mkdir(parents=True)
        calls.append([e.tolist() for e in excluded])
        return dict(success=True, goal=goals[len(calls)-1], events=[])
    monkeypatch.setattr(aperture_dock, '_dock_once', once)
    return calls


def test_blocked_arm_start_excludes_the_goal_and_redocks(monkeypatch, tmp_path):
    calls = fake_docks(monkeypatch, [[0., 0., 1.], [.1, 0., 1.]])
    report = aperture_dock.dock_to_aperture(None, {}, None, tmp_path/'dock', arm_clear=lambda attempt: attempt == 1)
    assert report['goal'] == [.1, 0., 1.] and report['arm_start_blocked_docks'] == [[0., 0., 1.]]
    assert calls == [[], [[0., 0., 1.]]]
    assert json.loads((tmp_path/'dock/report.json').read_text())['goal'] == [.1, 0., 1.]


def test_every_blocked_dock_fails_clearly(monkeypatch, tmp_path):
    fake_docks(monkeypatch, [[0., 0., 1.], [.2, 0., 1.], [.4, 0., 1.], [.6, 0., 1.]])
    with pytest.raises(RuntimeError, match='arm start blocked'):
        aperture_dock.dock_to_aperture(None, {}, None, tmp_path/'dock', arm_clear=lambda attempt: False)


def test_without_a_check_one_dock_is_returned(monkeypatch, tmp_path):
    calls = fake_docks(monkeypatch, [[0., 0., 1.]])
    assert aperture_dock.dock_to_aperture(None, {}, None, tmp_path/'dock')['goal'] == [0., 0., 1.]
    assert len(calls) == 1


def test_closest_standoff_ring_is_planned_first(monkeypatch, tmp_path):
    """The multi-goal search runs per stand-off ring, closest first, and stops at the first reachable ring."""
    from types import SimpleNamespace as NS
    from cross_episode_sim.navigation import placement_docks
    rings = []
    def plan(dock, pose, q, candidates, max_nodes=4000):
        rings.append(sorted({round(float(g[0]), 3) for g in candidates}))
        if len(rings) < 2:
            return False
        dock.path = [np.r_[candidates[0][:2], candidates[0][2]]]
        return True
    monkeypatch.setattr(placement_docks, 'plan_to_any_dock', plan)
    class Stop(Exception):
        pass
    monkeypatch.setattr(aperture_dock, 'departure_route', lambda *a, **k: (_ for _ in ()).throw(Stop()))
    T = np.eye(4)
    camera = NS(rgb=None)
    packet = NS(cameras={}, arm=NS(position=np.zeros(7)), timestamp=0.)
    runtime = NS(model=None, data=NS(time=0.), observed_navigation_map=NS(),
                 mapper=NS(view=lambda: NS(update=lambda c, t: NS(accepted=True, T_map_base=T))),
                 sensors=NS(capture=lambda include_robot_mask=False: NS(cameras={'navigation': camera}, timestamp=0.)))
    monkeypatch.setattr(aperture_dock, 'RobotGeometry', lambda m: NS(boxes=lambda q: []))
    monkeypatch.setattr(aperture_dock, 'ObservedPayload', lambda g, p: NS(filter_packet=lambda raw: packet, box=lambda q: NS(center=np.zeros(3), axes=np.eye(3), half=np.zeros(3))))
    monkeypatch.setattr(aperture_dock, 'ObservedShapeChecker', lambda *a, **k: NS(clear=lambda pose: True, margin=.035))
    aperture = dict(closed_normal=[1., 0., 0.], closed_handle=[0., .2, 1.], hinge=[0., -.2, 1.])
    with pytest.raises(Stop):
        aperture_dock.dock_to_aperture(runtime, aperture, None, tmp_path/'dock', distances=(.5, .35, .4),
                                       offsets=(0., .1), multi_goal=True)
    assert rings == [[.35], [.4]]


def test_blocked_single_dock_widens_to_nearby_poses_away_from_the_open_door(monkeypatch, tmp_path):
    """The microwave asks for one centred pose; after it blocks the arm, nearby shifts are offered, away from the leaf first."""
    from types import SimpleNamespace as NS
    seen = []
    class Stop(Exception):
        pass
    class Dock:
        def __init__(self, *a, **k):
            self.events = []
        def plan_candidates(self, pose, q, candidates, source=''):
            seen.extend(candidates)
            raise Stop()
    monkeypatch.setattr(aperture_dock, 'ShapedDocking', Dock)
    T = np.eye(4)
    packet = NS(cameras={'navigation': NS()}, arm=NS(position=np.zeros(7)), timestamp=0.)
    runtime = NS(model=None, data=NS(time=0.), observed_navigation_map=NS(),
                 mapper=NS(view=lambda: NS(update=lambda c, t: NS(accepted=True, T_map_base=T))),
                 sensors=NS(capture=lambda include_robot_mask=False: NS(cameras={'navigation': NS()}, timestamp=0.)))
    monkeypatch.setattr(aperture_dock, 'RobotGeometry', lambda m: NS(boxes=lambda q: []))
    monkeypatch.setattr(aperture_dock, 'ObservedPayload', lambda g, p: NS(filter_packet=lambda raw: packet, box=lambda q: NS(center=np.zeros(3), axes=np.eye(3), half=np.zeros(3))))
    monkeypatch.setattr(aperture_dock, 'ObservedShapeChecker', lambda *a, **k: NS(clear=lambda pose: True, margin=.035))
    monkeypatch.setattr(aperture_dock, 'observed_open_panel_points', lambda pts, ap: np.empty((0, 3)))
    import cross_episode_sim.control.sensor_appliance_backend as backend
    monkeypatch.setattr(backend, 'camera_points', lambda cam: (np.empty((0, 3)), None, None))
    # Opening centred at y=0 facing +x; the open leaf swung to +y.
    aperture = dict(closed_normal=[1., 0., 0.], closed_handle=[0., .2, 1.], hinge=[0., -.2, 1.], open_handle=[.4, .25, 1.])
    centred = np.array([.6, 0., np.pi])
    with pytest.raises(Stop):
        aperture_dock._dock_once(runtime, aperture, None, tmp_path/'d', 60., (.6,), True, (0.,), False, [centred], first=True)
    offsets = [round(float(g[1]), 2) for g in seen]
    assert all(np.linalg.norm(g[:2]-centred[:2]) > .04 for g in seen)
    assert offsets[:3] == [-.06, -.12, -.18]          # away from the open leaf first
    assert .06 in offsets and .12 in offsets          # then toward it
    assert {round(float(g[0]), 2) for g in seen} == {.6, .55, .65}


def test_open_panel_points_follow_a_door_that_swung_from_where_it_was_opened():
    """Remembered open leaf at 90 degrees; the door later drifted to about 45 degrees."""
    aperture = dict(hinge=[0., 0., 1.], open_handle=[.5, 0., 1.], closed_handle=[0., .5, 1.], closed_normal=[1., 0., 0.])
    remembered_leaf = np.array([[.25, 0., 1.]])
    drifted_leaf = np.array([[.25, .25, 1.]])          # 45 degrees, 0.35 m from the hinge
    behind_closed_face = np.array([[-.2, .25, 1.]])   # inside the appliance
    far = np.array([[.9, .9, 1.]])
    floor = np.array([[.2, .2, .01]])
    kept = aperture_dock.observed_open_panel_points(
        np.concatenate([remembered_leaf, drifted_leaf, behind_closed_face, far, floor]), aperture)
    assert {tuple(np.round(p, 2)) for p in kept} == {(.25, 0., 1.), (.25, .25, 1.)}


def test_oracle_side_dock_stands_past_the_handle_away_from_the_hinge():
    # Door plane x=0 facing +x; hinge at y=-0.15, closed handle at y=+0.15 (3 cm proud of the door).
    aperture = dict(closed_normal=[1., 0., 0.], hinge=[0., -.15, .7], closed_handle=[.03, .15, .7])
    (dock,) = aperture_dock.side_dock_poses(aperture, [(.51, .33)])
    assert np.allclose(dock[:2], [.51, .48])          # 0.51 m out, 0.33 m past the handle
    assert np.isclose(abs(dock[2]), np.pi)             # facing the cabinet squarely
    assert aperture_dock.side_dock_poses(dict(aperture, closed_handle=[.03, -.15, .7]), [(.51, .33)]) == []
