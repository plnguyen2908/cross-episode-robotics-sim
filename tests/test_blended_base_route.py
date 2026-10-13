"""Blended holonomic routes: short turns, no in-place spins, collision fallback."""
import numpy as np

from cross_episode_sim.navigation.blended_route import blend_route, wrap


def always_clear(pose):
    return True


def route(*poses):
    return [np.array([x, y, np.radians(deg)]) for x, y, deg in poses]


# Planned by the cross-room planner in breakfast_gather_20261004_232706:
# a 7 cm grid hop out of the dock, three long legs, a 5 cm hop into the dock.
LAST_NIGHT = route(
    (2.525, -1.150, 90.0), (2.525, -1.150, -157.1), (2.460, -1.178, -157.1),
    (2.460, -1.178, -76.0), (2.960, -3.178, -76.0), (2.960, -3.178, -110.6),
    (2.660, -3.978, -110.6), (2.660, -3.978, -94.1), (2.560, -5.378, -94.1),
    (2.560, -5.378, 62.0), (2.584, -5.333, 62.0), (2.584, -5.333, -90.0))


def test_planned_route_has_no_in_place_turns_and_no_long_way_round():
    segments = blend_route(LAST_NIGHT, np.pi / 2, always_clear, nav_speed=.12, turn_speed=.08)
    assert [s.kind for s in segments] == ['slide', 'blend', 'blend', 'blend', 'slide']
    np.testing.assert_allclose(segments[-1].end, [2.584, -5.333, -np.pi / 2], atol=1e-9)
    total = sum(s.rotation for s in segments)
    planned_shortest = sum(abs(wrap(b[2] - a[2])) for a, b in zip(LAST_NIGHT, LAST_NIGHT[1:]))
    # 90 -> -76 -> -110.6 -> -94.1 -> -90: 166 + 34.6 + 16.5 + 4.1 degrees.
    assert np.degrees(total) < 222
    assert total < planned_shortest
    for segment in segments:
        assert segment.rotation < np.pi + 1e-9


def test_rotation_happens_early_so_most_travel_faces_forward():
    segments = blend_route(LAST_NIGHT, np.pi / 2, always_clear, nav_speed=.12, turn_speed=.4)
    leg = segments[1]
    heading = np.arctan2(*(leg.end[:2] - leg.start[:2])[::-1])
    aligned = [abs(wrap(leg.pose(u)[2] - heading)) < np.radians(5)
               for u in np.linspace(0, 1, 101)]
    assert np.mean(aligned) > .3
    np.testing.assert_allclose(leg.pose(1.0), leg.end)


def test_continuous_yaw_follows_measured_hinge_angle():
    # Measured hinge one revolution past the planned heading: keep the chain
    # continuous from the measured value, no 2pi jump back.
    segments = blend_route(route((0, 0, 170), (0, 0, -170), (1, 0, -170)),
                           np.radians(530), always_clear, nav_speed=.12, turn_speed=.08)
    assert abs(segments[0].start[2] - np.radians(530)) < 1e-9
    assert segments[-1].rotation < np.radians(25)


def test_blocked_blend_falls_back_to_short_turns_then_none():
    path = route((0, 0, 0), (0, 0, 90), (0, 1, 90))
    # Forbid poses that are both off the start and rotated: blending is
    # unclear, in-place turn then straight drive is clear.
    def corridor(pose):
        return np.linalg.norm(pose[:2]) < 1e-6 or abs(wrap(pose[2] - np.pi / 2)) < 1e-6
    segments = blend_route(path, 0., corridor, nav_speed=.12, turn_speed=.08)
    assert [s.kind for s in segments] == ['turn', 'drive']
    assert blend_route(path, 0., lambda pose: False, nav_speed=.12, turn_speed=.08) is None


def test_reverse_undock_is_kept_straight():
    path = route((0, 0, 90), (0, -.3, 90), (0, -.3, 0), (2, -.3, 0))
    segments = blend_route(path, np.pi / 2, always_clear, nav_speed=.12,
                           turn_speed=.08, reverse_first=True)
    assert segments[0].kind == 'reverse'
    assert segments[0].rotation == 0
    assert segments[1].kind == 'blend'


def test_shortcut_reduces_stop_segments_and_preserves_endpoint():
    from cross_episode_sim.navigation.blended_route import make_segment, shortcut_segments
    points = [np.array([x, 0., 0.]) for x in (0., .2, .4, .6, .8)]
    segments = [make_segment(a, b[:2], 0., 0., 'slide', .24, .6) for a, b in zip(points, points[1:])]
    probes = []
    merged = shortcut_segments(segments, lambda pose: probes.append(pose.copy()) or True,
                               nav_speed=.24, turn_speed=.6)
    assert len(merged) == 1 and probes
    assert sum(s.duration for s in merged) < sum(s.duration for s in segments)
    np.testing.assert_allclose(merged[0].start, points[0])
    np.testing.assert_allclose(merged[-1].end, points[-1])


def test_shortcut_keeps_obstacle_detour_and_reverse_undock():
    from cross_episode_sim.navigation.blended_route import make_segment, shortcut_segments
    points = [np.array(p) for p in [(0., 0., 0.), (0., 1., 0.), (1., 1., 0.), (1., 0., 0.)]]
    segments = [make_segment(a, b[:2], 0., 0., 'slide', .24, .6) for a, b in zip(points, points[1:])]
    clear = lambda p: not (.1 < p[0] < .9 and p[1] < .95)
    result = shortcut_segments(segments, clear, nav_speed=.24, turn_speed=.6)
    assert len(result) == 3
    reverse = make_segment(np.array([-.1, 0., 0.]), points[0][:2], 0., 0., 'reverse', .08, .6)
    result = shortcut_segments([reverse, *segments], lambda p: True, nav_speed=.24, turn_speed=.6)
    assert result[0] is reverse


def test_shortcut_probe_budget_never_accepts_partially_checked_motion():
    from cross_episode_sim.navigation.blended_route import make_segment, shortcut_segments
    points = [np.array([x, 0., 0.]) for x in (0., .2, .4)]
    segments = [make_segment(a, b[:2], 0., 0., 'drive', .24, .6) for a, b in zip(points, points[1:])]
    probes = []
    result = shortcut_segments(segments, lambda p: probes.append(p) or True,
                               nav_speed=.24, turn_speed=.6, probe_budget=3)
    assert len(probes) == 3
    assert len(result) == 2 and result[0] is segments[0] and result[1] is segments[1]
