from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.skills.fixture_limits import opening_target
from cross_episode_sim.skills.household_fixtures import fixture_satisfied


@pytest.mark.parametrize('bounds,sliding,expected', [
    ((0., np.pi/2), False, np.radians(89.5)),
    ((-np.pi/2, 0.), False, -np.radians(89.5)),
    ((-.4, 0.), True, -.397),
    ((0., .5), True, .497),
])
def test_target_respects_signed_native_limit(bounds, sliding, expected):
    joint = SimpleNamespace(range=bounds, limited=True)
    assert opening_target(joint, sliding=sliding) == pytest.approx(expected)


def test_unlimited_joint_cannot_claim_fully_open():
    with pytest.raises(ValueError, match='finite physical'):
        opening_target(SimpleNamespace(range=(0., 0.), limited=False))


def test_drawer_dock_is_same_when_open_or_closed():
    from cross_episode_sim.skills.fixture_limits import drawer_dock
    closed = np.array([.5,-.64,.7])
    axis = np.array([0.,1.,0.])
    target = -.597
    a=drawer_dock(closed,axis,0.,target)
    b=drawer_dock(closed+axis*target,axis,target,target)
    np.testing.assert_allclose(a,b)
    np.testing.assert_allclose(a[:2],[.8,-1.34])


@pytest.mark.parametrize('kind,bounds,partial,full', [
    ('cabinet-door', (0., np.pi/2), np.radians(65), np.radians(89)),
    ('drawer', (-.4, 0.), -.20, -.396),
])
def test_partial_open_is_not_success_and_close_is_checked(kind,bounds,partial,full):
    joint = SimpleNamespace(range=bounds, limited=True)
    c = SimpleNamespace(fixture_joint='hinge',model=SimpleNamespace(joint=lambda _:joint),angle=lambda:partial)
    assert not fixture_satisfied(c,kind,'open')
    c.angle=lambda:full
    assert fixture_satisfied(c,kind,'open')
    assert not fixture_satisfied(c,kind,'closed')
    c.angle=lambda:0.
    assert fixture_satisfied(c,kind,'closed')



def test_base_assisted_drawer_dock_tracks_slider_and_returns_to_safe_closed_dock():
    from cross_episode_sim.skills.fixture_limits import drawer_dock
    closed=np.array([.5,-.64,.7]);axis=np.array([0.,1.,0.]);target=-.597
    a=drawer_dock(closed,axis,0.,target,base_assisted=True)
    b=drawer_dock(closed+axis*target,axis,target,target,base_assisted=True)
    np.testing.assert_allclose(b[:2],a[:2]+axis[:2]*target)
    np.testing.assert_allclose(b[:2]+axis[:2]*(0.-target),a[:2])
    assert b[2] == pytest.approx(a[2])
