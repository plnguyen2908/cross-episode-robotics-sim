"""Vectorized contacts preserve the scalar collision rules and target changes."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from cross_episode_sim.manipulation.recovery import RecoveringManipulation


NAMES = ['world', 'floor_kitchen', 'wall', 'table', 'cup', 'food',
         'robot_0/base', 'robot_0/left_pad', 'robot_0/right_pad', 'handle']


class Contacts(list):
    def __getattr__(self, name):
        return np.asarray([getattr(c, name) for c in self],
                          dtype=int if name in ('geom1', 'geom2') else float)


def runner(seed):
    rng = np.random.default_rng(seed)
    c = GatherBreakfast.__new__(GatherBreakfast)
    c.model = SimpleNamespace(nbody=len(NAMES), ngeom=len(NAMES),
        geom_bodyid=np.arange(len(NAMES)), geom_type=np.full(len(NAMES), 6),
        body=lambda i: SimpleNamespace(name=NAMES[i]),
        geom=lambda i: SimpleNamespace(name=NAMES[i]))
    c.model.geom_type[0] = 0  # plane
    c.bread_bids = {4}
    c.fridge_bids = {9}
    c.table_gids = {3}
    c.handle_bid = 9
    c.content_states = {}
    c.args = SimpleNamespace(native_object=bool(seed % 2), kitchen=True)
    c.embodiment = SimpleNamespace(is_finger=lambda n: n.endswith('_pad'))
    c.data = SimpleNamespace(contact=Contacts(
        SimpleNamespace(geom1=int(a), geom2=int(b), dist=float(d),
                        frame=np.array([0., 0., normal]), pos=np.zeros(3),
                        includemargin=.001)
        for a,b,d,normal in zip(rng.integers(0,len(NAMES),100),
            rng.integers(0,len(NAMES),100),rng.uniform(-.01,.01,100),
            rng.choice([-1., 1.],100))))
    c.allowed_panel_contact = lambda robot,other: robot.endswith('_pad') and other == 2
    return c


def scalar_navigation(c, carrying, contents=None):
    worst = 0.
    for contact in c.data.contact:
        a,b = contact.geom1,contact.geom2
        na,nb = NAMES[a],NAMES[b]
        if na.startswith('floor_') or nb.startswith('floor_'):
            continue
        ra,rb = na.startswith('robot_0/'),nb.startswith('robot_0/')
        if contents is None:
            if ra != rb or (carrying and ((a in c.bread_bids) != (b in c.bread_bids)) and not (ra or rb)):
                if carrying and ((ra and b in c.bread_bids) or (rb and a in c.bread_bids)):
                    continue
                worst = max(worst,-contact.dist+.001)
        else:
            payload = c.bread_bids | contents
            ma,mb = ra or (carrying and a in payload),rb or (carrying and b in payload)
            oa,ob = not ra and a not in payload,not rb and b not in payload
            if (ma and ob) or (mb and oa):
                worst = max(worst,-contact.dist+.001)
    return worst


def scalar_penetration(c):
    worst = 0.
    for contact in c.data.contact:
        a,b = contact.geom1,contact.geom2
        if NAMES[a].startswith('robot_0/'):
            robot,other = NAMES[a],b
        elif NAMES[b].startswith('robot_0/'):
            robot,other = NAMES[b],a
        else:
            continue
        if contact.dist >= 0.:
            continue
        if (other in c.bread_bids or other == c.handle_bid) and c.is_finger(robot):
            continue
        if c.allowed_panel_contact(robot,other):
            continue
        native = c.args.native_object and not NAMES[other].startswith('robot_0/') and c.model.geom_type[other] != 0
        if native or other in c.fridge_bids or other in c.bread_bids or other in c.table_gids:
            worst = max(worst,-contact.dist)
    return worst


@pytest.mark.parametrize('seed', range(16))
def test_contact_rules_match_scalar_reference_across_target_changes(seed):
    c = runner(seed)
    for obj,table,fridge in (({4},{3},{9}),({5},{2,3},set())):
        c.bread_bids,c.table_gids,c.fridge_bids = obj,table,fridge
        assert c.unintended_penetration() == scalar_penetration(c)
        for carrying in (False,True):
            assert RecoveringManipulation.navigation_penetration(c,c.data,carrying) == scalar_navigation(c,carrying)
            with patch('cross_episode_sim.manipulation.physical_contents.active_content_bodies', return_value={5}):
                assert c.navigation_penetration(c.data,carrying) == scalar_navigation(c,carrying,{5})


def test_empty_contact_arrays_are_supported():
    c = runner(0)
    c.data.contact = Contacts()
    assert c.unintended_penetration() == 0.
    assert c.navigation_penetration(c.data,True) == 0.
    assert c.finger_object_contact() == dict(fingers=[],depth_m=0.,normal_force_n={})


def test_pickup_clearance_is_monotonic_but_drop_detection_continues():
    c = runner(0)
    c.holding_loaf = True
    c._drop_reference = np.zeros(3)
    c._pickup_cleared = True
    c._drop_seconds = 0.
    c.model.opt = SimpleNamespace(timestep=.05)
    c.pickup_start_height = 0.
    c.report = {}
    pose = np.eye(4);pose[:3,3] = [.2,0.,1.]
    c.bread_pose = lambda: pose
    c.tcp = lambda: np.eye(4)
    def redundant_check(**kwargs):
        raise AssertionError('Pickup clearance must not be rescanned after it is established')
    c.support_contacts = redundant_check
    contact = dict(fingers=[],normal_force_n={})
    GraspQualification.validate_loaded_hold(c,contact)
    with pytest.raises(RuntimeError,match='Object dropped after clearing pickup support'):
        GraspQualification.validate_loaded_hold(c,contact)
