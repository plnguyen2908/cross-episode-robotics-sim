"""Atomic skill interface for the room-based gather/fill/serve history."""

import numpy as np

from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.skills.interface import SkillBinding, SkillInterface, dock_parameters, parameters
from cross_episode_sim.skills.vessels import navigate_to_vessel, navigate_with_vessel, pick_vessel, place_vessel


def gather_skill_interface(c):
    roles = [i['role'] for i in c.manifest['bindings']]

    def info(role):
        return next(i for i in c.manifest['bindings'] if i['role'] == role)

    def empty():
        return bool(not getattr(c, 'holding_loaf', False) and not c.attached and not c.contacts())

    def held(role):
        return bool(c.object_name == info(role)['body'] and getattr(c, 'holding_loaf', False) and len(c.contacts()) >= 2)

    def at_dock(role, destination):
        dock = getattr(c, '_skill_dock', None)
        if dock is None or dock[:2] != (role, destination):
            return False
        delta = c.base_pose() - np.asarray(dock[2])
        return bool(np.linalg.norm(delta[:2]) < .025 and abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))) < .03)

    def navigate(role, destination, dock='0'):
        c._public_dock_skip = int(dock)
        c._public_navigation_goal = None
        if destination == 'source':
            info(role)['source'] = c.current_room(info(role))
            navigate_to_vessel(c, role)
        else:
            navigate_with_vessel(c, role, destination)
        c._skill_dock = (role, destination, c.base_pose().tolist())

    def nav_pre(role, destination, dock='0'):
        return dict(hand_state=empty() if destination == 'source' else held(role),
                    phase_matches=destination == 'source' or destination == ('filling' if c.phase == 'GATHER' else 'serving'),
                    contents_ready=c.phase == 'GATHER' or c.content_states[role] == 'filled')

    def nav_verify(role, destination, dock='0'):
        goal = getattr(c, '_public_navigation_goal', None)
        if goal is None:
            return False
        xy, yaw = goal
        base = c.base_pose()
        return bool(np.linalg.norm(base[:2]-xy) < .025
                    and (yaw is None or abs(np.arctan2(np.sin(base[2]-yaw), np.cos(base[2]-yaw))) < .03)
                    and (empty() if destination == 'source' else held(role)))

    def wait():
        # TaskEnv can already have fired the same human event after the last
        # placement. Waiting for that completed event must not refill vessels.
        if not c.filled():
            c.human_fill(prepared=True)
        if c.phase != 'SERVE':
            c.configure_phase('SERVE')

    def wait_pre():
        completed = bool(c.filled())
        return dict(vessels_gathered=completed or bool(c.all_at_counter()), empty_hand=empty(),
                    withdrawn=completed or (hasattr(c, '_human_wait_goal') and bool(
                        np.linalg.norm(c.base_pose()[:2] - c._human_wait_goal) < .025)),
                    tucked=completed or bool(c.in_default_travel_posture(False)))

    bindings = dict(
        navigate=SkillBinding(BoundSkill(c, 'navigate', navigate, nav_verify),
            'Navigate to the source vessel or carry it to its kitchen filling site or dining place setting.',
            dock_parameters(parameters(role=roles, destination=('source', 'filling', 'serving'))), nav_pre, navigation=True),
        pick=SkillBinding(BoundSkill(c, 'pick', lambda role, source: pick_vessel(c), lambda role, source: held(role)),
            'Grasp and lift the selected vessel at the current dock; verify physical hold. No navigation.',
            parameters(role=roles, source=('surface',)),
            lambda role, source: dict(empty_hand=empty(), at_source=at_dock(role, 'source'),
                              pickup_prepared=getattr(c, '_public_pickup_ready', None) == info(role)['body'])),
        place=SkillBinding(BoundSkill(c, 'place', lambda role, destination: place_vessel(c, role),
                                     lambda role, destination: c.verify_role(role)),
            'Place, release, and withdraw at the destination dock. Verify support, pose, and retained contents.',
            parameters(role=roles, destination=('filling', 'serving')),
            lambda role, destination: dict(object_held=held(role), at_destination=at_dock(role, destination),
                                           phase_matches=destination == ('filling' if c.phase == 'GATHER' else 'serving'))),
        navigate_to_wait=SkillBinding(BoundSkill(c, 'navigate_to_wait', c.prepare_human_fill,
            lambda: bool(c.all_at_counter() and c.in_default_travel_posture(False)
                         and np.linalg.norm(c.base_pose()[:2] - c._human_wait_goal) < .025)),
            'Withdraw the empty robot from the kitchen filling station and tuck its arm.', parameters(),
            lambda: dict(vessels_gathered=bool(c.all_at_counter()), empty_hand=empty()), navigation=True),
        wait_for_fill=SkillBinding(BoundSkill(c, 'wait_for_fill', wait, c.filled),
            'Wait for the authored human filling event, verify contents, then use serving destinations.', parameters(),
            wait_pre),
    )
    return SkillInterface(c, bindings)


def gather_skill_plan(c):
    steps = []

    def call(name, **arguments):
        steps.append(dict(operation=name, arguments=arguments))

    for destination in ('filling', 'serving'):
        for info in c.manifest['bindings']:
            if destination == 'filling' and c.is_filling_site(c.current_room(info)):
                continue
            role = info['role']
            call('navigate', role=role, destination='source')
            call('pick', role=role, source='surface')
            call('navigate', role=role, destination=destination)
            call('place', role=role, destination=destination)
        if destination == 'filling':
            call('navigate_to_wait')
            call('wait_for_fill')
    return steps
