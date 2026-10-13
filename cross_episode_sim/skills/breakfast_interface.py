"""Grounded atomic action domain for the current breakfast-with-coffee scene.

History recipes and planner calls share these physical implementations. The
public domain has no ``coffee``, ``breakfast``, ``serve`` or ``transfer`` action.
"""

import numpy as np

from cross_episode_sim.skills.atomic import AtomicSkill, BoundSkill
from cross_episode_sim.skills.coffee import ParkDosingBoxSkill, PickDosingBoxSkill, PourGroundsSkill, press_button_skill
from cross_episode_sim.skills.coffee_cups import (
    PickCounterCupSkill,
    PickCupFromSpoutSkill,
    PlaceCupAtSettingSkill,
    PlaceCupAtSpoutSkill,
    PlaceCupOnCounterSkill,
)
from cross_episode_sim.skills.interface import SkillBinding, SkillInterface, dock_parameters, parameters
from cross_episode_sim.skills.portafilter import AttachPortafilterSkill, DetachPortafilterSkill, detached
from cross_episode_sim.skills.vessels import navigate_to_vessel, navigate_with_vessel, pick_vessel, place_vessel


def breakfast_skill_interface(c):
    """Construct the same live executor for oracle or VLM-selected actions."""
    cups = c.coffee_mugs_roles()
    bowls = [i["role"] for i in c.manifest["bindings"] if i["role"].startswith("bowl")]
    roles = cups + bowls
    full = 'full_breakfast' in c.manifest
    bindings = {}

    def empty():
        return bool(not c.holding_loaf and not c.attached and not c.contacts())

    def held(role):
        return bool(c.object_name == c.info(role)["body"] and c.holding_loaf and len(c.contacts()) >= 2)

    def at_dock(role, destination):
        dock = getattr(c, '_skill_dock', None)
        if dock is None or dock[:2] != (role, destination):
            return False
        base = c.base_pose()
        delta = base - np.asarray(dock[2])
        return bool(np.linalg.norm(delta[:2]) < .025 and abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))) < .03)

    def add(name, skill, schema, pre, description, navigation=False):
        argument_docs = {
            'role': 'The cup or bowl to act on. Keep the same vessel across navigation, pickup and placement; not the navigation destination.',
            'dock': 'Optional approach candidate index as a string, default "0". Use another available index to try another dock; this does not change the target object.',
        }
        if name == 'navigate':
            argument_docs['destination'] = ('source: empty-hand approach for pickup; spout: carry cup to machine '
                '(or return empty-handed to its ready cup); home: carry cup to recorded home; '
                'serving: carry cup/bowl to dining. Only values in this enum are available.')
        elif name == 'pick':
            argument_docs['source'] = 'surface: cup/bowl at source dock; spout: cup ready under machine at spout dock. Both require empty hand.'
        elif name == 'place':
            argument_docs['destination'] = 'spout: release cup under machine; home: release cup at home; serving: release cup/bowl at dining. Requires matching dock and held vessel.'
        for key, text in argument_docs.items():
            if key in schema['properties']:
                schema['properties'][key]['description'] = text
        bindings[name] = SkillBinding(skill, description, schema, pre, navigation)

    add("detach_portafilter", DetachPortafilterSkill(c), parameters(),
        lambda: dict(empty_hand=empty(), seated=c.fixture_phase in ("seated", "locked")),
        "Arguments: {}. Requires empty hand and a seated/locked portafilter. Detach it and leave it supported on the loading counter; the hand is empty afterward. Does not pick up or pour grounds.")
    add("pick_dosing_box", PickDosingBoxSkill(c), parameters(),
        lambda: dict(empty_hand=empty(), portafilter_detached=bool(detached(c))),
        "Arguments: {}. Requires empty hand and detached portafilter. Grasp the CoffeeDosingBox at the machine dock; afterward the box is held, but no grounds have been poured. Use pour_grounds to dose, then place_dosing_box to free the hand. Calling pick_dosing_box again while holding it is invalid.")
    add("pour_grounds", PourGroundsSkill(c, track_portafilter=True), parameters(),
        lambda: dict(box_held=bool(c.object_name == c.spec['dosing_body'] and c.holding_loaf and len(c.contacts()) >= 2)),
        "Arguments: {}. Requires the grounds box already held (pick_dosing_box). Pour grounds into the detached portafilter and return the box upright. The box remains held: use place_dosing_box before any empty-hand skill.")
    add("place_dosing_box", ParkDosingBoxSkill(c), parameters(),
        lambda: dict(box_held=bool(c.object_name == c.spec['dosing_body'] and c.holding_loaf and len(c.contacts()) >= 2)),
        "Arguments: {}. Requires the grounds box held. Put it on its parking counter and release it; afterward the hand is empty. This does not pour grounds. Use this dedicated skill for the box, not vessel place.")
    add("attach_portafilter", AttachPortafilterSkill(c), parameters(),
        lambda: dict(empty_hand=empty(), detached=bool(detached(c)), grounds_ready=bool(c.grounds_ready())),
        "Arguments: {}. Requires empty hand, detached portafilter and grounds ready. Grasp the supported portafilter, insert and lock it, then release; afterward the hand is empty.")

    def configure_bowl(role):
        info = c.info(role)
        support = c.assignment(objects=[info['body']])[info['body']]
        source = info.get('source')
        if c.task_supports.get(source) != support:
            source = next(k for k, v in c.task_supports.items() if v == support)
        info.update(source=source,
                    destination='dining', destination_position=list(info['serving_position']))
        navigate_to_vessel(c, role)

    def navigate(role, destination, dock='0', setting='default'):
        c._public_dock_skip = int(dock)
        c._public_navigation_goal = None
        info = c.info(role)
        if destination == 'serving':
            if setting != 'default':
                kind = 'cup' if role in cups else 'bowl'
                target_role = c.manifest['full_breakfast']['settings'][int(setting)-1][kind]
                info['serving_position'] = list(serving_targets[target_role])
            info.update(destination='dining', destination_position=list(info['serving_position']))
        if role in bowls:
            if destination == "source":
                configure_bowl(role)
            else:
                navigate_with_vessel(c, role, destination)
        else:
            c.active_mug = info['body']
            if destination == "spout":
                c.return_to_machine()
            else:
                point = (c.data.body(info['body']).xpos.copy() if destination == "source" else
                         np.asarray(c.mug_homes[info['body']])[:3, 3] if destination == "home" else
                         np.asarray(info['serving_position']))
                c.walk_to_counter(info['body'], point, carrying=held(role),
                                  room='dining' if destination == 'serving' else info['source'])
        c._skill_dock = (role, destination, c.base_pose().tolist())

    def navigation_hand(role, destination):
        if destination == 'source':
            return empty()
        if held(role):
            return True
        return bool(destination == 'spout' and role in cups and empty()
                    and c.info(role)['body'] == c.mug and c.mug_ready())

    def nav_pre(role, destination, dock='0', setting='default'):
        return dict(setting_supported=setting == 'default' or (full and destination == 'serving' and int(setting) <= len(c.manifest['full_breakfast']['settings'])),
                    destination_supported=role in cups or destination in ('source', 'serving'),
                    hand_state=navigation_hand(role, destination),
                    home_known=destination != 'home' or c.info(role)['body'] in getattr(c, 'mug_homes', {}))

    def nav_verify(role, destination, dock='0', setting='default'):
        goal = getattr(c, '_public_navigation_goal', None)
        if goal is None:
            return False
        xy, yaw = goal
        base = c.base_pose()
        yaw_ok = yaw is None or abs(np.arctan2(np.sin(base[2]-yaw), np.cos(base[2]-yaw))) < np.deg2rad(1.1)
        return bool(np.linalg.norm(base[:2]-xy) < .025 and yaw_ok
                    and navigation_hand(role, destination))

    serving_targets = {i['role']: list(i['serving_position']) for i in c.manifest['bindings'] if 'serving_position' in i}
    navigation_schema = dock_parameters(parameters(role=roles, destination=('source', 'spout', 'home', 'serving') if full else ('source', 'spout', 'home')))
    if full and c.manifest['full_breakfast'].get('interchangeable_vessels'):
        navigation_schema['properties']['setting'] = dict(type='string', enum=['default', '1', '2', '3'], default='default',
            description='Dining place setting to use when destination is serving; choose the requested settings from the histories.')
    add('navigate', BoundSkill(c, 'navigate', navigate, nav_verify),
        navigation_schema, nav_pre,
        'Move the base and dock; never pick or place implicitly. destination=source: approach the selected vessel for surface pickup; requires empty hand. destination=spout: cups only, carry the selected held cup to the machine; also allows empty-hand return to that same cup when it is already ready under the spout. destination=home: cups only, carry the selected held cup to its recorded home counter; home must be known. destination=serving: carry the selected held cup or bowl to its dining setting. Holding a different object (including the grounds box) is invalid. For surface pickup use navigate(destination=source), then pick(source=surface) on the same vessel. For delivery use navigate(destination=X), then place(destination=X) on the same vessel. dock chooses an approach candidate, not an object; setting selects the dining placement only for destination=serving.', True)

    def pick(role, source, candidate=None):
        if role in cups:
            c.active_mug = c.info(role)['body']
            skill = PickCounterCupSkill(c) if source == 'surface' else PickCupFromSpoutSkill(c)
            skill.execute_candidate(candidate)
        else:
            # Keep the inherited grasp preflights and measured hold qualification,
            # but navigation has already happened in a public navigate call.
            pick_vessel(c)

    def pick_pre(role, source):
        body = c.info(role)['body']
        return dict(empty_hand=empty(), source_supported=role in cups or source == 'surface',
                    at_pickup_dock=at_dock(role, 'source' if source == 'surface' else 'spout'),
                    pickup_prepared=role in cups or getattr(c, '_public_pickup_ready', None) == body,
                    spout_ready=source != 'spout' or (c.mug == body and bool(c.mug_ready())))

    class VesselPickSkill(AtomicSkill):
        def candidates(self, role, source):
            return PickCupFromSpoutSkill(c).candidates() if role in cups and source == 'spout' else (None,)

        def execute_candidate(self, candidate, role, source):
            return pick(role, source, candidate)

        def verify(self, role, source):
            return held(role)

    add('pick', VesselPickSkill(c, 'pick'),
        parameters(role=roles, source=('surface', 'spout')), pick_pre,
        'Requires empty hand and the matching vessel pickup dock; no base navigation. source=surface: lift a cup or bowl after navigate(destination=source) for the same vessel. source=spout: cups only, extract the selected cup already ready under the machine spout at its spout dock. To collect brewed coffee, finish wait_until_brewed first. Success leaves the selected vessel held. Neither mode picks the grounds box; use pick_dosing_box for that.')

    def place(role, destination, candidate=None):
        if destination == 'serving' and not hasattr(c, 'serve_start_time'):
            c.serve_start_time = float(c.data.time)
        if role in cups:
            c.active_mug = c.info(role)['body']
            if destination == 'spout':
                PlaceCupAtSpoutSkill(c).execute_candidate(candidate)
            elif destination == 'home':
                PlaceCupOnCounterSkill(c).execute_candidate(None)
            else:
                PlaceCupAtSettingSkill(c).execute_candidate(None, role, c.task_supports['dining'])
        else:
            place_vessel(c, role)
        if destination == 'serving':
            c.event('served', role=role, position=c.data.body(c.info(role)['body']).xpos.tolist())

    def place_verify(role, destination):
        if not empty():
            return False
        if destination == 'serving':
            return bool(c.served(role))
        return (PlaceCupAtSpoutSkill(c).verify() if destination == 'spout' else PlaceCupOnCounterSkill(c).verify())

    class VesselPlaceSkill(AtomicSkill):
        def candidates(self, role, destination):
            return PlaceCupAtSpoutSkill(c).candidates() if role in cups and destination == 'spout' else (None,)

        def execute_candidate(self, candidate, role, destination):
            return place(role, destination, candidate)

        def verify(self, role, destination):
            return place_verify(role, destination)

    add('place', VesselPlaceSkill(c, 'place'),
        parameters(role=roles, destination=('spout', 'home', 'serving') if full else ('spout', 'home')),
        lambda role, destination: dict(object_held=held(role), at_destination_dock=at_dock(role, destination),
                                       destination_supported=role in cups or destination == 'serving'),
        'Requires the selected vessel held and a matching destination dock from navigate for the same vessel. destination=spout: cups only, position and release under the machine for brewing. destination=home: cups only, return and release at its recorded counter home. destination=serving: release a cup or bowl at the dining setting selected during navigate. Success leaves the hand empty. No base navigation; no pouring or brewing. Bowls support only serving. Grounds box uses place_dosing_box instead.')

    for action in ('power', 'brew'):
        def pre(action=action):
            return dict(empty_hand=empty(), prerequisites=(action == 'power' or bool(
                c.machine_cycle.ready(c.data.time) and c.installed() and c.grounds_ready() and c.mug_ready())))
        add('press_' + action, press_button_skill(c, action), parameters(), pre,
            ('Arguments: {}. Requires empty hand. Physically press and release power to power the machine; '
             'use wait_until_ready for warmup. Does not brew.' if action == 'power' else
             'Arguments: {}. Requires empty hand, warmed-up machine, installed portafilter, grounds ready '
             'and a cup ready under the spout. Press and release brew to start the cycle; '
             'use wait_until_brewed before collecting the cup.'))
    add('wait_until_ready', BoundSkill(c, 'wait_until_ready', c.wait_machine_ready, lambda: c.machine_cycle.ready(c.data.time)),
        parameters(), lambda: dict(powered=c.machine_cycle.powered_at is not None, empty_hand=empty()),
        'Arguments: {}. Requires machine powered on and empty hand. Advance simulated warmup until the machine is ready. Does not press power or start brewing.')
    add('wait_until_brewed', BoundSkill(c, 'wait_until_brewed', c.finish_brew,
                                      lambda: c.machine_cycle.completed and c.mug_ready()),
        parameters(), lambda: dict(brew_started=c.machine_cycle.started_at is not None, empty_hand=empty()),
        'Arguments: {}. Requires an already-started brew cycle and empty hand. Wait for completion with the cup still ready under the spout. Does not start brewing or pick the cup; use press_brew to start and pick(source=spout) to collect afterward.')
    if full:
        add('wait_for_bowls', BoundSkill(c, 'wait_for_bowls', c.fill_bowls, c.bowls_filled), parameters(),
            lambda: dict(empty_hand=empty(), bowls_in_kitchen=c.bowls_in_kitchen()),
            'Arguments: {}. Requires empty hand and requested bowls in the kitchen. Wait for the authored human intervention that fills the requested bowls. Filling is an environment event, not robot manipulation. Does not navigate, pick or serve bowls.')
    return SkillInterface(c, bindings)


def current_skill_plan(manifest):
    """A grounded witness plan, executable unchanged by either planner client.

    This is an oracle-provided plan, not a claim of general symbolic search or
    learned planner performance. Every action must pass the physical executor.
    """
    plan = []

    def call(skill, **arguments):
        plan.append(dict(operation=skill, arguments=arguments))

    for name in ('detach_portafilter', 'pick_dosing_box', 'pour_grounds', 'place_dosing_box', 'attach_portafilter'):
        call(name)
    mugs = manifest['coffee']['coffee_mugs']
    if manifest.get('full_breakfast', {}).get('interchangeable_vessels'):
        selected = {s['cup'] for s in manifest['full_breakfast']['settings']}
        mugs = [i['body'] for i in manifest['bindings'] if i['role'] in selected]
    roles = [next(i['role'] for i in manifest['bindings'] if i['body'] == body) for body in mugs]
    for index, role in enumerate(roles):
        call('navigate', role=role, destination='source')
        call('pick', role=role, source='surface')
        call('navigate', role=role, destination='spout')
        call('place', role=role, destination='spout')
        if index == 0:
            call('press_power')
            call('wait_until_ready')
        call('press_brew')
        call('wait_until_brewed')
        call('pick', role=role, source='spout')
        destination = 'serving' if 'full_breakfast' in manifest else 'home'
        call('navigate', role=role, destination=destination)
        call('place', role=role, destination=destination)
    if 'full_breakfast' not in manifest:
        return plan
    call('wait_for_bowls')
    for setting in manifest['full_breakfast']['settings']:
        role = setting['bowl']
        call('navigate', role=role, destination='source')
        call('pick', role=role, source='surface')
        call('navigate', role=role, destination='serving')
        call('place', role=role, destination='serving')
    return plan
