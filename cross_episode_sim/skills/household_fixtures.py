"""Shared physical fixture primitives, independent of a composite recipe.

The standalone demos and persistent-scene adapters call these same functions.
They never build/reset a scene or assign a fixture joint position.
"""
import numpy as np
from cross_episode_sim.skills.fixture_limits import opening_target, opening_tolerance


def articulate_fixture(controller, kind, state):
    """Passive fixture motion with release and any required door regrasp."""
    targets = {
        'cabinet-door': {'open': np.pi/3, 'closed': 0.},
        'drawer': {'open': -.20, 'closed': 0.},
        'stove-knob': {'on': np.pi/3, 'off': 0.},
        'faucet': {'on': .48, 'off': 0.},
    }
    target = targets[kind][state]
    if state == 'open' and kind in ('cabinet-door', 'drawer'):
        target = opening_target(controller.model.joint(controller.fixture_joint),
                                sliding=kind == 'drawer')
    if kind == 'cabinet-door':
        from cross_episode_sim.skills.door_motion import staged_door_motion
        return staged_door_motion(controller, target)
    previous = getattr(controller, 'requested_hinge_target', None)
    try:
        if kind in ('cabinet-door', 'drawer'):
            controller.requested_hinge_target = target
        controller.grasp_handle()
    finally:
        controller.requested_hinge_target = previous
    (controller.follow_slide if kind == 'drawer' else controller.follow_hinge)(target)
    controller.release_handle()


def fixture_satisfied(controller, kind, state):
    from cross_episode_sim.fixtures.faucet import water_on
    from cross_episode_sim.fixtures.stove_knob import burner_on
    q = controller.angle()
    if kind in ('cabinet-door', 'drawer'):
        sliding = kind == 'drawer'
        target = opening_target(controller.model.joint(controller.fixture_joint),
                                sliding=sliding) if state == 'open' else 0.
        return bool(abs(q-target) < opening_tolerance(sliding=sliding))
    predicate = burner_on if kind == 'stove-knob' else water_on
    return bool(predicate(q) if state == 'on' else not predicate(q) and abs(q) < .05)


def press_fixture_button(controller, button):
    """Contact press, shared with the original microwave atomic demonstration."""
    controller.embodiment.command_gripper(controller.data, controller.profile.gripper_close)
    from cross_episode_sim.control.timing import wait_for_gripper_command
    wait_for_gripper_command(controller, 1.)
    controller.rebuild_planner()
    controller.press(button)


class SharedFixtureSkills:
    """Bind existing fixture controller methods to a single live controller state.

    Views share the model/data/planner/trace dictionary. Their constructors are
    deliberately not called: scene initialization belongs exclusively to datagen.
    Geometry selection is restored after each action, never physics state.
    """
    def __init__(self, environment):
        self.env = environment

    def view(self, target, button=False, binding_override=None):
        from contextlib import contextmanager

        from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
        from cross_episode_sim.fixtures.drawer import DrawerTest
        from cross_episode_sim.fixtures.stove_knob import StoveKnobTest
        binding = binding_override or self.env.fixture_bindings[target]
        if binding.get('room_door'):
            # A door call can follow a different navigation/action; resolve
            # its accessible face from the current robot pose, not stale nav.
            c = self.env.controller
            binding = dict(binding)
            normal = (c.data.body(binding['body']).xmat.reshape(3, 3) @ np.array([0., -1., 0.]))[:2]
            point = c.data.geom(binding['handle_geom']).xpos
            sign = 1. if np.dot(c.base_pose()[:2]-point[:2], normal) >= 0 else -1.
            prefix, side = binding['body'].rsplit('_leaf_', 1)
            face = -1 if sign > 0 else 1
            binding.update(door_approach_sign=sign,
                handle_body=f'{prefix}_handle_body_{side}_{face}',
                handle_geom=f'{prefix}_handle_{side}_{face}')
        cls = {'cabinet-door': CabinetDoorTest, 'drawer': DrawerTest, 'stove-knob': StoveKnobTest}[binding['kind']]
        if button:
            from cross_episode_sim.fixtures.microwave_button import MicrowaveButtonTest
            cls = MicrowaveButtonTest
        def before_step(view):
            cls.before_step(view)
            observer = getattr(view, '_world_observer', None)
            if observer is not None:
                observer()
        methods={'before_step': before_step}
        if (binding.get('native_contact_refinement') or binding.get('room_door')) and not button:
            from cross_episode_sim.fixtures.faucet import FaucetTest
            methods.update(load_world=FaucetTest.load_world,
                           plan_handle_grasp=FaucetTest.plan_handle_grasp)
        view_class = type('Shared'+cls.__name__, (cls,), methods)
        c = self.env.controller
        @contextmanager
        def selected():
            view = object.__new__(view_class)
            view.__dict__ = c.__dict__
            values = dict(fixture_body=binding['body'], fixture_joint=binding['joint'],
                handle_body=binding['handle_body'], handle_geometry=binding['handle_geom'],
                door_joint=c.model.joint(binding['joint']).id,
                door_address=c.model.jnt_qposadr[c.model.joint(binding['joint']).id],
                door_bids=c.descendants(binding['body']), handle_bids=c.descendants(binding['handle_body']),
                handle_geom=c.model.geom(binding['handle_geom']).id, articulating=False,
                prefer_flipped_handle=binding.get('prefer_flipped_handle',False),
                door_stroke_degrees=binding.get('door_stroke_degrees', 15. if target=='fridge' else None),
                base_assisted_articulation=binding.get('base_assisted_articulation', True),
                fixed_arm_panel_push=binding.get('fixed_arm_panel_push', False),
                base_assisted_close=binding.get('base_assisted_close', False),
                contact_base_speed=binding.get('contact_base_speed', .24),
                fixture_nav_speed=binding.get('fixture_nav_speed', .30),
                fixture_turn_speed=binding.get('fixture_turn_speed', .8),
                panel_push_height_offset=binding.get('panel_push_height_offset', .12 if binding.get('room_door') else -.15 if target == 'fridge' else 0.),
                retry_panel_after_progress=binding.get('retry_panel_after_progress', target == 'fridge'),
                door_continue_grasp=binding.get('door_continue_grasp',
                    binding['kind'] == 'cabinet-door'),
                door_push_contact=binding.get('door_push_contact', binding['kind'] == 'cabinet-door'),
                door_push_close=binding.get('door_push_close',
                    binding['kind'] == 'cabinet-door' and binding.get('native_contact_refinement', False)),
                door_approach_sign=binding.get('door_approach_sign',1.),
                room_door=binding.get('room_door',False),
                handle_pitch_candidates=binding.get('handle_pitch_candidates',(0.,)),
                handle_yaw_candidates=binding.get('handle_yaw_candidates',(0.,)),
                _precise_contact_boxes=set(), _door_plans={}, _handle_grasp_cache=[])
            if button:
                values.update(handle_body='microwave_main_group_main', handle_bids=set(),
                              handle_geom=c.button_ids['start'])
            absent = object()
            saved = {k: c.__dict__.get(k, absent) for k in values}
            c.__dict__.update(values)
            try:
                yield view
            finally:
                for k, value in saved.items():
                    if value is absent: c.__dict__.pop(k, None)
                    else: c.__dict__[k] = value
        return selected()

    def articulate(self, target, state):
        binding = self.env.fixture_bindings[target]
        if binding.get('leaves'):
            leaves = binding['leaves']
            if state == 'closed':
                c = self.env.controller
                leaves = sorted(leaves, key=lambda leaf: np.linalg.norm(
                    c.data.geom(leaf['handle_geom']).xpos[:2] - c.base_pose()[:2]))
            for leaf in leaves:
                with self.view(target, binding_override=leaf) as c:
                    from cross_episode_sim.skills.door_motion import staged_door_motion
                    staged_door_motion(c, leaf['open_position'] if state == 'open' else 0.)
            return
        with self.view(target) as c:
            # Use the same physical handle/stroke/release methods as atomic demos.
            if binding['kind'] == 'stove-knob':
                articulate_fixture(c, binding['kind'], state)
            else:
                goal = binding['open_position'] if state == 'open' else 0.
                if binding['kind']=='cabinet-door':
                    if target == 'fridge' and state == 'closed':
                        from cross_episode_sim.skills.fridge_door_motion import fridge_door_motion
                        fridge_door_motion(c, goal)
                    else:
                        from cross_episode_sim.skills.door_motion import staged_door_motion
                        staged_door_motion(c, goal)
                    return
                previous=getattr(c,'requested_hinge_target',None)
                try:
                    c.requested_hinge_target=goal
                    c.grasp_handle()
                finally:
                    c.requested_hinge_target=previous
                (c.follow_slide if binding['kind']=='drawer' else c.follow_hinge)(goal)
                c.release_handle()

    def satisfied(self, target, state):
        binding = self.env.fixture_bindings[target]
        c = self.env.controller
        if binding.get('leaves'):
            return all(abs(float(c.data.joint(leaf['joint']).qpos[0]) -
                       (leaf['open_position'] if state == 'open' else 0.)) < opening_tolerance()
                       for leaf in binding['leaves'])
        q = float(c.data.joint(binding['joint']).qpos[0])
        if binding['kind'] == 'stove-knob':
            from cross_episode_sim.fixtures.stove_knob import burner_on
            return bool(burner_on(q) if state == 'on' else not burner_on(q))
        tolerance = opening_tolerance(sliding=binding['kind']=='drawer')
        return bool(abs(q-(binding['open_position'] if state=='open' else 0.)) < tolerance)

    def registrations(self):
        from cross_episode_sim.skills.atomic import BoundSkill
        from cross_episode_sim.skills.interface import SkillBinding
        c = self.env.controller
        result = {}
        kinds = {v['kind'] for v in self.env.fixture_bindings.values()}
        for kind, noun in [('cabinet-door','door'), ('drawer','drawer')]:
            if kind not in kinds: continue
            for prefix,state in [('open','open'),('close','closed')]:
                name = prefix+'_'+noun
                def execute(target, state=state): self.articulate(target, state)
                def verify(target, state=state): return self.satisfied(target,state)
                def pre(target, kind=kind):
                    return dict(empty_hand=self.env.held() is None,
                        correct_fixture=self.env.fixture_bindings.get(target,{}).get('kind')==kind)
                result[name] = SkillBinding(BoundSkill(c,name,execute,verify),'',{},pre)
        if 'stove-knob' in kinds:
            result['rotate_knob'] = SkillBinding(BoundSkill(c,'rotate_knob',self.articulate,self.satisfied),'',{},
                lambda target,state:dict(empty_hand=self.env.held() is None,
                    correct_fixture=self.env.fixture_bindings.get(target,{}).get('kind')=='stove-knob'))
        if 'microwave' in self.env.fixture_bindings:
            def press(target):
                with self.view('microwave', button=True) as view:
                    press_fixture_button(view, target.removeprefix('microwave_'))
            def pressed(target):
                with self.view('microwave', button=True) as view:
                    button = target.removeprefix('microwave_')
                    return bool(view.turned_on == (button=='start') and not view.contacts(button)[0])
            result['push_button'] = SkillBinding(BoundSkill(c,'push_button',press,pressed),'',{},
                lambda target:dict(empty_hand=self.env.held() is None,
                                   known_button=target in ('microwave_start','microwave_stop')))
        return result
