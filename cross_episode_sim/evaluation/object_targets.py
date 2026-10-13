"""Generic source navigation and pickup for scene objects outside task vessels.

Registration is evaluation-local: it never adds distractors to the task goal or
mutates the generated manifest. Failed physical branches use the atomic runner.
"""
import re
from copy import deepcopy

import numpy as np


def object_type(record):
    name = record.get('asset') or record.get('key', '').split('__')[-1]
    if 'RoboTHOR_' in name:
        name = name.removeprefix('RoboTHOR_').split('_')[0]
    name = re.split(r'[_\d\s]', name)[0]
    return name[:1].upper() + name[1:]


class ObjectTargets:
    def __init__(self, controller):
        self.c = controller
        self.records = {}
        for record in controller.manifest.get('scattered_objects', []):
            self.records[record['body']] = dict(record, asset_type=object_type(record))
        for record in controller.manifest['bindings']:
            if record['role'].split('_')[0] not in ('cup', 'bowl'):
                self.records[record['body']] = dict(record, asset_type=object_type(record))
        self.history = []
        self.target = None
        self.dock = None

    def types(self):
        return sorted({r['asset_type'] for r in self.records.values() if r['asset_type']})

    def candidates(self, kind):
        c = self.c
        return sorted((b for b, r in self.records.items() if r['asset_type'].casefold() == kind.casefold()),
                      key=lambda b: float(np.linalg.norm(c.data.body(b).xpos[:2] - c.base_pose()[:2])))

    def _support(self, body):
        c = self.c
        from cross_episode_sim.tasks.floor_objects import FLOORS
        r = self.records[body]
        support = r.get('support') or c.task_supports.get(r.get('source'))
        if support is None:
            support = FLOORS[r['room']]
        c.table_bids.setdefault(support, c.descendants(support))
        return support

    def _register(self, body):
        c = self.c
        if body in c.object_info:
            return
        from cross_episode_sim.tasks.breakfast.scene import read_qualified
        r = self.records[body]
        _, setup, annotation, _ = read_qualified({'key': r['key']}, c.output, body)
        support = self._support(body)
        site = next((k for k, v in c.task_supports.items() if v == support), 'generic_' + body)
        c.task_supports[site] = support
        c.task_outward.setdefault(site, [0., -1.])
        c.object_info[body] = dict(body=body, role='generic_' + body, asset=setup['asset'],
                                  setup=setup, grasp_path=str(annotation), source=site,
                                  destination=site, position=c.data.body(body).xpos.tolist())

    def _navigate(self, body, dock='0'):
        c = self.c
        r = self.records[body]
        point = c.data.body(body).xpos.copy()
        c.tuck_for_navigation()
        c._public_dock_skip = int(dock)
        if r.get('kind') == 'floor_object':
            # Approach a floor object locally, not the far edge of its room slab.
            angle = np.arctan2(c.base_pose()[1] - point[1], c.base_pose()[0] - point[0])
            angle += int(dock) * np.pi / 2
            xy = point[:2] + .55 * np.array([np.cos(angle), np.sin(angle)])
            yaw = float(np.arctan2(point[1] - xy[1], point[0] - xy[0]))
            c._public_navigation_goal = (xy.copy(), yaw)
            c.task_navigate(xy, False, face=yaw)
        else:
            support = self._support(body)
            site = next((k for k, v in c.task_supports.items() if v == support), 'generic_' + body)
            c.task_supports[site] = support
            c.task_outward.setdefault(site, [0., -1.])
            c.navigate_to_site(site, point, False)

    def _nav_ok(self, body, dock='0'):
        c = self.c
        goal = getattr(c, '_public_navigation_goal', None)
        if goal is None:
            return False
        xy, yaw = goal
        base = c.base_pose()
        return bool(np.linalg.norm(base[:2] - xy) < .025 and
                    (yaw is None or abs(np.arctan2(np.sin(base[2]-yaw), np.cos(base[2]-yaw))) < .03))

    def _pick(self, body):
        c = self.c
        self._register(body)
        if c.model.body_jntadr[c.model.body(body).id] < 0:
            raise RuntimeError('Target has no movable joint; enable dynamic clutter when creating the session')
        c.select_object(body)
        c.source = c.destination = self._support(body)
        c.support_bids = c.table_bids[c.source]
        c.table_gids = {g for g in range(c.model.ngeom) if c.model.geom_bodyid[g] in c.support_bids
                        and (c.model.geom_contype[g] or c.model.geom_conaffinity[g])}
        c.initial_object_pose = c.bread_pose().copy()
        c.transfer_start = c.data.joint(c.object_joint).qpos.copy()
        # The separate navigate call already approached this exact body. Suppress
        # pickup's legacy preparation navigation, preserving the fixed base.
        c._public_pickup_ready = c._fixture_pick_prepared = body
        from cross_episode_sim.skills.vessels import pick_vessel
        previous = getattr(c, '_trial_storage_dock', None)
        c._trial_storage_dock = c.base_pose().copy()
        try:
            pick_vessel(c)
        finally:
            c._trial_storage_dock = previous

    def _held(self, body):
        c = self.c
        return bool(c.object_name == body and c.holding_loaf and len(c.contacts()) >= 2)

    def execute(self, name, arguments):
        from cross_episode_sim.skills.atomic import BoundSkill
        from cross_episode_sim.skills.interface import SkillBinding, SkillInterface, parameters
        c = self.c
        def fail(kind):
            return dict(skill=name, arguments=deepcopy(arguments), success=False, error=dict(kind=kind))
        candidates = self.candidates(arguments['asset_type'])
        if not candidates:
            return fail('target_not_found')
        if name == 'navigate':
            self.target = self.dock = None
            if arguments.get('destination', 'source') != 'source':
                return fail('unsupported_destination')
            body = candidates[0]
            skill = BoundSkill(c, name, self._navigate, self._nav_ok)
            args = dict(body=body, dock=arguments.get('dock', '0'))
        elif name == 'pick':
            if self.target not in candidates:
                return fail('target_type_mismatch' if self.target else 'source_navigation_required')
            if arguments.get('source') != 'surface':
                return fail('unsupported_source')
            body = self.target
            if self.dock is None or np.linalg.norm(c.base_pose() - self.dock) > .03:
                return fail('not_at_pickup_dock')
            def attempt_pick(body):
                try:
                    return self._pick(body)
                except Exception:
                    import traceback
                    (c.output / 'generic_pick_error.txt').write_text(traceback.format_exc())
                    raise
            skill = BoundSkill(c, name, attempt_pick, self._held)
            args = dict(body=body)
        else:
            return fail('unsupported_skill_for_target')
        def pre(**kwargs):
            return dict(empty_hand=bool(not c.holding_loaf and not c.attached and not c.contacts()))
        schema = parameters(body=list(self.records))
        if name == 'navigate':
            schema['properties']['dock'] = dict(type='string', enum=['0', '1', '2', '3', '4'])
        class GenericInterface(SkillInterface):
            def _save(self):
                from cross_episode_sim.evaluation.dataset import write_json
                write_json(self.controller.output / 'generic_skill_calls.json', self.history)

        api = GenericInterface(c, {name: SkillBinding(skill, 'Generic object attempt', schema, pre, name == 'navigate')})
        api.history = self.history
        result = api.execute_skill(name, args)
        if result['success'] and name == 'navigate':
            self.target, self.dock = body, c.base_pose().copy()
        result['arguments'] = deepcopy(arguments)
        return result
