"""Transactional cabinet/drawer access over the existing physical primitives."""
import numpy as np

from cross_episode_sim.skills.atomic import AtomicSkill


class FixtureAccessSkill(AtomicSkill):
    def __init__(self, controller, kind, execute_once):
        if kind not in ('cabinet', 'drawer'):
            raise ValueError(f'Unknown fixture: {kind}')
        super().__init__(controller, kind + '_access')
        self.kind = kind
        self.execute_once = execute_once

    def run(self, opening):
        self.validate_arguments(opening=opening)
        if self.verify(opening=opening):
            self.controller.record(fixture_skill=self.name, opening=opening,
                                   goal_already_satisfied=True)
            return
        return super().run(opening=opening)

    def validate_arguments(self, opening):
        if type(opening) is not bool:
            raise ValueError('opening must be boolean')
        c = self.controller
        if getattr(c, 'attached', False) or getattr(c, 'holding_loaf', False):
            raise ValueError('Fixture access requires an empty hand')

    def candidates(self, opening):
        # Preserve the demonstrated dock first, then try small physical offsets.
        # A* and full-body/handle motion checks still validate every candidate.
        if self.kind == 'drawer' and not opening:
            return ((0., 0.), (0., .10), (0., .20), (.04, .10), (-.04, .10))
        return ((0., 0.), (.04, 0.), (-.04, 0.), (0., -.05), (0., .05))

    def execute_candidate(self, candidate, opening):
        c = self.controller
        old = getattr(c, '_trial_fixture_offset', None)
        c._trial_fixture_offset = candidate
        try:
            return self.execute_once(opening)
        finally:
            c._trial_fixture_offset = old

    def verify(self, opening):
        c = self.controller
        q = c.angle()
        reached = (q <= -.32 if opening else abs(q) <= .01) if self.kind == 'drawer' else (
            q >= np.radians(50) if opening else abs(q) <= np.radians(3))
        return bool(reached and not getattr(c, 'articulating', False)
                    and not getattr(c, 'holding_loaf', False) and not getattr(c, 'attached', False))


def fixture_navigate(c, xy, carrying=False, face=None):
    offset = getattr(c, '_trial_fixture_offset', None)
    goal = np.asarray(xy) + (np.asarray(offset) if offset is not None else 0.)
    return c.task_navigate(goal, carrying, face=face)


def fixture_cycle(c, kind, opening, execute_once):
    if not getattr(c, 'speculative_atomic_skills', True):
        return execute_once(opening)
    skill = FixtureAccessSkill(c, kind, execute_once)
    return skill.run(opening=opening)


class StoragePickupSkill(AtomicSkill):
    """An opened storage fixture remains committed while extraction is retried."""
    def __init__(self, controller, docks):
        super().__init__(controller, 'storage_pickup')
        self.docks = list(docks)
        self.physical_failures = 0
        self.limit = getattr(controller, 'max_physical_grasp_attempts', 5)

    def candidates(self):
        for dock in self.docks:
            if self.physical_failures >= self.limit:
                break
            yield dock

    def execute_candidate(self, candidate):
        from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
        c = self.controller
        old = getattr(c, '_trial_storage_dock', None)
        budget = c.max_physical_grasp_attempts
        c._trial_storage_dock = candidate
        c.max_physical_grasp_attempts = 1
        c.report['physical_grasp_attempted'] = False
        try:
            return GraspQualification.pick_payload(c)
        except RuntimeError:
            if c.report.get('physical_grasp_attempted'):
                self.physical_failures += 1
            raise
        finally:
            c._trial_storage_dock = old
            c.max_physical_grasp_attempts = budget

    def verify(self):
        c = self.controller
        return bool(c.holding_loaf and c.report.get('grasp_qualified'))
