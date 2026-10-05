"""Controller-level regression: reachable release must not require high staging."""
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode


class BreakfastPlacementTest(unittest.TestCase):
    def test_loaded_high_counter_docks_include_close_reachable_positions(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.task_outward = {'kitchen': [0., -1.]}
        c.task_supports = {'kitchen': 'counter'}
        c.model = c.data = None
        c.room_id = lambda xy: 'kitchen'
        with patch('cross_episode_sim.tasks.breakfast.episode.support_bounds',
                   return_value=(np.array([.25, -.65, .89]), np.array([2.75, 0., .92]))):
            docks = list(c.dock_candidates('kitchen', np.array([2., -.55, .97]), pickup=False))
        np.testing.assert_allclose(docks[0], [2., -.95, np.pi/2])
        np.testing.assert_allclose(docks[1], [2., -.91, np.pi/2])

    def runner(self, blocked=False, flat=False):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        # This fixture mocks all physics. Real settling is tested separately
        # with a MuJoCo servo in test_motion_timing.py.
        c.adaptive_motion_settling = False
        c.args = SimpleNamespace(motion_slowdown=1.)
        c.annotation_asset = 'test_vessel'
        c.destination_pose = np.eye(4)
        c.destination_pose[:3, 3] = [2.56, -.57, .98]
        c.placement_pose_options = []
        c.grasp_relative = np.eye(4)
        c.destination = 'counter'
        c.object_name = 'vessel'
        c.object_joint = 'vessel_joint'
        c.holding_loaf = c.attached = True
        c.report = {}
        c.placement_history = defaultdict(list)
        c.events = []
        c.record = lambda **kw: None
        c.tick = lambda seconds: None
        c.base_pose = lambda: np.array([2.56, -1.03, np.pi/2])
        start = np.eye(4)
        start[:3, 3] = [2.56, -.77, 1.17]
        c.tcp = lambda: start.copy()
        c.assignment = lambda: {'vessel': 'counter'}
        c.data = SimpleNamespace(
            joint=lambda name: SimpleNamespace(qvel=np.zeros(6)),
            body=lambda name: SimpleNamespace(xpos=c.destination_pose[:3, 3]))
        c.embodiment = SimpleNamespace(open_gripper=lambda runner: c.events.append('open'))
        c.planner = SimpleNamespace(detach_block=lambda: c.events.append('detach'))
        c.in_default_travel_posture = lambda loaded: True
        c.redock_loaded_for_placement = lambda: c.events.append('redock')
        c.tuck_after_placement = lambda: c.events.append('tuck')
        c.mesh_contact_move = lambda stage, pose: c.events.append(stage)

        def move(stage, pose):
            c.events.append((stage, pose.copy()))
            if stage == 'above destination table' and not flat:
                raise RuntimeError('cuRobo failed to plan unreachable high staging')
            if stage == 'approach destination table' and blocked:
                raise RuntimeError('cuRobo failed to plan loaded release approach')
        c.move = move
        if flat:
            c._edge_flat_rotation = np.eye(3)
        return c, start

    def test_upright_release_and_retreat_without_unreachable_high_waypoint(self):
        c, start = self.runner()
        c.place_payload()
        moves = [e for e in c.events if isinstance(e, tuple)]
        self.assertEqual([e[0] for e in moves],
                         ['approach destination table', 'withdraw from released table object'])
        self.assertAlmostEqual(moves[0][1][2, 3], 1.02)  # 4 cm release gap
        np.testing.assert_allclose(moves[1][1], start)
        self.assertEqual(c.events[1:3], ['open', 'detach'])
        self.assertTrue(c.report['success'])
        self.assertFalse(c.holding_loaf or c.attached)

    def test_failed_loaded_approaches_never_open_gripper(self):
        c, _ = self.runner(blocked=True)
        with self.assertRaisesRegex(RuntimeError, 'No placement.*5 docks'):
            c.place_payload()
        self.assertNotIn('open', c.events)
        self.assertNotIn('detach', c.events)
        self.assertTrue(c.holding_loaf and c.attached)
        self.assertEqual(c.args.motion_slowdown, 1.)

    def test_flat_book_retains_staged_approach_and_checked_descent(self):
        c, _ = self.runner(flat=True)
        c.place_payload()
        names = [e[0] if isinstance(e, tuple) else e for e in c.events]
        self.assertEqual(names[:4], ['above destination table', 'approach destination table',
                                    'lower object onto destination table', 'open'])

    def test_low_staging_recovers_a_blocked_direct_branch_before_release(self):
        c, _ = self.runner()
        original = c.move
        attempted = []
        def move(stage, pose):
            attempted.append(stage)
            if len(attempted) == 1:
                raise RuntimeError('cuRobo failed to plan direct branch')
            if stage == 'above destination table':
                self.assertAlmostEqual(pose[2, 3], 1.08)
                self.assertLess(pose[1, 3], -.57)
                return
            original(stage, pose)
        c.move = move
        c.place_payload()
        self.assertEqual(attempted[:3], ['approach destination table',
                         'above destination table', 'approach destination table'])
        self.assertIn('open', c.events)

    def test_execution_failure_does_not_trigger_staging_or_release(self):
        c, _ = self.runner()
        def move(stage, pose):
            raise RuntimeError('Object dropped after clearing pickup support')
        c.move = move
        with self.assertRaisesRegex(RuntimeError, 'Object dropped'):
            c.place_payload()
        self.assertFalse(getattr(c, '_placement_staging_attempted', False))
        self.assertNotIn('open', c.events)

    def test_gather_relocates_filling_site_after_local_failure(self):
        c, _ = self.runner()
        c.phase = 'GATHER';c._redock_index = 0;c._failed_placement_regions = []
        c.model = c.data = None
        c.task_supports = {'kitchen': 'counter'}
        c.task_outward = {'kitchen': [0., -1.]}
        info = dict(destination='kitchen', destination_position=[2.56, -.57, .98])
        c.object_info = {'vessel': info}
        c.tuck_loaded_for_navigation = lambda: None
        c.dock_candidates = lambda *args, **kw: iter(())
        c.navigate_to_site = lambda room, point, loaded: c.events.append(('nav', point.copy()))
        def prepare():
            if c._search_whole_placement_surface:
                c.destination_pose[:3, 3] = [1., -.57, .98]
        c.prepare_destination = prepare
        with patch('cross_episode_sim.tasks.breakfast.episode.support_bounds',
                   return_value=(np.array([.25, -.65, .89]), np.array([2.75, 0., .92]))):
            BreakfastEpisode.redock_loaded_for_placement(c)
        self.assertEqual(info['destination_position'], [1., -.57, .98])
        self.assertEqual(info['filling_position'], [1., -.57, .98])
        self.assertEqual(c._failed_placement_regions, [([2.56, -.57], .35)])
        self.assertFalse(c._search_whole_placement_surface)

    def test_gather_retries_stance_before_discarding_supported_spot(self):
        c, _ = self.runner()
        c.phase='GATHER';c._redock_index=0
        c.object_info={'vessel':dict(destination='kitchen', destination_position=[2.56,-.57,.98])}
        c.tuck_loaded_for_navigation=lambda:None
        new=np.array([2.56,-.99,np.pi/2])
        c.dock_candidates=lambda *args, **kw:iter((c.base_pose(),new))
        c.plan_route=lambda *args, **kw:[new]
        c.task_navigate=lambda xy, loaded, face:c.events.append(('dock',xy.copy()))
        c.prepare_destination=lambda:c.events.append('prepare')
        c.relocate_filling_placement=lambda:self.fail('Relocated before trying the other stance')
        BreakfastEpisode.redock_loaded_for_placement(c)
        np.testing.assert_allclose(c.events[0][1],new[:2])
        self.assertEqual(c.events[1],'prepare')

    def test_vessel_can_use_lower_curobo_release_when_high_goal_is_unreachable(self):
        c, _ = self.runner()
        c._allow_lower_vessel_release=True
        c.object_info={'vessel':dict(vessel_type='cup')}
        candidate=np.eye(4);candidate[2,3]=1.04
        heights=[]
        def move(stage, pose):
            heights.append(pose[2,3])
            if pose[2,3]>1.006:raise RuntimeError('cuRobo failed to plan')
        c.move=move
        def blocked(*args):raise RuntimeError('Actual-mesh collision in contact move')
        c.plan_contact_path=blocked
        BreakfastEpisode.approach_table_placement(c,candidate,candidate)
        np.testing.assert_allclose(heights,[1.04,1.02,1.005])
        self.assertNotIn('open',c.events)

    def test_standard_release_candidates_precede_lower_release_fallback(self):
        c, _ = self.runner()
        alternate=c.destination_pose.copy();alternate[0,3]+=.1
        c.placement_pose_options=[alternate]
        attempts=[]
        def approach(candidate, staging):
            attempts.append(c._allow_lower_vessel_release)
            if not c._allow_lower_vessel_release:
                raise RuntimeError('cuRobo failed to plan')
            return c.tcp()
        c.approach_table_placement=approach
        c.place_payload()
        self.assertEqual(attempts,[False,False,True])

    def test_serving_preserves_assigned_place_setting(self):
        c, _ = self.runner()
        c.phase = 'SERVE';c._redock_index = 0
        info = dict(destination='dining', destination_position=[2.56, -.57, .98])
        c.object_info = {'vessel': info}
        c.tuck_loaded_for_navigation = lambda: None
        c.navigate_to_site = lambda room, point, loaded, skip: c.events.append((room, point, skip))
        c.prepare_destination = lambda: None
        BreakfastEpisode.redock_loaded_for_placement(c)
        self.assertEqual(info['destination_position'], [2.56, -.57, .98])
        self.assertEqual(c.events, [('dining', [2.56, -.57, .98], 1)])


if __name__ == '__main__':
    unittest.main()
