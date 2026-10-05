"""Task docks and bounded family fallback for breakfast pickup."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from cross_episode_sim.manipulation.robocasa import RoboCasaManipulation
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.recovery import RecoveringManipulation
from cross_episode_sim.controller.manipulation import TableReorder


class BreakfastGraspRecoveryTest(unittest.TestCase):
    def test_above_rim_grasp_is_not_misclassified_as_side_grasp(self):
        from scipy.spatial.transform import Rotation
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(annotation_source='qualified_registry', approach_policy='any-above-table')
        for degrees, allowed in ((22., True), (27., True), (74., False)):
            pose = np.eye(4)
            pose[:3,:3] = Rotation.from_euler('y',180-degrees,degrees=True).as_matrix()
            self.assertEqual(c.annotation_approach_allowed(pose), allowed)

    def test_all_top_down_docks_precede_diagonal_search(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(annotation_source='qualified_registry')
        c.local_annotations = np.eye(4)[None];c.annotation_path='saved.npz'
        c.annotation_vertical_offsets=(0.,);c.physically_rejected_annotation_variants=set()
        c.record=lambda **kw:None
        phases=[]
        def search():
            phases.append((c._top_down_dock_search,getattr(c,'_qualified_angles_only',False)))
            if len(phases)==1:
                c.args.annotation_source='droid';c._recovery_exhausted=True
                raise RuntimeError('No saved grasp plan after bounded alternate navigation stances')
            self.assertEqual(c.args.annotation_source,'qualified_registry')
            self.assertFalse(c._recovery_exhausted)
            return 'diagonal'
        with patch.object(RecoveringManipulation,'select_annotated_grasp',side_effect=search):
            self.assertEqual(c.select_annotated_grasp(),'diagonal')
        self.assertEqual(phases,[(True,False),(False,True)])

    def test_top_down_success_never_starts_diagonal_pass(self):
        c=BreakfastEpisode.__new__(BreakfastEpisode)
        c.args=SimpleNamespace(annotation_source='qualified_registry')
        c.local_annotations=np.eye(4)[None];c.annotation_path='saved.npz'
        c.annotation_vertical_offsets=(0.,);c.physically_rejected_annotation_variants=set()
        with patch.object(RecoveringManipulation,'select_annotated_grasp',return_value='top') as plan:
            self.assertEqual(c.select_annotated_grasp(),'top')
        plan.assert_called_once()
        self.assertFalse(c._top_down_dock_search)

    def test_blocked_planner_start_skips_goal_sampling(self):
        c = RecoveringManipulation.__new__(RecoveringManipulation)
        c.planner_world_boxes = ['counter']
        c.planner = SimpleNamespace(names=[], world_clearance=lambda q, boxes: -.002)
        c.record = lambda **kw: None
        with patch.object(GraspQualification, 'select_annotated_grasp') as sample:
            with self.assertRaisesRegex(RuntimeError, 'No annotated grasp'):
                c._plan_with_grasp_fallback()
        sample.assert_not_called()

    def test_new_dock_restarts_saved_library_after_raw_fallback(self):
        c = RecoveringManipulation.__new__(RecoveringManipulation)
        c.args = SimpleNamespace(annotation_source='qualified_registry')
        c.report = {};c.local_annotations = np.eye(4)[None]
        saved = c.local_annotations.copy()
        c.annotation_path = 'saved.npz';c.annotation_vertical_offsets = (0.,)
        c.physically_rejected_annotation_variants = {3}
        c.base_pose = lambda: np.zeros(3)
        c.tuck_for_navigation = lambda: None
        c.grasp_recovery_stances = lambda: iter((np.array([.1, 0., 0.]),))
        c.navigate = lambda *a, **kw: None
        c.record = lambda **kw: None
        c.make_planner = lambda: SimpleNamespace(names=[])
        c.actuator_ids = lambda names: []
        c.load_world = lambda: None
        attempts = []
        def plan():
            attempts.append(c.args.annotation_source)
            if len(attempts) == 1:
                c.args.annotation_source = 'droid'
                c.local_annotations = np.zeros((2, 4, 4))
                c.annotation_path = 'raw.npz';c.annotation_vertical_offsets = (0., -.03)
                c.physically_rejected_annotation_variants = set()
                raise RuntimeError('No annotated grasp passed robot reachability and collision checks')
            np.testing.assert_array_equal(c.local_annotations, saved)
            self.assertEqual(c.annotation_path, 'saved.npz')
            self.assertEqual(c.annotation_vertical_offsets, (0.,))
            self.assertEqual(c.physically_rejected_annotation_variants, {3})
            return 'planned'
        c._plan_with_grasp_fallback = plan
        self.assertEqual(c.select_annotated_grasp(), 'planned')
        self.assertEqual(attempts, ['qualified_registry', 'qualified_registry'])

    def test_all_saved_angles_rejected_retry_before_renavigation(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(annotation_source='qualified_registry', approach_policy='any-above-table')
        c.report = {'annotation_selection': {'tested_orientation_variants':16,
                    'filter_rejections': {'approach':16}}}
        c.record = lambda **kw: None
        error = RuntimeError('No annotated grasp passed robot reachability and collision checks')
        with patch.object(RecoveringManipulation, '_plan_qualified_grasp', side_effect=[error, 'planned']) as plan:
            self.assertEqual(c._plan_qualified_grasp(), 'planned')
            self.assertEqual(plan.call_count, 2)
        self.assertFalse(c._allow_qualified_angled_grasps)
        c._allow_qualified_angled_grasps = True
        pose = np.eye(4);pose[:3,:3] = [[1,0,0],[0,0,-1],[0,1,0]]
        self.assertTrue(c.annotation_approach_allowed(pose))
        c.args.annotation_source = 'droid'
        self.assertFalse(c.annotation_approach_allowed(pose))

    def test_failed_top_plans_try_saved_angled_grasps_before_raw_library(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(annotation_source='qualified_registry')
        c.report = {'annotation_selection': {'tested_orientation_variants':16,
                    'filter_rejections': {'approach':12}}}
        c.record = lambda **kw: None
        error = RuntimeError('No annotated grasp passed robot reachability and collision checks')
        with patch.object(RecoveringManipulation, '_plan_qualified_grasp', side_effect=error) as plan:
            with self.assertRaisesRegex(RuntimeError, 'No annotated grasp'):
                c._plan_qualified_grasp()
            self.assertEqual(plan.call_count, 2)
        self.assertFalse(getattr(c, '_allow_qualified_angled_grasps', False))

    def test_depth_refinement_is_only_enabled_after_saved_library_fails(self):
        c = RecoveringManipulation.__new__(RecoveringManipulation)
        c.args = SimpleNamespace(annotation_source='qualified_registry')
        c.annotation_asset = 'example'
        c.asset_metadata = {'source': 'molmo'}
        c.annotation_vertical_offsets = (0.,)
        c.annotation_recovery_vertical_offsets = (0., -.015, -.030, -.045)
        c.record = lambda **kw: None
        with patch.object(GraspQualification, 'select_annotated_grasp', return_value='saved'):
            self.assertEqual(c._plan_with_grasp_fallback(), 'saved')
        self.assertEqual(c.annotation_vertical_offsets, (0.,))
        error = RuntimeError('No annotated grasp passed robot reachability and collision checks')
        with patch.object(GraspQualification, 'select_annotated_grasp', side_effect=[error, 'raw']):
            with patch('pathlib.Path.is_file', return_value=True):
                with patch('numpy.load', return_value={'transforms': np.eye(4)[None]}):
                    self.assertEqual(c._plan_with_grasp_fallback(), 'raw')
        self.assertEqual(c.annotation_vertical_offsets, c.annotation_recovery_vertical_offsets)
        self.assertEqual(c.args.annotation_source, 'droid')

    def test_saved_angled_success_precedes_raw_library_fallback(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(annotation_source='qualified_registry')
        c.report = {'annotation_selection': {'tested_orientation_variants':16,
                    'filter_rejections': {'approach':12}}}
        c.record = lambda **kw: None
        error = RuntimeError('No annotated grasp passed robot reachability and collision checks')
        with patch.object(GraspQualification, 'select_annotated_grasp', side_effect=[error, 'angled']):
            self.assertEqual(c._plan_with_grasp_fallback(), 'angled')
        self.assertEqual(c.args.annotation_source, 'qualified_registry')
        self.assertFalse(c._allow_qualified_angled_grasps)

    def test_angled_retry_does_not_retest_failed_top_down_candidates(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.args = SimpleNamespace(annotation_source='qualified_registry', approach_policy='any-above-table')
        c._allow_qualified_angled_grasps = True
        pose = np.diag([1., -1., -1., 1.])
        self.assertFalse(c.annotation_approach_allowed(pose))

    def test_unrelated_error_does_not_enable_fallback(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        with patch.object(RecoveringManipulation, '_plan_qualified_grasp', side_effect=RuntimeError('Object dropped')) as plan:
            with self.assertRaisesRegex(RuntimeError, 'Object dropped'):
                c._plan_qualified_grasp()
        self.assertEqual(plan.call_count, 1)

    def test_exhausted_any_family_does_not_repeat_identical_search(self):
        c = GraspQualification.__new__(GraspQualification)
        c.active_family = 'any'
        c.forced_family = 'auto'
        c.record = lambda **kw: None
        error = RuntimeError('No annotated grasp passed robot reachability and collision checks')
        with patch.object(RoboCasaManipulation, 'select_annotated_grasp', side_effect=error) as pick:
            with self.assertRaisesRegex(RuntimeError, 'No annotated grasp'):
                c.select_annotated_grasp()
        self.assertEqual(pick.call_count, 1)

    def test_preferred_family_can_still_fall_back_once(self):
        c = GraspQualification.__new__(GraspQualification)
        c.active_family = 'top'
        c.forced_family = 'auto'
        c.record = lambda **kw: None
        error = RuntimeError('No annotated grasp passed robot reachability and collision checks')
        with patch.object(RoboCasaManipulation, 'select_annotated_grasp', side_effect=error) as pick:
            with self.assertRaisesRegex(RuntimeError, 'No annotated grasp'):
                c.select_annotated_grasp()
        self.assertEqual(pick.call_count, 2)

    def test_breakfast_tries_remaining_close_docks_before_radial_search(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.object_name = 'object'
        c.object_info = {'object': {'source': 'kitchen'}}
        c.base_pose = lambda: np.array([.1, -.95, np.pi/2])
        c.bread_pose = lambda: np.eye(4)
        c.dock_candidates = lambda *a, **kw: iter((c.base_pose(), np.array([.1, -.91, np.pi/2])))
        candidate = next(c.grasp_recovery_stances())
        np.testing.assert_allclose(candidate, [.1, -.91, np.pi/2])

    def test_blocked_docking_turn_retries_with_checked_reverse(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.base_pose = lambda: np.zeros(3)
        c.room_id = lambda xy: 2
        c.record = lambda **kw: None
        def route(self, *a):
            self.assertion = self._pickup_pre_nav_undock
            return [np.zeros(3), np.array([-.15, 0., 0.])]
        with patch.object(CrossRoomManipulation, 'plan_route', side_effect=RuntimeError('turn blocked')):
            with patch.object(TableReorder, 'plan_route', route):
                path = c.plan_route([.1, 0.], False)
        self.assertEqual(c.assertion, .15)
        c._accepted_route = path
        # Consuming the cache must preserve the reverse flag used by execution.
        result = c.plan_route([.1, 0.], False)
        self.assertIs(result, path)
        self.assertFalse(hasattr(c, '_accepted_route'))
        self.assertEqual(c._pickup_pre_nav_undock, .15)

    def test_failed_reverse_does_not_disable_route_collision_checks(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.base_pose = lambda: np.zeros(3)
        c.room_id = lambda xy: 2
        with patch.object(CrossRoomManipulation, 'plan_route', side_effect=RuntimeError('turn blocked')):
            with patch.object(TableReorder, 'plan_route', side_effect=RuntimeError('reverse blocked')):
                with self.assertRaisesRegex(RuntimeError, 'turn blocked'):
                    c.plan_route([.1, 0.], False)
        self.assertEqual(c._pickup_pre_nav_undock, 0.)


if __name__ == '__main__':
    unittest.main()
