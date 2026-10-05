"""Exercise real annotation selection with a planner that rejects a high approach."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from cross_episode_sim.controller.annotated_grasp import AnnotatedGraspMixin
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode


class PregraspStandoffTest(unittest.TestCase):
    def selector(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="robot_0/pad"><joint name="robot_0/arm" type="hinge" range="-2 2"/>
            <geom type="box" size=".02 .02 .02" mass=".1"/>
          </body></worldbody><compiler angle="radian"/></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pose = np.diag([1., -1., -1., 1.])
        goals, probes, lifts = [], [], []

        def plan(positions, goal):
            goals.append(goal)
            if goal[2] > .03:
                raise RuntimeError('cuRobo failed to plan high pregrasp')
            return np.zeros((3, 1))

        c = SimpleNamespace(model=model, data=data, annotation_asset='test', annotation_path='test.npz',
            local_annotations=np.array([pose]), annotation_standoff=.06, annotation_standoffs=(.06, .025),
            annotation_candidate_budget=1, annotation_mesh_contact_approach=True,
            bread_pose=lambda: np.eye(4), tcp=lambda: np.eye(4),
            bread_vertices=lambda: np.array([[-.01, -.015, 0.], [.01, .015, 0.]]),
            is_finger=lambda name: name.endswith('pad'),
            profile=SimpleNamespace(name='test', grasp_width_mode='whole_object',
                minimum_contact_width_m=.006, gripper_stroke_m=.085, preferred_contact_width_m=.03),
            args=SimpleNamespace(grip_open=.04), report={}, record=lambda **kw: None,
            planner=SimpleNamespace(names=['arm'], plan=plan),
            embodiment=SimpleNamespace(planner_tool_offset=lambda: 0., probe_aperture=lambda *a: None),
            nearby_ik=lambda target, q, **kw: [0.],
            validate_grasp_probe=lambda probe: probes.append(True),
            validate_grasp_lift=lambda joints, aperture: lifts.append(True))
        return c, goals, probes, lifts

    def test_shorter_approach_is_checked_and_cached_before_selection(self):
        c, goals, probes, lifts = self.selector()
        grasp, pre = AnnotatedGraspMixin.select_annotated_grasp(c)
        np.testing.assert_allclose([g[2] for g in goals], [.06, .025])
        self.assertEqual(c.report['annotation_selection']['selected_index'], 0)
        self.assertEqual(c.report['annotation_selection']['planning_attempts'], 2)
        self.assertEqual(len(probes), 22)  # 19 contact samples plus 3 free-space waypoints
        self.assertEqual(lifts, [True])
        self.assertAlmostEqual(pre[2, 3], .025)
        np.testing.assert_allclose(c.preplanned_moves['pregrasp'][0], pre)
        np.testing.assert_allclose(c.preplanned_moves['grasp approach 3/3'][0], grasp)

    def test_shorter_approach_does_not_bypass_contact_rejection(self):
        c, _, _, _ = self.selector()
        def blocked(probe):
            raise RuntimeError('Open hand/arm collision')
        c.validate_grasp_probe = blocked
        with self.assertRaisesRegex(RuntimeError, 'No annotated grasp passed'):
            AnnotatedGraspMixin.select_annotated_grasp(c)
        self.assertFalse(hasattr(c, 'preplanned_moves'))

    def test_default_standoff_behavior_is_unchanged(self):
        c, goals, _, _ = self.selector()
        del c.annotation_standoffs
        with self.assertRaisesRegex(RuntimeError, 'No annotated grasp passed'):
            AnnotatedGraspMixin.select_annotated_grasp(c)
        self.assertEqual(len(goals), 1)

    def test_reach_ranking_uses_near_contact_without_bypassing_checks(self):
        c, goals, probes, lifts = self.selector()
        far, near = c.local_annotations[0].copy(), c.local_annotations[0].copy()
        far[0, 3], near[0, 3] = .20, .10
        c.local_annotations = np.array([far, near])
        c.annotation_reach_weight = .5
        c.base_pose = lambda: np.zeros(3)
        AnnotatedGraspMixin.select_annotated_grasp(c)
        self.assertEqual(c.report['annotation_selection']['selected_index'], 1)
        self.assertTrue(all(abs(g[0]-.10) < 1e-8 for g in goals))
        self.assertEqual(len(probes), 22)
        self.assertEqual(lifts, [True])

    def test_closer_high_counter_docks_serve_pickup_and_release(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.model = c.data = None
        c.room_id = lambda xy: 2
        with patch('cross_episode_sim.tasks.breakfast.episode.support_bounds',
                   return_value=(np.array([0., -.65, 0.]), np.array([3., 0., .92]))):
            pickup = next(c.dock_candidates('kitchen', [.43, -.57, 1.], pickup=True))
            loaded = next(c.dock_candidates('kitchen', [.43, -.57, 1.]))
        # High counters put the base close for both pickup and loaded release.
        np.testing.assert_allclose(pickup, loaded)
        self.assertAlmostEqual(pickup[1], -.65 - .30)  # 30 cm out from the counter edge

    def test_high_counter_lift_policy_resets_for_next_object(self):
        c = BreakfastEpisode.__new__(BreakfastEpisode)
        c.model = c.data = None
        c.object_name = 'mug'
        c.report = {}
        c.object_info = {'mug': {'source': 'kitchen', 'role': 'cup_one'},
                         'book': {'setup': {}}}
        c.profile = SimpleNamespace(pregrasp_standoff_m=.06)
        c.args = SimpleNamespace(lift_height=.12, lift_retreat=0.)
        c._default_pickup_lift_height = .12
        c.tuck_for_navigation = lambda: None
        c.edge_fallback_eligible = lambda: False
        c.bread_pose = lambda: np.eye(4)
        c.record = lambda **kw: None
        c.navigate_to_site = lambda *a: None
        with patch('cross_episode_sim.tasks.breakfast.episode.support_bounds',
                   return_value=(np.array([0., -.65, 0.]), np.array([3., 0., .92]))):
            c.prepare_pickup()
        self.assertEqual(c.annotation_standoffs, (.06, .025))
        self.assertEqual(c.initial_lift_height, .025)
        self.assertEqual(c.args.lift_height, .105)
        self.assertEqual(c.args.lift_retreat, .10)
        self.assertEqual(c.annotation_reach_weight, .5)
        self.assertEqual(c.annotation_recovery_vertical_offsets, (0., -.015, -.030, -.045))
        self.assertEqual(c.minimum_pickup_lift_m, .05)
        c._allow_qualified_angled_grasps = True
        with patch('cross_episode_sim.tasks.breakfast.episode.CrossRoomManipulation.select_object'):
            c.select_object('book')
        self.assertFalse(c._allow_qualified_angled_grasps)
        self.assertFalse(hasattr(c, 'annotation_standoffs'))
        self.assertEqual(c.args.lift_height, .12)
        self.assertEqual(c.args.lift_retreat, 0.)
        self.assertEqual(c.annotation_reach_weight, 0.)
        self.assertEqual(c.annotation_recovery_vertical_offsets, (0.,))
        self.assertEqual(c.minimum_pickup_lift_m, .10)
        self.assertEqual(c.report['qualification_protocol']['lift_m'], .10)


if __name__ == '__main__':
    unittest.main()
