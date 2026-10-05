import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.control.upright_placement import upright_targets


class UprightPlacementTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[x, y, z] for x in (-.06, .06)
                                  for y in (-.04, .04) for z in (0., .10)])
        self.low = np.array([-1., -.5, 0.])
        self.high = np.array([1., .5, .9])
        self.preferred = np.array([.5, -.40, .9])

    def targets(self, **kw):
        return upright_targets(self.vertices, np.eye(3), self.preferred,
                               self.low, self.high, (0., -1.), np.pi/2, **kw)

    def test_arrival_yaw_preserves_pickup_relative_grasp_and_upright(self):
        targets = self.targets()
        self.assertTrue(targets)
        # A 90-degree change of docking direction rotates the hand/object
        # relationship by 90 degrees without rolling or tipping the mug.
        np.testing.assert_allclose(targets[0][:3, :3],
                                   Rotation.from_euler('z', np.pi/2).as_matrix())
        for pose in targets:
            np.testing.assert_allclose(pose[:3, 2], [0., 0., 1.])
            points = self.vertices @ pose[:3, :3].T + pose[:3, 3]
            self.assertAlmostEqual(points[:, 2].min(), .9)
            self.assertTrue(np.all(points[:, :2].min(0) >= self.low[:2]+.005))
            self.assertTrue(np.all(points[:, :2].max(0) <= self.high[:2]-.005))
        self.assertLessEqual(len(targets), 18)
        self.assertGreater(len({tuple(p[:2, 3]) for p in targets}), 1)

    def test_preserves_native_up_axis_when_asset_local_z_is_not_up(self):
        source = Rotation.from_euler('x', np.pi/2).as_matrix()
        targets = upright_targets(self.vertices, source, self.preferred,
            self.low, self.high, (0., -1.), np.pi/2)
        self.assertTrue(targets)
        for pose in targets:
            np.testing.assert_allclose((pose[:3, :3] @ source.T)[:, 2],
                                       [0., 0., 1.], atol=1e-10)

    def test_skips_occupied_or_reserved_original_slot(self):
        obstacle = (np.array([.44, -.48, .90]), np.array([.56, -.32, 1.1]))
        targets = self.targets(obstacles=[obstacle])
        self.assertTrue(targets)
        self.assertTrue(all(abs(p[0, 3]-.5) > .1 for p in targets))

    def test_sink_cutout_rejects_unsupported_candidates(self):
        self.assertEqual(self.targets(supported=lambda lo, hi: False), [])
        targets = self.targets(supported=lambda lo, hi: lo[0] > .5)
        self.assertTrue(targets)
        self.assertTrue(all(p[0, 3] > .5 for p in targets))

    def test_local_search_can_place_shallower_without_overhang(self):
        self.preferred[1]=-.30
        targets=self.targets()
        self.assertTrue(any(p[1,3]<self.preferred[1]-.03 for p in targets))
        for pose in targets:
            points=self.vertices@pose[:3,:3].T+pose[:3,3]
            self.assertGreaterEqual(points[:,1].min(),self.low[1]+.005)

    def test_recovery_search_leaves_failed_region_but_preserves_support_checks(self):
        targets = self.targets(search_surface=True,
            excluded_regions=[(self.preferred[:2], .35)],
            supported=lambda lo, hi: hi[0] < .1)
        self.assertTrue(targets)
        for pose in targets:
            self.assertGreaterEqual(np.linalg.norm(pose[:2, 3]-self.preferred[:2]), .35)
            points = self.vertices @ pose[:3, :3].T + pose[:3, 3]
            self.assertLess(points[:, 0].max(), .1)
        self.assertEqual(self.targets(search_surface=True,
            supported=lambda lo, hi: False), [])

    def test_recovery_does_not_overlap_other_vessels(self):
        obstacle = (np.array([-.9, -.5, .90]), np.array([.1, .5, 1.2]))
        targets = self.targets(search_surface=True, obstacles=[obstacle],
            excluded_regions=[(self.preferred[:2], .35)])
        for pose in targets:
            points = self.vertices @ pose[:3, :3].T + pose[:3, 3]
            self.assertFalse(np.all(points.min(0)[:2]-.025 < obstacle[1][:2])
                             and np.all(points.max(0)[:2]+.025 > obstacle[0][:2]))

    def test_repeated_recovery_does_not_drift_deeper_into_counter(self):
        first = self.targets(search_surface=True)
        preferred = self.preferred.copy();preferred[1] += .2
        second = upright_targets(self.vertices, np.eye(3), preferred,
            self.low, self.high, (0., -1.), np.pi/2, search_surface=True)
        np.testing.assert_allclose(first[0], second[0])
        points = self.vertices @ first[0][:3, :3].T + first[0][:3, 3]
        self.assertAlmostEqual(points[:, 1].min(), self.low[1]+.025)

    def test_recovery_samples_gap_boundaries_between_other_objects(self):
        obstacles = [(np.array([-1., -.5, .9]), np.array([.30, .5, 1.2])),
                     (np.array([.50, -.5, .9]), np.array([1., .5, 1.2]))]
        targets = self.targets(search_surface=True, obstacles=obstacles)
        self.assertTrue(targets)
        for pose in targets:
            points = self.vertices @ pose[:3, :3].T + pose[:3, 3]
            self.assertGreaterEqual(points[:, 0].min(), .325)
            self.assertLessEqual(points[:, 0].max(), .475)

    def test_wide_island_search_can_use_farther_supported_space(self):
        targets = self.targets(search_surface=True,
                               supported=lambda lo, hi: lo[1] > .1)
        self.assertTrue(targets)
        self.assertTrue(all(p[1, 3] > .1 for p in targets))


if __name__ == '__main__':
    unittest.main()
