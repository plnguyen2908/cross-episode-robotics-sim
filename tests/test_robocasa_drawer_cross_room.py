import unittest
from unittest.mock import Mock, patch
import numpy as np
from cross_episode_sim.fixtures.drawer_cross_room import DrawerCrossRoom
from cross_episode_sim.fixtures.drawer import DRAWER
from cross_episode_sim.fixtures.drawer_pick_place import DrawerPickPlace

class DrawerTransferTest(unittest.TestCase):
    def test_return_carry_tucks_before_navigation_and_uses_table_surface(self):
        check = DrawerCrossRoom.__new__(DrawerCrossRoom)
        check.source = DRAWER
        check.dining = check.destination = 'dining'
        check.saved_table_pose = np.eye(4)
        check.saved_table_pose[:3, 3] = [2.6, -6., .8]
        check.table_bids = {'dining': {7}}
        calls = []
        check.tuck_loaded_for_navigation = lambda: calls.append('tuck')
        check.task_navigate = lambda xy, loaded, face: calls.append(('nav', list(xy), loaded))
        check.prepare_loaded_manipulation = lambda: calls.append('prepare')
        check.surface_boxes = lambda name: [(12, np.array([2., -7., .7]), np.array([3., -5., .8]))]
        check.bread_pose = lambda: np.eye(4)
        check.bread_vertices = lambda: np.array([[0., 0., -.03], [.03, .03, .03]])
        check.transport_payload()
        self.assertEqual(calls[0], 'tuck')
        self.assertEqual(calls[1], ('nav', [2.6, -5.58], True))
        self.assertEqual(check.table_gids, {12})
        np.testing.assert_allclose(check.destination_pose[:3, 3], [2.6, -6., .83])

    def test_completion_uses_measured_route_instead_of_unset_flag(self):
        route = dict(navigation=[dict(carrying=True, measured_distance_m=6., endpoint_error_m=.001)])
        self.assertTrue(DrawerCrossRoom.completed_cross_room_route(route, 2, 3))
        self.assertFalse(DrawerCrossRoom.completed_cross_room_route(route, 2, 2))
        self.assertFalse(DrawerCrossRoom.completed_cross_room_route({}, 2, 3))

    def test_refresh_uses_shared_preparation_and_measured_hold(self):
        check = DrawerCrossRoom.__new__(DrawerCrossRoom)
        tool = np.eye(4); tool[0, 3] = 1.
        obj = tool.copy(); obj[2, 3] = .04
        check.tcp = lambda: tool
        check.bread_pose = lambda: obj
        with patch.object(DrawerPickPlace, 'prepare_loaded_manipulation', Mock()) as prepare:
            check.prepare_loaded_manipulation()
            prepare.assert_called_once()
        np.testing.assert_allclose(check.grasp_relative[:3, 3], [0., 0., .04])

if __name__ == '__main__':
    unittest.main()
