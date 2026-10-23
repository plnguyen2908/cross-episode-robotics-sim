import unittest

import numpy as np

from cross_episode_sim.control.contact_surface import clip_contact_surface, object_mesh_triangles


class ContactSurfaceTest(unittest.TestCase):
    def test_box_face_contact_without_vertices_in_pad_window(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <body name="package" pos="0 0 .2"><geom type="box" size=".025 .035 .055"/>
            <geom type="box" size=".1 .1 .1" contype="0" conaffinity="0"/>
            </body></worldbody></mujoco>''')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        triangles = object_mesh_triangles(model, data, {model.body('package').id})
        self.assertEqual(triangles.shape, (12, 3, 3))
        points = clip_contact_surface(triangles, [-.01,-.1,.19], [.01,.1,.21])
        self.assertAlmostEqual(np.ptp(points[:,1]), .07)

    def test_contact_inside_face_without_any_mesh_vertex(self):
        triangles = np.array([[[-1, y, -1], [1, y, -1], [0, y, 1]]
                              for y in (-.01, .01)])
        low, high = np.array([-.02, -.05, -.02]), np.array([.02, .05, .02])
        vertices = triangles.reshape(-1, 3)
        self.assertFalse(np.any(np.all((vertices[:, (0,2)] >= low[[0,2]]) &
                                      (vertices[:, (0,2)] <= high[[0,2]]), axis=1)))
        points = clip_contact_surface(triangles, low, high)
        self.assertAlmostEqual(np.ptp(points[:, 1]), .02)
        self.assertTrue(np.all(points[:, (0,2)] >= low[[0,2]] - 1e-12))
        self.assertTrue(np.all(points[:, (0,2)] <= high[[0,2]] + 1e-12))

    def test_empty_window_stays_empty(self):
        tri = [[[1, 0, 1], [2, 0, 1], [1, 0, 2]]]
        self.assertEqual(len(clip_contact_surface(tri, [-.1]*3, [.1]*3)), 0)

    def test_does_not_bridge_gap_between_disconnected_surfaces(self):
        tri = [[[-2, 0, -1], [-1, 0, -1], [-1, 0, 1]],
               [[1, 0, -1], [2, 0, -1], [1, 0, 1]]]
        self.assertEqual(len(clip_contact_surface(tri, [-.1]*3, [.1]*3)), 0)


if __name__ == '__main__': unittest.main()
