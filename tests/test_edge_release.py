import unittest
import numpy as np
from cross_episode_sim.control.edge_release import flat_edge_targets


class FlatEdgeReleaseTest(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[x,y,z] for x in (-.12,.12)
                                for y in (-.08,.08) for z in (-.01,.01)])
        self.low=np.array([9.4,-4.425,.0]);self.high=np.array([10.05,-3.775,.78])
        self.rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])

    def test_flat_object_faces_new_edge_and_keeps_supported_com(self):
        candidates=flat_edge_targets(self.vertices,self.rotation,[9.52,-4.2,.80],
            self.low,self.high,[-1.,0.],[0.,0.,0.])
        self.assertTrue(candidates)
        for pose in candidates:
            np.testing.assert_allclose(pose[:3,:3],self.rotation)
            points=self.vertices@pose[:3,:3].T+pose[:3,3]
            self.assertAlmostEqual(points[:,2].min(),.78)
            self.assertGreaterEqual(pose[0,3],self.low[0]+.03)
            overhang=max(0.,self.low[0]-points[:,0].min())
            self.assertLessEqual(overhang,.35*np.ptp(points[:,0])+1e-10)

    def test_rejects_object_too_wide_along_table_edge(self):
        vertices=self.vertices.copy();vertices[:,0]*=4
        self.assertEqual(flat_edge_targets(vertices,self.rotation,[9.52,-4.2,.80],
            self.low,self.high,[-1.,0.],[0.,0.,0.]),[])

    def test_no_candidate_if_actual_com_would_be_outside_support(self):
        self.assertEqual(flat_edge_targets(self.vertices,self.rotation,[9.52,-4.2,.80],
            self.low,self.high,[-1.,0.],[0.,1.,0.]),[])


if __name__=='__main__':unittest.main()
