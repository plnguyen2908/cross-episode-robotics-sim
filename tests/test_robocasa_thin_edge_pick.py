"""Thin edge access: support, rotated outlines, and native child geometry."""
import itertools
from pathlib import Path
import tempfile
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.manipulation.thin_edge_pick import (
    edge_candidates,side_annotations,outline_interval,install_thin_scene,object_bodies)
from cross_episode_sim.fixtures.toaster_insertion import collision_vertices


def box(center,half):
    return np.asarray(list(itertools.product((-1,1),repeat=3)))*half+center


class ThinEdgeTests(unittest.TestCase):
    def test_edge_leaves_center_supported_for_rotated_table(self):
        table=box(np.array([0.,0.,.8]),np.array([.5,.5,.02]))
        obj=box(np.array([0.,.40,.831]),np.array([.07,.05,.01]))
        for yaw in (0.,37.,90.,-60.):
            R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            edges=edge_candidates(obj@R.T,table@R.T,(R@np.array([0.,.9,0.]))[:2])
            e=edges[0];final=obj@R.T;final[:,:2]+=e.normal*e.distance
            self.assertAlmostEqual(float((final[:,:2]@e.normal).max()-e.boundary),e.overhang)
            self.assertGreater(e.boundary-float(final[:,:2].mean(0)@e.normal),.008)
            self.assertLessEqual(e.distance,.30)

    def test_rotated_outline_contact_is_not_empty_bounding_box_corner(self):
        v=box(np.zeros(3),np.array([.075,.095,.008]))
        R=Rotation.from_euler('z',25,degrees=True).as_matrix();v=v@R.T
        rear,front=outline_interval(v[:,:2],0.)
        self.assertLess(front,v[:,1].max()-.01)
        poses=side_annotations(v,np.eye(4),[0.,1.])
        self.assertGreaterEqual(len(poses),5)
        for index,pose in enumerate(poses):
            center=pose[:3,3]+.008*pose[:3,2]
            a,b=outline_interval(v[:,:2],center[0])
            self.assertGreater(center[1],a);self.assertLess(center[1],b)
            np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-8)

    def test_rotated_object_gets_jaw_clearance_not_only_corner_overhang(self):
        table=box(np.array([0.,0.,.8]),np.array([.5,.5,.02]))
        R=Rotation.from_euler('z',12,degrees=True).as_matrix()
        obj=box(np.zeros(3),np.array([.075,.095,.008]))@R.T
        obj+=np.array([0.,.44-obj[:,1].max(),.829])
        e=edge_candidates(obj,table,[0.,.9])[0]
        n=obj[:,:2]@e.normal;t=obj[:,:2]@e.tangent
        for dx in (0.,.012,-.012):
            _,front=outline_interval(np.column_stack((t,n)),(t.min()+t.max())/2+dx)
            self.assertGreaterEqual(front+e.distance-e.boundary,.0399)
        self.assertGreater(e.boundary-((obj[:,:2].mean(0)+e.distance*e.normal)@e.normal),.01)

    def test_ineligible_objects_not_pushed(self):
        table=box(np.array([0.,0.,.8]),np.array([.5,.5,.02]))
        for center,half in [([0.,.4,.79],[.07,.05,.01]),([0.,.4,.86],[.07,.05,.04]),([0.,0.,.83],[.02,.02,.001])]:
            with self.assertRaises(ValueError):edge_candidates(box(np.array(center),np.array(half)),table,[0.,.9])

    def test_authored_setup_includes_child_collision_without_resizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scene.xml'
            p.write_text('''<mujoco><asset/><worldbody><body name="table"><geom type="box" pos="0 0 .8" size=".5 .5 .02"/></body>
            <body name="obj" pos="0 0 .84"><freejoint name="obj_joint"/><body name="collision"><geom type="box" size=".075 .095 .008" mass=".1"/></body></body>
            </worldbody></mujoco>''')
            install_thin_scene(p,'obj','table',yaw=12.,inset=.06)
            m=mujoco.MjModel.from_xml_path(str(p));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
            v=collision_vertices(m,d,object_bodies(m,'obj'))
            self.assertAlmostEqual(v[:,1].max(),.44)
            self.assertAlmostEqual(v[:,2].min(),.821)
            self.assertEqual(m.joint('obj_joint').type,mujoco.mjtJoint.mjJNT_FREE)
            self.assertAlmostEqual(m.body_mass.sum(),.1+float(m.body_mass[m.body('table').id]))

if __name__=='__main__':unittest.main()
