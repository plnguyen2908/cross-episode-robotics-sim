"""Flat initial bread and freely moving native plate remain physical bodies."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from cross_episode_sim.fixtures.toaster_insertion import install_toaster, collision_vertices
from cross_episode_sim.fixtures.toaster_plate import install_plate, PLATE, PLATE_XML

class PlateSetupTests(unittest.TestCase):
    def test_flat_slice_free_plate_native_mass_and_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scene.xml'
            path.write_text('''<mujoco><compiler angle="radian"/><asset/><worldbody>
            <body name="table"><geom type="box" pos="0 0 .8" size="1 1 .02"/></body>
            <body name="bread" pos="0 0 .831"><freejoint name="bread_joint"/>
            <geom type="box" size=".055 .055 .009" mass=".02"/></body>
            </worldbody></mujoco>''')
            install_toaster(path,'bread','table',np.array([0.,0.,.831]));install_plate(path,'bread')
            m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
            self.assertEqual(m.jnt_type[m.joint(PLATE+'_joint').id],mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(m.jnt_type[m.joint('bread_joint').id],mujoco.mjtJoint.mjJNT_FREE)
            v=collision_vertices(m,d,{m.body('bread').id})
            np.testing.assert_allclose(np.ptp(v,axis=0),[.11,.11,.018],atol=1e-7)
            pm=mujoco.MjModel.from_xml_path(str(PLATE_XML))
            bids={m.body(PLATE).id}
            for bid in range(m.nbody):
                if m.body_parentid[bid] in bids:bids.add(bid)
            self.assertAlmostEqual(float(m.body_mass[list(bids)].sum()),float(pm.body_mass.sum()),places=8)
            plate=collision_vertices(m,d,bids)
            self.assertAlmostEqual(float(plate[:,1].max()),.992,places=6)
            self.assertTrue(np.all(np.abs(plate[:,:2])<1.))
            self.assertLess(v[:,1].max(),plate[:,1].max())
            tree=ET.parse(path)
            self.assertIsNone(tree.find('equality/weld'))
            self.assertGreater(v[:,2].min(),.82)

if __name__=='__main__':unittest.main()
