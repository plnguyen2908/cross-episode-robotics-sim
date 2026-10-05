"""Native toaster scene preservation and fitted-object geometry checks."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from cross_episode_sim.paths import robocasa_dir

from cross_episode_sim.fixtures.toaster_insertion import (
    GENERATED, TOASTER, collision_vertices, install_toaster, edge_annotations)


class ToasterInsertionTests(unittest.TestCase):
    def test_fixture_geometry_and_free_slice_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scene.xml'
            path.write_text('''<mujoco><compiler angle="radian"/><asset/><worldbody>
            <body name="table"><geom type="box" pos="0 0 .8" size="1 1 .02"/><geom type="box" pos="0 0 10" size=".01 .01 .01"/></body>
            <body name="bread" pos="0 0 .831"><freejoint name="bread_joint"/>
            <geom type="box" size=".055 .055 .009" mass=".02"/></body>
            </worldbody></mujoco>''')
            install_toaster(path,'bread','table',np.array([0.,0.,.831]))
            actual=ET.parse(path);native=ET.parse(GENERATED)
            for original in native.iter('geom'):
                self.assertEqual(actual.find(f".//geom[@name='{original.get('name')}']").attrib,original.attrib)
            self.assertIsNotNone(actual.find(".//body[@name='bread']/freejoint"))
            self.assertEqual(actual.find(".//body[@name='bread']/geom").get('size'),'.055 .055 .009')
            m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
            vertices=collision_vertices(m,d,{m.body('bread').id})
            np.testing.assert_allclose(np.ptp(vertices,axis=0),[.018,.11,.11],atol=1e-7)
            self.assertAlmostEqual(vertices[:,2].min(),.821,places=6)
            hypotheses=np.load(edge_annotations(path,'bread',folder))['transforms']
            self.assertGreaterEqual(len(hypotheses),5)
            self.assertTrue(np.isfinite(hypotheses).all())

    def test_native_slot_fits_unscaled_selected_slice(self):
        root=Path(__file__).resolve().parents[3]
        p=robocasa_dir()/'models/assets/objects/lightwheel/sandwich_bread/SandwichBread005/model.xml'
        m=mujoco.MjModel.from_xml_path(str(p));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        extent=np.ptp(collision_vertices(m,d,set(range(m.nbody))),axis=0)
        floor=ET.parse(GENERATED).find(".//geom[@name='skill_toaster_slotR_floor']")
        slot=2*np.fromstring(floor.get('size'),sep=' ')
        # Upright slice: original thickness becomes slot width, original y length stays y.
        self.assertLess(extent[2]+.004,slot[0])
        self.assertLess(extent[1]+.004,slot[1])


if __name__=='__main__':unittest.main()
