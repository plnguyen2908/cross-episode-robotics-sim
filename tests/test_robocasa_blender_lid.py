"""Native lid geometry preservation and parity with RoboCasa's closed predicate."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from cross_episode_sim.paths import robocasa_dir
from cross_episode_sim.fixtures.blender_lid import (
    BLENDER, LID, GENERATED, BlenderLidTest, cylinder_cover, install_blender, lid_closed_state)

ROOT=Path(__file__).resolve().parents[3]

class BlenderLidTests(unittest.TestCase):
    def test_native_cylinder_covers_enclose_surface(self):
        counts=[]
        for radius,half in ((.0767563557,.0013114765),(.0241449428,.0282228402)):
            cover=cylinder_cover(radius,half);counts.append(len(cover))
            points=np.array([[r*np.cos(a),r*np.sin(a),z]
                for r in np.linspace(0,radius,33)
                for a in np.linspace(0,2*np.pi,128,endpoint=False)
                for z in np.linspace(-half,half,9)])
            gap=np.linalg.norm(points[:,None,:]-cover[None,:,:3],axis=2)-cover[None,:,3]
            self.assertLessEqual(float(gap.min(axis=1).max()),1e-10)
        self.assertLessEqual(sum(counts),40)

    def test_cylinder_side_contact_region_has_real_surface_samples(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="lid">
        <geom type="cylinder" size=".024 .028"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        fixture=SimpleNamespace(model=model,data=data,bread_bids={model.body('lid').id})
        points=BlenderLidTest.bread_vertices(fixture)
        # A narrow band at the handle's middle has no bounding-box corners,
        # but must expose both sides of its 48 mm diameter to opposing pads.
        middle=points[np.abs(points[:,2])<.004]
        self.assertGreater(len(middle),0)
        self.assertAlmostEqual(float(np.ptp(middle[:,0])),.048)
        np.testing.assert_allclose(np.linalg.norm(points[:,:2],axis=1),.024)

    def test_closed_state_matches_native_update(self):
        source=robocasa_dir()/'models/fixtures/blender.py'
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Blender')
        update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_state')
        utilities=robocasa_dir()/'utils/object_utils.py'
        upright=next(n for n in ast.parse(utilities.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_fxtr_upright')
        ns=dict(np=np,R=Rotation)
        exec(compile(ast.Module(body=[upright],type_ignores=[]),str(utilities),'exec'),ns)
        ns['OU']=SimpleNamespace(check_fxtr_upright=ns['check_fxtr_upright'])
        exec(compile(ast.Module(body=[update],type_ignores=[]),str(source),'exec'),ns)
        for distance in (0.,.02,.039,.04,.041):
            for roll,pitch in ((0.,0.),(6.9,0.),(7.1,0.),(0.,6.9),(0.,7.1),(180.,0.)):
                with self.subTest(distance=distance,roll=roll,pitch=pitch):
                    rotation=Rotation.from_euler('xyz',[roll,pitch,33.],degrees=True)
                    position=np.array([distance,0.,0.])
                    fixture=SimpleNamespace(name='blender',blender_lid=SimpleNamespace(name='lid'),
                        _BLENDER_LID_POS_THRESH=.04,_turned_on=False,_button_contact_prev_timestep=False,
                        get_curr_lid_pos=lambda env:position,get_lid_closed_pos=lambda env:np.zeros(3))
                    env=SimpleNamespace(robots=[SimpleNamespace(gripper={'right':None})],
                        check_contact=lambda *args:False,
                        sim=SimpleNamespace(data=SimpleNamespace(get_body_xquat=lambda name:rotation.as_quat(scalar_first=True))))
                    ns['update_state'](fixture,env)
                    self.assertEqual(lid_closed_state(position,rotation.as_matrix(),np.zeros(3)),fixture._lid_on_blender)

    def test_install_preserves_free_lid_and_native_colliders(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene=Path(tmp)/'scene.xml'
            scene.write_text('''<mujoco><compiler angle="radian"/><asset/><worldbody>
            <body name="table"><geom type="box" pos="0 0 .8" size="1 1 .02"/></body>
            <body name="old_object" pos="0 0 .9"><freejoint/><geom type="sphere" size=".02"/></body>
            </worldbody></mujoco>''')
            install_blender(scene,'old_object','table',np.array([0.,0.,.9]))
            native=ET.parse(GENERATED);actual=ET.parse(scene)
            for name in (BLENDER,LID):
                before=native.find(f".//body[@name='{name}']")
                after=actual.find(f".//body[@name='{name}']")
                self.assertEqual([g.attrib for g in before.iter('geom')],[g.attrib for g in after.iter('geom')])
            lid=actual.find(f".//body[@name='{LID}']")
            self.assertEqual(lid.find('joint').attrib,native.find(f".//body[@name='{LID}']/joint").attrib)
            self.assertEqual(lid.find('joint').get('type'),'free')
            self.assertIsNone(actual.find('actuator'))
            self.assertIsNone(actual.find('equality'))
            self.assertEqual(len(actual.findall('.//joint')),1)

if __name__=='__main__':unittest.main()
