"""Blender release-latch parity and composite scene preservation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.blender_load import (
    next_power_state,install_loading_scene,align_disk_center,BUTTON_BODY)
from cross_episode_sim.fixtures.blender_lid import GENERATED,LID,BLENDER
from cross_episode_sim.paths import robocasa_dir
ROOT=Path(__file__).resolve().parents[3]

class BlenderLoadTests(unittest.TestCase):
    def test_lid_disk_stays_centered_when_grasp_changes_yaw(self):
        center=np.array([-.0101937,.0084864,-.0101117])
        closed=np.eye(4);closed[:3,3]=[2.6,-5.96,1.14]
        closed[:3,:3]=Rotation.from_euler('z',180,degrees=True).as_matrix()
        for yaw in (-145,-90,0,90,180):
            target=closed.copy();target[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            adjusted=align_disk_center(target,closed,center)
            np.testing.assert_allclose(adjusted[:3,:3]@center+adjusted[:3,3],closed[:3,:3]@center+closed[:3,3])
            np.testing.assert_allclose(adjusted[:3,:3],target[:3,:3])

    def test_native_power_release_and_lid_interlock(self):
        path=robocasa_dir()/'models/fixtures/blender.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Blender')
        update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_state')
        for closed in (False,True):
            ns=dict(np=np,OU=SimpleNamespace(check_fxtr_upright=lambda *args,**kwargs:closed))
            exec(compile(ast.Module(body=[update],type_ignores=[]),str(path),'exec'),ns)
            for initial in (False,True):
                fixture=SimpleNamespace(name='fixture',blender_lid=SimpleNamespace(name='lid'),
                    _BLENDER_LID_POS_THRESH=.04,_turned_on=initial,_button_contact_prev_timestep=False,
                    get_curr_lid_pos=lambda env:np.zeros(3),get_lid_closed_pos=lambda env:np.zeros(3))
                ours=initial;previous=False
                for contact in (False,True,True,False,False,True,False):
                    env=SimpleNamespace(robots=[SimpleNamespace(gripper={'right':None})],check_contact=lambda *args:contact)
                    ns['update_state'](fixture,env)
                    ours=next_power_state(ours,previous,contact,closed);previous=contact
                    self.assertEqual(ours,fixture._turned_on)
        self.assertFalse(next_power_state(False,False,True,True))
        self.assertTrue(next_power_state(False,True,False,True))
        self.assertFalse(next_power_state(True,False,False,False))

    def test_loading_scene_retains_object_and_native_button_joint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.xml'
            path.write_text('''<mujoco><compiler angle="radian"/><asset/><worldbody>
            <body name="table"><geom type="box" pos="0 0 .8" size="1 1 .02"/></body>
            <body name="test_object_main" pos="0 0 .85"><freejoint name="food_joint"/><geom type="sphere" size=".02" mass=".04"/></body>
            </worldbody></mujoco>''')
            lid,pos,food=install_loading_scene(path,'test_object_main','table',np.array([0.,0.,.85]))
            actual=ET.parse(path);native=ET.parse(GENERATED)
            self.assertEqual(actual.find(".//body[@name='test_object_main']/geom").get('size'),'.02')
            np.testing.assert_allclose(food['position'],[.32,0.,.85])
            self.assertEqual(actual.find(f".//body[@name='{BUTTON_BODY}']/joint").attrib,
                             native.find(f".//body[@name='{BUTTON_BODY}']/joint").attrib)
            self.assertIsNotNone(actual.find(f".//body[@name='{LID}']/joint"))
            self.assertEqual(actual.find(f".//body[@name='{BLENDER}']").get('quat'),'0 0 0 1')

if __name__=='__main__':unittest.main()
