"""Native latch parity and preservation of the articulated bread tray."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from cross_episode_sim.fixtures.toaster_lever import ToasterLatch,LeverCatch,add_lever_catch,install_lever_scene,JOINT,LEVER
from cross_episode_sim.fixtures.toaster_insertion import install_toaster,GENERATED
from cross_episode_sim.paths import robocasa_dir

ROOT=Path(__file__).resolve().parents[3]

class ToasterLeverTests(unittest.TestCase):
    def test_native_activation_latch_timeout_cooldown_and_repress(self):
        path=robocasa_dir()/'models/fixtures/toaster.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Toaster')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_state')
        ns={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
        fixture=SimpleNamespace(_slot_pairs=[0],_controls={'lever':'lever'},_joint_names={'lever_0':'lever'},
            _state={0:{'lever':0.}},_turned_on={0:False},_num_steps_on={0:0},_cooldown={0:0})
        env=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(joint_names=['lever'])))
        ours=ToasterLatch()
        for value in [0.,.5,.899,.901]+[1.]*510+[.8]*1000+[.3,.91,1.]:
            calls=[]
            fixture.get_joint_state=lambda env,names:{'lever':value}
            fixture.set_lever=lambda *args:calls.append(args[-1])
            ns['update_state'](fixture,env)
            latched=ours.step(value)
            self.assertEqual((ours.turned_on,ours.steps_on,ours.cooldown),
                             (fixture._turned_on[0],fixture._num_steps_on[0],fixture._cooldown[0]))
            self.assertEqual(latched,bool(calls))
        self.assertFalse(ToasterLatch().step(.899))
        latch=ToasterLatch();self.assertTrue(latch.step(.90));self.assertTrue(latch.turned_on)

    def test_solver_catch_holds_without_resets_and_releases_on_timeout(self):
        # Native spring/friction/damping, isolated from the robot to exercise
        # latch dynamics over the entire 25-second timer and release.
        root=ET.fromstring(f"""<mujoco><option timestep=".002"/>
        <worldbody><body pos="0 0 1"><joint name="{JOINT}" type="slide"
        axis="0 0 -1" range="0 .05447" damping="1" frictionloss="1"
        armature=".1" stiffness="3.6" springref="-.5"/>
        <geom type="box" size=".01 .01 .01" mass=".04"/></body></worldbody></mujoco>""")
        add_lever_catch(root)
        m=mujoco.MjModel.from_xml_string(ET.tostring(root).decode());d=mujoco.MjData(m)
        d.qpos[0]=.05;d.qvel[0]=.01;mujoco.mj_forward(m,d)
        catch=LeverCatch(m,d);state=ToasterLatch()
        q=d.qpos.copy();v=d.qvel.copy();catch.update(state.step(.05/.05447))
        np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
        values=[]
        for _ in range(499):
            for _ in range(25):mujoco.mj_step(m,d)
            values.append(float(d.qpos[0]));catch.update(state.step(float(d.qpos[0]/.05447)))
        self.assertTrue(state.turned_on)
        self.assertLess(np.ptp(values[10:]),.0001)
        self.assertLess(abs(d.qpos[0]-.05),.0005)
        self.assertEqual(m.nu,0)
        q=d.qpos.copy();v=d.qvel.copy();catch.update(state.step(float(d.qpos[0]/.05447)))
        self.assertFalse(state.turned_on);self.assertFalse(d.eq_active[catch.eq])
        np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
        for _ in range(1000):mujoco.mj_step(m,d)
        self.assertLess(d.qpos[0],.005)

    def test_native_slide_and_tray_preserved_bread_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scene.xml'
            p.write_text('''<mujoco><compiler angle="radian"/><asset/><worldbody>
            <body name="table"><geom type="box" pos="0 0 .8" size="1 1 .02"/></body>
            <body name="bread" pos="0 0 .83"><freejoint name="bread_joint"/><geom type="box" size=".055 .055 .009" mass=".02"/></body>
            </worldbody></mujoco>''')
            install_toaster(p,'bread','table',np.array([0.,0.,.83]));install_lever_scene(p,'bread')
            tree=ET.parse(p);native=ET.parse(GENERATED)
            self.assertEqual(tree.find(f".//joint[@name='{JOINT}']").attrib,native.find(f".//joint[@name='{JOINT}']").attrib)
            self.assertIsNotNone(tree.find(f".//body[@name='{LEVER}']/geom[@name='skill_toaster_slotR_floor']"))
            model=mujoco.MjModel.from_xml_path(str(p));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            floor=data.geom('skill_toaster_slotR_floor').xpos.copy();data.joint(JOINT).qpos[0]=.04;mujoco.mj_forward(model,data)
            np.testing.assert_allclose(data.geom('skill_toaster_slotR_floor').xpos-floor,[0.,0.,-.04],atol=1e-8)
            self.assertEqual(model.joint('bread_joint').type,mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(model.nu,0)

if __name__=='__main__':unittest.main()
