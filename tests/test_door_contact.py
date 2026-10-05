"""Contact ownership and bidirectional hinge force regression tests."""
import unittest
import mujoco
import numpy as np
from cross_episode_sim.control.door_contact import (
    contact_path_collision, external_door_contacts, hand_panel_contact, solve_contact_ik,
)


def fixture(sign=1, hand='robot_0/EE_BODY_R', x=.0395):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".002" gravity="0 0 0"/>
    <worldbody>
      <body name="fridge_panel" pos="0 {sign*.5} 1">
        <joint name="hinge" axis="0 0 -1" damping=".1"/>
        <geom type="box" pos="0 {-sign*.25} 0" size=".02 .25 .3" mass="1"/>
      </body>
      <body name="{hand}" pos="{x} {sign*.25} 1">
        <joint name="hand" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".02" mass=".1"/>
      </body>
    </worldbody><actuator><position joint="hand" kp="200" kv="10"/></actuator></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data


class DoorContactTest(unittest.TestCase):
    def test_palm_push_generates_force_and_opens_either_hinge(self):
        for sign in (1,-1):
            with self.subTest(sign=sign):
                model,data=fixture(sign)
                initial=data.qpos[0]
                maximum=0.
                for i in range(250):
                    data.ctrl[0]=-.06*min(1.,i/200)
                    mujoco.mj_step(model,data)
                    maximum=max(maximum,hand_panel_contact(model,data,1)['normal_force_n'])
                self.assertGreater(maximum,.1)
                self.assertGreater(sign*(data.qpos[0]-initial),np.radians(1.))

    def test_only_hand_can_make_panel_contact(self):
        model,data=fixture(x=.0392)
        self.assertIsNone(contact_path_collision(model,data,handle_id=-1,panel_id=1))
        self.assertIsNotNone(contact_path_collision(model,data,handle_id=-1))
        model,data=fixture(hand='robot_0/link_right_arm_6',x=.0392)
        self.assertIsNotNone(contact_path_collision(model,data,handle_id=-1,panel_id=1))

    def test_intended_hand_contact_still_has_depth_limit(self):
        model,data=fixture(x=.038)
        bad=contact_path_collision(model,data,handle_id=-1,panel_id=1)
        self.assertIsNotNone(bad)
        self.assertGreater(bad['depth_m'],.001)

    def test_contact_ik_rejects_a_target_inside_the_wall(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="box" pos=".2 0 0" size=".04 .1 .1"/>
          <body name="robot_0/EE_BODY_R">
            <joint name="robot_0/right_arm_0" type="slide" axis="1 0 0" range="-.5 .5"/>
            <geom size=".03" mass="1"/><site name="robot_0/ee_site_r"/>
          </body></worldbody></mujoco>''')
        live=mujoco.MjData(model);probe=mujoco.MjData(model)
        goal=np.eye(4);goal[0,3]=.1
        q=solve_contact_ik(model,probe,goal,['right_arm_0'],[0.],handle_id=-1)
        self.assertAlmostEqual(q[0],.1,places=4)
        goal[0,3]=.2
        with self.assertRaisesRegex(RuntimeError,'No collision-clear contact IK'):
            solve_contact_ik(model,probe,goal,['right_arm_0'],q,handle_id=-1)
        self.assertEqual(live.qpos[0],0.)

    def test_door_wall_contact_is_not_an_allowed_hand_contact(self):
        model,data=fixture(hand='wall',x=.0392)
        contacts=external_door_contacts(model,data,{1},'fridge')
        self.assertEqual(contacts[0]['body'],'wall')
        self.assertGreater(contacts[0]['depth_m'],.0005)

if __name__=='__main__':unittest.main()
