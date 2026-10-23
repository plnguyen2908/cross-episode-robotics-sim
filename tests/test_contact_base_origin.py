"""A lagging servo reference must not change the collision-checked endpoint."""
from types import SimpleNamespace
import mujoco
import numpy as np
from cross_episode_sim.fixtures.base_articulation import move_contact


def test_contact_translation_uses_measured_base_origin():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body><joint name="x" type="slide" axis="1 0 0" damping="20"/>
      <joint name="y" type="slide" axis="0 1 0" damping="20"/><geom size=".1" mass="1"/></body>
      <body pos="2 0 0"><joint name="hinge"/><geom size=".1" mass="1"/></body></worldbody>
      <actuator><position name="ax" joint="x" kp="500"/><position name="ay" joint="y" kp="500"/></actuator></mujoco>''')
    d=mujoco.MjData(m);d.ctrl[:]=[.2,-.1];mujoco.mj_forward(m,d)
    c=SimpleNamespace(model=m,data=d,profile=SimpleNamespace(namespace='',base_joints=['x','y'],base_actuators=['ax','ay']),
        planner=SimpleNamespace(names=[]),door_joint=2,door_address=2,angle=lambda:0.,door_collision=lambda data:None,record=lambda **kw:None)
    def tcp():
        p=np.eye(4);p[:2,3]=d.qpos[:2];return p
    def tick(seconds):
        for _ in range(int(np.ceil(seconds/.002))):mujoco.mj_step(m,d)
    c.tcp=tcp;c.tick=tick
    target=np.eye(4);target[0,3]=.01
    move_contact(c,'push',target,0.,follow_arm=False)
    np.testing.assert_allclose(d.ctrl,[.01,0.],atol=1e-12)
