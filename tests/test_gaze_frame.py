"""The optical axis must follow the target when the torso rotates or leans."""
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.controller.navigation import NavigationTransfer


class GazeFrameTest(unittest.TestCase):
    def test_look_at_compensates_torso_rotation(self):
        for angles in ([0,0,0],[30,20,15],[-40,-25,20]):
            with self.subTest(torso=angles):
                rotation=Rotation.from_euler('zyx',angles,degrees=True)
                quat=' '.join(map(str,rotation.as_quat(scalar_first=True)))
                model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
                  <body name="torso" quat="{quat}">
                    <body name="pan"><joint name="robot_0/head_0" axis="0 0 1" range="-1.5 1.5"/>
                      <geom size=".01" mass=".1"/>
                      <body name="tilt"><joint name="robot_0/head_1" axis="0 1 0" range="-1.2 1.2"/>
                        <geom size=".01" mass=".1"/>
                        <camera name="head" quat=".7071067812 0 -.7071067812 0"/>
                      </body>
                    </body>
                  </body></worldbody></mujoco>''')
                c=NavigationTransfer.__new__(NavigationTransfer)
                c.model=model;c.data=mujoco.MjData(model);c.head_camera_id=model.camera('head').id
                mujoco.mj_forward(model,c.data)
                pan,tilt=.3,-.2
                local=np.array([np.cos(pan)*np.cos(tilt),np.sin(pan)*np.cos(tilt),-np.sin(tilt)])
                target=rotation.apply(local)*3
                actual=c.gaze_angles(c.data,target)
                np.testing.assert_allclose(actual,[pan,tilt],atol=1e-8)
                c.data.qpos[:]=actual;mujoco.mj_forward(model,c.data)
                optical_axis=-c.data.cam_xmat[c.head_camera_id].reshape(3,3)[:,2]
                np.testing.assert_allclose(optical_axis,target/np.linalg.norm(target),atol=1e-8)


if __name__=='__main__':unittest.main()
