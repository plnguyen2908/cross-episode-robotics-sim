from types import SimpleNamespace
import mujoco
import numpy as np
from cross_episode_sim.robot.cameras import install_onboard_cameras


def test_mast_tracks_base_not_arm_and_retains_wrist():
    s=mujoco.MjSpec.from_string('''<mujoco><worldbody><body name="robot_0/base">
    <joint name="robot_0/yaw" axis="0 0 1"/><geom size=".1" mass="1"/>
    <body name="hand" pos="0 0 .7"><joint name="arm"/><geom size=".03" mass=".1"/>
    <site name="robot_0/tcp"/><camera name="robot_0/wrist_cam" fovy="60"/></body>
    </body></worldbody></mujoco>''')
    profile=SimpleNamespace(base_joints=('yaw',),tcp_site='tcp')
    install_onboard_cameras(s,profile);m=s.compile();d=mujoco.MjData(m)
    mujoco.mj_forward(m,d);nav=m.camera('robot_0/navigation_cam').id
    assert d.cam_xpos[nav, 2] >= 1.70
    np.testing.assert_allclose(d.xpos[m.body('robot_0/navigation_mast').id,:2], [-.40,0.])
    initial=d.cam_xmat[nav].copy();d.joint('arm').qpos[0]=.5;mujoco.mj_forward(m,d)
    np.testing.assert_allclose(initial,d.cam_xmat[nav])
    d.joint('robot_0/yaw').qpos[0]=.5;mujoco.mj_forward(m,d)
    assert not np.allclose(initial,d.cam_xmat[nav])
    assert m.cam_fovy[m.camera('robot_0/wrist_cam').id]==60
    assert m.geom('robot_0/navigation_mast_pole').contype[0]!=0
    plate=m.geom('robot_0/navigation_mast_plate')
    assert plate.contype[0]!=0
    np.testing.assert_allclose(plate.size, [.1225,.04,.003])


def test_only_two_views_rendered_and_recorded():
    from cross_episode_sim.tasks.history_views import render_history_views
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <camera name="robot_0/navigation_cam"/><camera name="robot_0/wrist_cam"/>
    <camera name="history_kitchen_wall_camera"/></worldbody></mujoco>''')
    seen=[];recorded=[]
    r=SimpleNamespace(update_scene=lambda *a,**kw:seen.append(kw['camera']))
    r.depth=False
    r.enable_depth_rendering=lambda:setattr(r,'depth',True)
    r.disable_depth_rendering=lambda:setattr(r,'depth',False)
    r.render=lambda:np.ones((48,64),dtype=np.float32) if r.depth else np.zeros((48,64,3),dtype=np.uint8)
    c=SimpleNamespace(model=m,data=mujoco.MjData(m),profile=SimpleNamespace(namespace='robot_0/'),renderer=r,record_camera_frames=recorded.append)
    assert render_history_views(c,record_raw=True).shape==(96,128,3)
    assert seen==['robot_0/navigation_cam','robot_0/navigation_cam','robot_0/wrist_cam','robot_0/wrist_cam']
    assert len(recorded[0])==2
