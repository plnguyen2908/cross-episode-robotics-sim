"""The coupled prediction must check moving-base collisions without editing live physics."""
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest

from cross_episode_sim.fixtures import door_push
from cross_episode_sim.fixtures.base_articulation import preflight_panel


@pytest.mark.parametrize('blocked', [False, True])
def test_coupled_sweep_restores_live_state_and_checks_base(monkeypatch, blocked):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="panel"><joint name="hinge"/><geom size=".02"/></body>
      <body name="robot"><joint name="robot_0/base_x" type="slide" axis="1 0 0"/>
        <joint name="robot_0/base_y" type="slide" axis="0 1 0"/>
        <joint name="robot_0/arm"/><geom size=".02"/></body>
    </worldbody></mujoco>''')
    live = mujoco.MjData(model)
    live.qpos[0] = .5
    mujoco.mj_forward(model, live)
    before = live.qpos.copy()
    visited = []
    def collision(probe):
        visited.append(probe.qpos.copy())
        if blocked and np.linalg.norm(probe.qpos[1:3]) > .001:
            return {'moving_base_obstruction': True}
    c = SimpleNamespace(model=model, data=live, door_joint=0, door_address=0,
        profile=SimpleNamespace(namespace='robot_0/', base_joints=['base_x', 'base_y']),
        angle=lambda: float(c.data.qpos[0]), nearby_ik=lambda *a, **kw: [0.],
        door_collision=collision)
    pose = np.eye(4); pose[:3, 3] = [.5, 0., 0.]
    def contact_pose(c, surface, direction, tilt, rotation):
        p = np.eye(4); p[:3, 3] = surface; p[:3, :3] = rotation
        return p
    monkeypatch.setattr(door_push, 'pushing_pose', contact_pose)
    if blocked:
        with pytest.raises(RuntimeError, match='coupled push sweep blocked'):
            preflight_panel(c, pose, [0.], .7, [3], pose[:3, 3], [0., 1., 0.])
    else:
        preflight_panel(c, pose, [0.], .7, [3], pose[:3, 3], [0., 1., 0.])
    assert visited
    assert any(np.linalg.norm(q[1:3]) > .001 for q in visited)
    assert c.data is live
    np.testing.assert_array_equal(live.qpos, before)


@pytest.mark.parametrize('blocked', [False, True])
def test_drawer_sweep_checks_local_base_and_preserves_live_state(blocked):
    from cross_episode_sim.fixtures.base_articulation import preflight_slide
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="drawer" type="slide" axis="0 1 0"/><geom size=".02"/></body>
      <body><joint name="robot_0/base_x" type="slide" axis="1 0 0"/>
        <joint name="robot_0/base_y" type="slide" axis="0 1 0"/>
        <joint name="robot_0/arm"/><geom size=".02"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    before=data.qpos.copy();visited=[]
    def collision(probe):
        visited.append(probe.qpos.copy())
        return {'obstruction':True} if blocked and probe.qpos[2]<-.05 else None
    c=SimpleNamespace(model=model,data=data,door_joint=0,door_address=0,angle=lambda:0.,
        planner=SimpleNamespace(names=['arm']),
        profile=SimpleNamespace(namespace='robot_0/',base_joints=['base_x','base_y']),
        door_collision=collision)
    if blocked:
        with pytest.raises(RuntimeError,match='drawer/base sweep blocked'):
            preflight_slide(c,np.eye(4),[0.],-.2)
    else:
        preflight_slide(c,np.eye(4),[0.],-.2)
        assert visited[-1][0]==pytest.approx(-.2)
        assert visited[-1][2]==pytest.approx(-.2)
    np.testing.assert_array_equal(data.qpos,before)



def test_drawer_sweep_projects_supported_contents_only_in_scratch_state():
    from cross_episode_sim.fixtures.base_articulation import project_slide_contents
    model=mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="drawer"><joint name="slide" type="slide" axis="0 1 0"/>
        <geom type="box" size=".3 .3 .01"/></body>
      <body name="cup" pos="0 0 .06"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
      <body name="other" pos="2 0 .06"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
      </worldbody></mujoco>""")
    live=mujoco.MjData(model);mujoco.mj_forward(model,live)
    before=live.qpos.copy()
    probe=mujoco.MjData(model);probe.qpos[:]=live.qpos;probe.qpos[0]=-.2
    mujoco.mj_forward(model,probe)
    c=SimpleNamespace(model=model,door_joint=0,door_address=0,
        door_bids={model.body('drawer').id},profile=SimpleNamespace(namespace='robot_0/'))
    project_slide_contents(c,probe,live)
    np.testing.assert_allclose(probe.body('cup').xpos,[0,-.2,.06])
    np.testing.assert_allclose(probe.body('other').xpos,[2,0,.06])
    np.testing.assert_array_equal(live.qpos,before)
    # Never project into live execution, even if the helper is called there.
    project_slide_contents(c,live,live)
    np.testing.assert_array_equal(live.qpos,before)


@pytest.mark.parametrize('blocked', [False, True])
def test_handle_sweep_checks_base_and_restores_live_state(monkeypatch, blocked):
    from cross_episode_sim.fixtures.base_articulation import preflight_handle_hinge
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="panel"><joint name="hinge"/><geom size=".02"/></body>
      <body name="robot"><joint name="robot_0/base_x" type="slide" axis="1 0 0"/>
        <joint name="robot_0/base_y" type="slide" axis="0 1 0"/>
        <joint name="robot_0/arm"/><geom size=".02"/></body>
    </worldbody></mujoco>''')
    live = mujoco.MjData(model)
    live.qpos[0] = .5
    mujoco.mj_forward(model, live)
    before = live.qpos.copy()
    visited = []
    def collision(probe):
        visited.append(probe.qpos.copy())
        if blocked and np.linalg.norm(probe.qpos[1:3]) > .001:
            return {'moving_base_obstruction': True}
    c = SimpleNamespace(model=model, data=live, door_joint=0, door_address=0,
        planner=SimpleNamespace(names=['arm']),
        profile=SimpleNamespace(namespace='robot_0/', base_joints=['base_x', 'base_y']),
        angle=lambda: float(c.data.qpos[0]), nearby_ik=lambda *a, **kw: [0.],
        door_collision=collision)
    pose = np.eye(4); pose[:3, 3] = [.5, 0., 0.]
    def contact_pose(c, surface, direction, tilt, rotation):
        p = np.eye(4); p[:3, 3] = surface; p[:3, :3] = rotation
        return p
    monkeypatch.setattr(door_push, 'pushing_pose', contact_pose)
    if blocked:
        with pytest.raises(RuntimeError, match='Coupled handle sweep blocked'):
            preflight_handle_hinge(c, pose, [0.], .7)
    else:
        preflight_handle_hinge(c, pose, [0.], .7)
    assert visited
    assert any(np.linalg.norm(q[1:3]) > .001 for q in visited)
    assert c.data is live
    np.testing.assert_array_equal(live.qpos, before)
