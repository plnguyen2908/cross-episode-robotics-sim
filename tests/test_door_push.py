"""Push geometry uses the opening torque, never a panel pinch or joint edit."""
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from cross_episode_sim.fixtures.door_push import panel_push_geometry, panel_contact


@pytest.mark.parametrize('start_angle', [.3, np.radians(11.8)])
@pytest.mark.parametrize('closing', [False, True])
@pytest.mark.parametrize('axis', [-1, 1])
@pytest.mark.parametrize('direction', [-1, 1])
def test_push_face_produces_requested_torque_and_preserves_live_state(axis, direction, closing, start_angle):
    m = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="door"><joint name="hinge" axis="0 0 {axis}"/>
        <body name="panel" pos=".3 0 0"><geom name="panel_box" type="box" size=".3 .015 .3"/></body>
        <body name="handle" pos=".55 -.04 0"><geom name="handle_box" type="box" size=".01 .01 .1"/></body>
      </body></worldbody></mujoco>''')
    d = mujoco.MjData(m);d.qpos[0] = direction*start_angle;mujoco.mj_forward(m, d)
    c = SimpleNamespace(model=m, data=d, door_joint=0, angle=lambda: float(d.qpos[0]),
        handle_bids={m.body('handle').id}, handle_geom=m.geom('handle_box').id,
        door_bids={m.body(n).id for n in ('door','panel','handle')},
        rebuild_planner=lambda: None)
    before=d.qpos.copy(); masks=m.geom_contype.copy()
    gid, surface, force = panel_push_geometry(c, 0. if closing else direction*.9)
    assert gid == m.geom('panel_box').id
    torque=np.dot(d.xaxis[0], np.cross(surface-d.xanchor[0], force))
    assert (-1 if closing else 1)*direction*torque > .05
    normal=d.geom_xmat[gid].reshape(3,3)[:,1]
    assert abs(np.dot(surface-d.geom_xpos[gid],normal)) == pytest.approx(.015)
    assert np.dot(surface-d.geom_xpos[gid],force) < 0
    with pytest.raises(RuntimeError,match='blocked'):
        with panel_contact(c,gid):
            assert c._door_push
            raise RuntimeError('blocked')
    assert not c._door_push
    assert c.handle_geom == m.geom('handle_box').id
    np.testing.assert_array_equal(d.qpos,before)
    np.testing.assert_array_equal(m.geom_contype,masks)
    d.qpos[0]=0
    with pytest.raises(RuntimeError,match='handle'):
        panel_push_geometry(c,direction*.9)


def test_push_preflight_rejects_leaf_sweep_into_base_without_live_motion():
    from cross_episode_sim.fixtures.door_push import preflight_push
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="hinge" axis="0 0 1"/><geom name="panel" type="box" pos=".3 0 0" size=".3 .01 .2"/></body>
      <body pos="0 0 2"><joint name="robot_0/arm"/><geom size=".01"/></body>
      <geom name="base" type="box" pos=".4 .4 0" size=".08 .08 .2"/>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    base=m.geom('base').id
    c=SimpleNamespace(model=m,data=d,door_joint=0,door_address=0,angle=lambda:0.,
        nearby_ik=lambda *args,**kw:np.zeros(1),
        local_contact_path=lambda *args:np.zeros((2,1)),
        door_collision=lambda probe: next(({'base_collision':True} for contact in probe.contact
             if base in (contact.geom1,contact.geom2) and contact.dist<-.001),None))
    before=d.qpos.copy()
    with pytest.raises(RuntimeError,match='Full panel push is blocked'):
        preflight_push(c,np.eye(4),np.zeros(1),np.pi/2,[1])
    np.testing.assert_array_equal(d.qpos,before)


def test_push_can_keep_wrist_orientation_as_contact_face_rotates():
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.fixtures.door_push import pushing_pose
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot_0/gripper/left_pad" pos="0 -.01 0"><geom type="box" size=".01 .005 .02"/></body>
      <body name="robot_0/gripper/right_pad" pos="0 .01 0"><geom type="box" size=".01 .005 .02"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=SimpleNamespace(model=m,data=d,tcp=lambda:np.eye(4))
    rotation=Rotation.from_euler('x',-.6).as_matrix()
    surface=np.array([1.,2.,3.])
    for angle in (0.,.4,.8):
        direction=Rotation.from_euler('z',angle).apply([0.,1.,0.])
        pose=pushing_pose(c,surface,direction,0.,rotation=rotation)
        np.testing.assert_allclose(pose[:3,:3],rotation)
        points=[]
        for gid in range(m.ngeom):
            for x in (-1,1):
                for y in (-1,1):
                    for z in (-1,1):
                        offset=d.geom_xpos[gid]+np.array([x,y,z])*m.geom_size[gid]
                        points.append(pose[:3,3]+rotation@offset)
        assert np.max((np.asarray(points)-surface)@direction)==pytest.approx(.0005)


def test_base_sweep_rejects_before_any_arm_ik_or_live_motion():
    from cross_episode_sim.fixtures.door_push import preflight_base_sweep, PushApproachError
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="door"><joint name="hinge" axis="0 0 1"/><geom type="box" pos=".3 0 0" size=".3 .01 .2"/></body>
      <body name="robot_0/base" pos=".4 .4 0"><geom type="box" size=".08 .08 .2"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=SimpleNamespace(model=m,data=d,door_address=0,door_bids={m.body('door').id},
                      profile=SimpleNamespace(namespace='robot_0/'),angle=lambda:0.)
    before=d.qpos.copy()
    with pytest.raises(PushApproachError,match='base outside'):
        preflight_base_sweep(c,np.pi/2)
    np.testing.assert_array_equal(d.qpos,before)
    preflight_base_sweep(c,.1)


def test_push_selects_panel_when_native_handle_shares_hinged_body():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="door"><joint name="hinge" axis="0 0 1"/>
        <geom name="panel" type="box" pos=".3 0 .8" size=".3 .015 .7"/>
        <geom name="handle" type="box" pos=".55 -.05 .8" size=".01 .01 .1"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);d.qpos[0]=.4;mujoco.mj_forward(m,d)
    c=SimpleNamespace(model=m,data=d,door_joint=0,angle=lambda:float(d.qpos[0]),
        handle_bids={m.body('door').id},door_bids={m.body('door').id},
        handle_geom=m.geom('handle').id)
    before=d.qpos.copy()
    gid,_,_=panel_push_geometry(c,1.5)
    assert gid==m.geom('panel').id
    np.testing.assert_array_equal(d.qpos,before)


def test_contact_refresh_tracks_passive_door_drift_without_mutating_state():
    from cross_episode_sim.fixtures.door_push import refresh_panel_contact
    from scipy.spatial.transform import Rotation
    c = SimpleNamespace(angle=lambda: .7, door_joint=0,
        data=SimpleNamespace(xanchor=np.array([[1.,2.,0.]]), xaxis=np.array([[0.,0.,1.]])),
        record=lambda **kw: None)
    contact=np.eye(4); contact[:3,3]=[1.5,2.,.8]
    surface=np.array([1.5,2.02,.8]); direction=np.array([0.,1.,0.])
    saved=contact.copy()
    pose,point,normal,angle=refresh_panel_contact(c,contact,surface,direction,.8)
    rotation=Rotation.from_euler('z',-.1).as_matrix()
    np.testing.assert_allclose(pose[:3,3],[1.,2.,0.]+rotation@np.array([.5,0.,.8]))
    np.testing.assert_allclose(normal,rotation@direction)
    np.testing.assert_allclose(pose[:3,:3],rotation)
    np.testing.assert_array_equal(contact,saved)
    assert angle==.7
