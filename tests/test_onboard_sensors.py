import json
from dataclasses import fields
from pathlib import Path
import mujoco
import numpy as np
import pytest
from cross_episode_sim.sensors import OnboardSensorInterface,SensorPacket

SOURCE=Path('/nobackup/le/cross-episode-robotics-sim/runs/train_test_v1/test_two_rooms/batches/round_continuous/set_table_000000/action_recording')

@pytest.fixture
def scene():
    if not SOURCE.exists():pytest.skip('Saved qualification scene unavailable')
    m=mujoco.MjModel.from_binary_path(str(SOURCE/'model.mjb'));d=mujoco.MjData(m)
    with np.load(SOURCE/'initial_state.npz') as x:mujoco.mj_setState(m,d,x['state'],int(x['state_type']))
    mujoco.mj_forward(m,d)
    with OnboardSensorInterface(m,d,160,120) as sensor:yield m,d,sensor


def test_packet_allowlist_and_synchronization(scene):
    m,d,s=scene;p=s.capture()
    assert {f.name for f in fields(SensorPacket)}=={'timestamp','cameras','arm','gripper','base_frame'}
    assert set(p.cameras)=={'navigation','wrist'}
    assert p.cameras['navigation'].robot_mask.shape==(120,160)
    assert .005 < p.cameras['navigation'].robot_mask.mean() < .8
    assert p.timestamp==d.time
    assert len(p.arm.names)==7 and len(p.gripper.names)==2
    assert all('base_' not in n for n in p.arm.names+p.gripper.names)
    for c in p.cameras.values():
        assert c.rgb.shape==(120,160,3) and c.depth_m.shape==(120,160)
        assert np.array_equal(c.valid_depth,np.isfinite(c.depth_m))
        assert np.all(c.depth_m[c.valid_depth]>0)
        assert np.all(c.depth_m[c.valid_depth]<=8)
        assert np.allclose(c.T_base_optical[3],[0,0,0,1])
        assert np.isclose(np.linalg.det(c.T_base_optical[:3,:3]),1)
        json.dumps(c.camera_info())


def test_calibration_and_kinematics(scene):
    m,d,s=scene
    for q in [0.,.3,-.2]:
        d.qpos[m.jnt_qposadr[m.joint('robot_0/fr3_joint2').id]]=q
        mujoco.mj_forward(m,d);p=s.capture()
        for k,c in p.cameras.items():
            cid=m.camera(s.CAMERA_NAMES[k]).id;bid=m.body('robot_0/base').id
            R=d.xmat[bid].reshape(3,3)
            assert np.allclose(c.T_base_optical[:3,:3],R.T@d.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]),atol=1e-10)
            assert np.allclose(c.T_base_optical[:3,3],R.T@(d.cam_xpos[cid]-d.xpos[bid]),atol=1e-10)
            assert c.K[0,2]==79.5 and c.K[1,2]==59.5
            assert np.isclose(c.K[1,1],60/np.tan(np.deg2rad(m.cam_fovy[cid]/2)))


def test_no_base_pose_in_extrinsics_or_proprioception(scene):
    m,d,s=scene;p=s.capture()
    for n,delta in [('base_x',3),('base_y',-4),('base_theta',1.2)]:d.qpos[m.jnt_qposadr[m.joint('robot_0/'+n).id]]+=delta
    mujoco.mj_forward(m,d);q=s.capture()
    for k in p.cameras:assert np.array_equal(p.cameras[k].T_base_optical,q.cameras[k].T_base_optical)
    assert np.array_equal(p.cameras['navigation'].robot_mask,q.cameras['navigation'].robot_mask)
    assert np.array_equal(p.arm.position,q.arm.position)
    assert np.array_equal(p.gripper.position,q.gripper.position)


def test_capture_does_not_step_or_mutate_physics(scene):
    m,d,s=scene;q=d.qpos.copy();v=d.qvel.copy();ctrl=d.ctrl.copy();time=d.time
    mount=m.cam_pos.copy();collisions=m.geom_contype.copy();packet=s.capture()
    assert d.time==time
    for a,b in [(q,d.qpos),(v,d.qvel),(ctrl,d.ctrl),(mount,m.cam_pos),(collisions,m.geom_contype)]:assert np.array_equal(a,b)
    packet.arm.position[:]=100
    assert np.array_equal(q,d.qpos)


def test_isolated_depth_returns_are_unknown_not_filled():
    from cross_episode_sim.sensors.onboard import depth_validity
    d=np.full((9,9),2.,np.float32);d[4,4]=.3
    valid=depth_validity(d)
    assert not valid[4,4] and valid.sum()==80
    assert d[4,4]==np.float32(.3)  # raw measurement unchanged
    d[:,4]=.3  # a one-pixel-wide connected pole must survive
    assert depth_validity(d).all()
    d[0,0]=np.nan;d[0,1]=np.inf;d[0,2]=0.;d[0,3]=9.
    assert not depth_validity(d)[0,:4].any()


def test_depth_edges_do_not_create_unmasked_robot_top_plate():
    source=Path('runs/set_table_pipeline_development/full_attempt_13/development/transfer_01_bowl/carry/private_checkpoint')
    if not source.exists():pytest.skip('Recorded depth-edge regression state unavailable')
    m=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,np.load(source/'state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    before=d.qpos.copy()
    with OnboardSensorInterface(m,d) as sensor:
        camera=sensor.capture().cameras['wrist']
        for u,v in [(100,272),(97,261),(108,301)]:
            assert camera.robot_mask[v,u] or not camera.valid_depth[v,u]
        assert m.vis.quality.offsamples==0
        assert np.array_equal(d.qpos,before)
