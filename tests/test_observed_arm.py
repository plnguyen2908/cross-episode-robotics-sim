from pathlib import Path
import numpy as np
import pytest
from cross_episode_sim.control.observed_arm import robot_calibration


def test_robot_export_ignores_world_pose_and_scene_state(tmp_path):
    import mujoco
    source=Path('/nobackup/le/cross-episode-robotics-sim/runs/train_test_v1/test_two_rooms/batches/round_continuous/set_table_000000/action_recording/model.mjb')
    if not source.exists():pytest.skip('qualification model unavailable')
    m=mujoco.MjModel.from_binary_path(str(source));d=mujoco.MjData(m)
    first=robot_calibration(m,d,tmp_path/'first')
    d.joint('robot_0/base_x').qpos[0]=12.;d.joint('robot_0/base_y').qpos[0]=-7.;d.joint('robot_0/base_theta').qpos[0]=1.2
    for j in range(m.njnt):
        if not (m.joint(j).name or '').startswith('robot_0/') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j]+3]+=5
    mujoco.mj_forward(m,d)
    second=robot_calibration(m,d,tmp_path/'second')
    assert Path(first['urdf_path']).read_text()==Path(second['urdf_path']).read_text()
    assert first['collision_spheres']==second['collision_spheres']
