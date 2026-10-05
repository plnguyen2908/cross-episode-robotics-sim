"""Physical retention checks use free-body contacts and location, not vessel tilt."""
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.physical_contents import (
    cavity_geometry, initialize_contents, activate_contents, contents_supported, retained)


def rig():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
      <body name="cup"><geom type="box" pos="0 0 .005" size=".05 .05 .005"/>
        <geom type="box" pos="-.045 0 .045" size=".005 .05 .04"/>
        <geom type="box" pos=".045 0 .045" size=".005 .05 .04"/>
        <geom type="box" pos="0 -.045 .045" size=".04 .005 .04"/>
        <geom type="box" pos="0 .045 .045" size=".04 .005 .04"/></body>
      <body name="food" pos="20 20 -10" gravcomp="1"><freejoint name="food_joint"/>
        <geom type="box" size=".01 .01 .01" mass=".002"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    item=dict(role='cup_one',body='food',joint='food_joint',cavity=cavity_geometry(m,d,'cup'),
        initial_rotation=np.eye(3).tolist(),origin_to_center=[0,0,0],origin_to_bottom=-.01,half_extents=[.01]*3)
    c=SimpleNamespace(model=m,data=d,manifest={'physical_contents':[item],
        'bindings':[dict(role='cup_one',body='cup')]})
    initialize_contents(c)
    return c,item


def test_fill_enables_free_object_gravity_and_settles_with_real_contact():
    c,item=rig();m,d=c.model,c.data
    assert not np.any(m.geom_contype[[g for g in c.content_collision_flags]])
    activate_contents(c)
    assert m.body_gravcomp[m.body('food').id]==0
    assert m.neq==0 and m.nq==7  # no weld or parent attachment
    high=d.body('food').xpos[2]
    mujoco.mj_step(m,d,nstep=500)
    assert d.body('food').xpos[2]<high-.02
    assert contents_supported(c,item)
    # A visible displaced food body is a spill, even with the cup upright.
    d.joint('food_joint').qpos[0]+=.2;mujoco.mj_forward(m,d)
    assert not retained(m,d,item,'cup')
    assert not contents_supported(c,item)


def test_retention_does_not_fail_only_because_vessel_is_tilted():
    c,item=rig();m,d=c.model,c.data
    activate_contents(c);mujoco.mj_step(m,d,nstep=500)
    position=d.body('food').xpos.copy()
    rotation=Rotation.from_euler('x',70,degrees=True)
    m.body_quat[m.body('cup').id]=rotation.as_quat(scalar_first=True)
    d.joint('food_joint').qpos[:3]=rotation.apply(position)
    d.joint('food_joint').qpos[3:]=rotation.as_quat(scalar_first=True)
    mujoco.mj_forward(m,d)
    assert retained(m,d,item,'cup')


def test_contact_outside_cavity_is_not_counted_as_filled():
    c,item=rig();m,d=c.model,c.data
    activate_contents(c)
    d.joint('food_joint').qpos[:3]=[.07,0,.02]
    mujoco.mj_forward(m,d)
    assert not contents_supported(c,item)


def test_scratch_payload_projection_never_changes_live_food_pose():
    from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="cup"><freejoint name="cup_joint"/><geom type="box" size=".05 .05 .05"/></body>
      <body name="food" pos=".01 0 .03"><freejoint name="food_joint"/><geom type="box" size=".01 .01 .01"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=GatherBreakfast.__new__(GatherBreakfast);c.model=m;c.data=d;c.object_name='cup'
    c.object_info={'cup':{'role':'cup_one'}};c.content_states={'cup_one':'filled'}
    c.content_body_ids={'cup_one':{m.body('food').id}}
    c.physical_content_items=[dict(body='food',joint='food_joint')]
    original=d.qpos.copy()
    c.project_payload_contents(d)
    np.testing.assert_array_equal(d.qpos,original)
    probe=mujoco.MjData(m);probe.qpos[:]=d.qpos
    probe.joint('cup_joint').qpos[:3]=[1,0,0]
    probe.joint('cup_joint').qpos[3:]=Rotation.from_euler('z',90,degrees=True).as_quat(scalar_first=True)
    mujoco.mj_forward(m,probe)
    c.project_payload_contents(probe)
    np.testing.assert_allclose(probe.body('food').xpos,[1,.01,.03],atol=1e-9)
    np.testing.assert_array_equal(d.qpos,original)


def test_physical_config_does_not_require_legacy_tilt_threshold():
    import json
    from cross_episode_sim.tasks.breakfast.gather import DEFAULT_CONFIG, validate_gather_config
    config=json.loads(DEFAULT_CONFIG.read_text())
    config.pop('spill_tilt_deg',None)
    validate_gather_config(config)
