"""Scratch geometric queries must retain full-forward collision decisions."""

import mujoco
import numpy as np
import pytest

from cross_episode_sim.control.collision_probe import forward_collision


@pytest.mark.parametrize('jacobian', ['dense', 'sparse'])
def test_collision_probe_matches_full_forward_with_changing_contacts(jacobian):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option jacobian="{jacobian}" noslip_iterations="5"/>
      <worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body name="cup" pos="0 0 .08">
          <freejoint/><geom type="box" size=".06 .06 .08" margin=".002"/>
          <site pos="0 0 .08"/>
        </body>
        <body name="particle" pos="0 0 .18">
          <freejoint/><geom type="sphere" size=".02"/>
        </body>
        <body pos=".12 0 .08">
          <joint type="slide" axis="1 0 0" range="-.05 .05"/>
          <geom type="box" size=".02 .02 .04" condim="4"/>
        </body>
      </worldbody>
    </mujoco>''')
    full, probe = mujoco.MjData(model), mujoco.MjData(model)
    initial = full.qpos.copy()
    rng = np.random.default_rng(7)
    counts = set()
    for _ in range(40):
        qpos = initial.copy()
        mujoco.mj_integratePos(model, qpos, rng.normal(size=model.nv), .015)
        velocity = rng.normal(size=model.nv)
        for data in (full, probe):
            data.qpos[:] = qpos
            data.qvel[:] = velocity
            data.time = 2.
        mujoco.mj_forward(model, full)
        forward_collision(model, probe)
        counts.add(full.ncon)
        for field in ('xpos', 'xmat', 'geom_xpos', 'site_xpos', 'site_xmat', 'cdof'):
            np.testing.assert_array_equal(getattr(full, field), getattr(probe, field))
        for field in ('geom1', 'geom2', 'dist', 'pos', 'frame', 'includemargin'):
            np.testing.assert_array_equal(getattr(full.contact, field), getattr(probe.contact, field))
        np.testing.assert_array_equal(probe.qpos, qpos)
        np.testing.assert_array_equal(probe.qvel, velocity)
        assert probe.time == 2.
    assert len(counts) > 1
