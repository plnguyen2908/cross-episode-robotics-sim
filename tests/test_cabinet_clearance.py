import mujoco
import numpy as np
import pytest

from cross_episode_sim.generation.cabinet_clearance import repair_cabinet_clearance


def cabinet_spec():
    return mujoco.MjSpec.from_string('''<mujoco>
      <option integrator="implicitfast"/>
      <worldbody><body name="cab_main" pos="0 0 1">
        <geom name="cab_top" type="box" pos="0 .015 .30" size=".35 .285 .015"/>
        <geom name="cab_bottom" type="box" pos="0 .015 -.30" size=".35 .285 .015"/>
        <geom name="cab_left" type="box" pos="-.335 .015 0" size=".015 .285 .285"/>
        <geom name="cab_right" type="box" pos=".335 .015 0" size=".015 .285 .285"/>
        <geom name="cab_back" type="box" pos="0 .285 0" size=".32 .015 .285"/>
        <body name="cab_hingedoor">
          <joint name="hinge" axis="0 0 1" pos=".335 -.3 0" range="0 89.95" damping="2"/>
          <body name="panel" pos="0 -.285 0">
            <geom name="panel_geom" type="box" size=".3485 .015 .3135" mass=".2"/>
          </body>
        </body>
      </body></worldbody></mujoco>''')


def test_clearance_repair_preserves_physics_and_is_idempotent():
    spec = cabinet_spec()
    original = spec.compile()
    positions = original.body_pos.copy()
    records = repair_cabinet_clearance(spec, original, ['cab_hingedoor'])
    assert len(records) == 1
    assert records[0]['outward_offset_m'] == pytest.approx(.004)
    np.testing.assert_array_equal(original.body_pos, positions)
    fixed = spec.compile()
    for attr in ('geom_contype', 'geom_conaffinity', 'geom_size', 'jnt_range',
                 'jnt_stiffness', 'dof_damping', 'dof_frictionloss'):
        np.testing.assert_array_equal(getattr(fixed, attr), getattr(original, attr))
    assert repair_cabinet_clearance(spec, fixed, ['cab_hingedoor']) == []
    for angle in (15., 30., 60., 89.):
        data = mujoco.MjData(fixed)
        data.qpos[0] = np.radians(angle)
        mujoco.mj_step(fixed, data, nstep=5000)
        assert np.degrees(data.qpos[0]) == pytest.approx(angle, abs=.01)


def test_unrequested_cabinet_is_not_modified():
    spec = cabinet_spec()
    model = spec.compile()
    assert repair_cabinet_clearance(spec, model, []) == []
    np.testing.assert_array_equal(spec.compile().body_pos, model.body_pos)
