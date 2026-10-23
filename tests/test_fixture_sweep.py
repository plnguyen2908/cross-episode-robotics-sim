import mujoco
import numpy as np
from cross_episode_sim.generation.fixture_sweep import door_sweep_boxes


def test_sweep_reserves_open_arc_and_preserves_live_state():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="door"><joint name="hinge" axis="0 0 1" range="0 90"/>
      <geom type="box" pos=".25 0 1" size=".25 .01 .2"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:] = .2
    data.time = 7.
    mujoco.mj_forward(model, data)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    boxes = door_sweep_boxes(model, data, dict(joint='hinge', body='door'))
    # A prop ahead of the closed panel is clear initially but hit at 90 degrees.
    point = np.array([0., .4, 1.])
    assert not np.all((point >= boxes[0][0]) & (point <= boxes[0][1]))
    assert any(np.all((point >= lo) & (point <= hi)) for lo, hi in boxes)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    assert data.time == 7.
