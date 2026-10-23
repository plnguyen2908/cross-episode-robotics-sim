import mujoco
import numpy as np
from cross_episode_sim.generation.household_bindings import beneath_sink


def test_sink_excludes_storage_below_but_not_adjacent_or_above():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="sink_main_group_main" pos="1 0 .9">
        <body name="basin"><geom type="box" size=".4 .25 .1"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert beneath_sink(model, data, np.array([.5,-.3,.4]), np.array([1.5,.3,.42]))
    assert not beneath_sink(model, data, np.array([2.,-.3,.4]), np.array([2.7,.3,.42]))
    assert not beneath_sink(model, data, np.array([.5,-.3,1.3]), np.array([1.5,.3,1.32]))
