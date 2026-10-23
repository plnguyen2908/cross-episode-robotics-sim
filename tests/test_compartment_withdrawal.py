from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cross_episode_sim.control.compartment_entry import plan_empty_withdrawal


def test_withdrawal_search_uses_scratch_state_and_preserves_released_object():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="robot_0/lift" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".01" mass="1"/>
      </body></worldbody></mujoco>''')
    live = mujoco.MjData(model)
    c = SimpleNamespace(model=model, data=live, planner=SimpleNamespace(names=['lift']),
                        holding_loaf=False, attached=False)
    def tcp():
        pose=np.eye(4); pose[2,3]=c.data.qpos[0]
        return pose
    def plan(stage, pose):
        assert c.data is not live
        if pose[2,3] < .015:
            raise RuntimeError('Blocked shelf withdrawal')
        return [np.array([pose[2,3]])]
    c.tcp=tcp; c.plan_contact_path=plan
    result=plan_empty_withdrawal(c,[np.eye(4)])
    assert len(result)==2
    assert result[0][0][2,3]==pytest.approx(.02)
    assert c.data is live
    assert live.qpos[0]==0.
    c.plan_contact_path=lambda *a: (_ for _ in ()).throw(RuntimeError('blocked'))
    with pytest.raises(RuntimeError,match='No native-mesh-clear'):
        plan_empty_withdrawal(c,[np.eye(4)])
    assert c.data is live
    assert live.qpos[0]==0.


def test_withdrawal_requires_release():
    with pytest.raises(RuntimeError,match='released object'):
        plan_empty_withdrawal(SimpleNamespace(holding_loaf=True),[])
