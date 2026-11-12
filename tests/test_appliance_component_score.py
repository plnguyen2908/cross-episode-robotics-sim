"""Private fixture-source identity remains separate from command completion."""
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.evaluation.qwen_component_score import ComponentScore


def score_with_held(held):
    score=ComponentScore.__new__(ComponentScore)
    score.initial_appliance_objects={'fridge':{1},'microwave':{2}}
    score.initial_heights={1:.8,2:1.}
    score.touching_hand=lambda data:set(held)
    return score


def test_pick_from_requires_the_object_from_requested_appliance():
    data=SimpleNamespace(xpos=np.array([[0,0,0],[0,0,1.],[0,0,1.2]]))
    score=score_with_held({2})
    assert not score.pick_from(data,'fridge')['success']
    assert score.pick_from(data,'microwave')['success']


def test_pick_from_requires_physical_hold_and_lift():
    data=SimpleNamespace(xpos=np.array([[0,0,0],[0,0,.82],[0,0,1.2]]))
    assert not score_with_held({1}).pick_from(data,'fridge')['success']
    assert not score_with_held(set()).pick_from(data,'microwave')['success']


def test_basket_component_requires_containment_and_release():
    import mujoco
    xml='''<mujoco><worldbody>
      <body name="purple_basket_prop"><geom name="purple_basket_prop_floor" type="box" size=".13 .15 .01"/>
        <geom name="purple_basket_prop_side" type="box" pos=".14 0 .05" size=".01 .15 .05"/></body>
      <body name="drink_bottle" pos="0 0 .035"><freejoint/><geom type="sphere" size=".025" mass=".1"/></body>
    </worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    for _ in range(200):mujoco.mj_step(m,d)
    score=ComponentScore(m,d);root=m.body('drink_bottle').id;score.initially_held={root}
    assert score.placement(d,'grocery_in_basket')['success']
    # A sphere on the rim has support but is not contained in the basket floor.
    d.qpos[:3]=[.14,0.,.125];d.qvel[:]=0
    for _ in range(200):mujoco.mj_step(m,d)
    assert not score.placement(d,'grocery_in_basket')['success']
    d.qpos[:3]=[0.,0.,.035];d.qvel[:]=0
    for _ in range(200):mujoco.mj_step(m,d)
    score.touching_hand=lambda data:{root}
    assert not score.placement(d,'grocery_in_basket')['success']
