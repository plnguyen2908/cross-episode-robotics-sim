import mujoco
from cross_episode_sim.evaluation.basket_support import basket_release_score


def test_stacked_item_can_be_supported_by_basket_through_another_item():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
    <geom name="purple_basket_floor" type="box" pos="0 0 .1" size=".2 .2 .03"/>
    <body name="lower" pos="0 0 .181"><freejoint/><geom type="box" size=".07 .07 .05" mass=".2"/></body>
    <body name="upper" pos="0 0 .284"><freejoint/><geom type="box" size=".04 .04 .05" mass=".1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m)
    for _ in range(600):mujoco.mj_step(m,d)
    upper=m.body('upper').id
    result=basket_release_score(m,d,{upper},'purple basket')
    assert result['success'] and upper in result['released_supported']
    assert not basket_release_score(m,d,{upper},'orange basket')['success']


def test_item_overhanging_basket_is_not_success_even_if_supported():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom name="purple_basket_floor" type="box" pos="0 0 .1" size=".1 .1 .03"/>
    <body name="wide" pos="0 0 .18"><freejoint/><geom type="box" size=".2 .2 .05" mass=".2"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m)
    for _ in range(600):mujoco.mj_step(m,d)
    assert not basket_release_score(m,d,{m.body('wide').id},'purple basket')['success']
