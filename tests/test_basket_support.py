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


WEDGE='''<mujoco><option timestep=".002"/><worldbody>
    <body name="purple_basket_prop" pos="0 0 .1">
      <geom name="purple_basket_prop_floor" type="box" size=".14 .14 .01"/>
      <geom name="purple_basket_prop_side_x-1" type="box" pos="-.135 0 .06" size=".005 .14 .05"/>
      <geom name="{wall}" type="box" pos=".135 0 .06" size=".005 .14 .05"/>
    </body>
    <body name="lower" pos="-.055 0 .151"><freejoint/><geom type="box" size=".075 .07 .04" mass="1"/></body>
    <body name="leaning" pos=".07 0 .224"><freejoint/><geom type="sphere" size=".06" mass=".1"/></body>
    </worldbody></mujoco>'''


def wedged(wall='purple_basket_prop_side_x1'):
    """A ball held between the lying item's top edge and one wall, clear of the floor."""
    m=mujoco.MjModel.from_xml_string(WEDGE.format(wall=wall));d=mujoco.MjData(m)
    for _ in range(1000):mujoco.mj_step(m,d)
    ball=m.body('leaning').id
    normals=[abs(c.frame[2]) for c in d.contact if ball in m.geom_bodyid[[c.geom1,c.geom2]]]
    # The case the old rule missed: touching, but no near-vertical support normal.
    assert len(normals)>=2 and max(normals)<.7
    return m,d,ball


def test_item_leaning_on_contents_and_a_wall_is_in_the_basket():
    m,d,ball=wedged()
    result=basket_release_score(m,d,{ball},'purple basket')
    assert result['success'] and ball in result['released_supported']


def test_item_leaning_on_something_outside_the_basket_is_not_in_it():
    m,d,ball=wedged(wall='counter_backsplash')
    assert not basket_release_score(m,d,{ball},'purple basket')['success']
