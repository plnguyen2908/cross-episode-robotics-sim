"""Final grocery score: what counts as being in a basket (user decision, November 13)."""
import json
import mujoco
import numpy as np
import pytest
from cross_episode_sim.finetuning.replay_scoring import CompositeReplayScorer

SCENE='''<mujoco><option timestep=".002"/><worldbody>
    <body name="counter"><geom type="box" pos="0 0 .04" size=".5 .5 .04"/></body>
    <body name="table" pos="3 0 0"><geom type="box" pos="0 0 .04" size=".5 .5 .04"/></body>
    <body name="purple_basket_prop" pos="0 0 .1">
      <geom name="purple_basket_prop_floor" type="box" size=".14 .14 .01"/>
      <geom type="box" pos="-.135 0 .06" size=".005 .14 .05"/>
      {inner}
    </body>
    <body name="orange_basket_prop" pos="1 0 .1"><geom name="orange_basket_prop_floor" type="box" size=".14 .14 .01"/></body>
    {outer}
    <body name="lower" pos="-.055 0 .151"><freejoint/><geom type="box" size=".075 .07 .04" mass="1"/></body>
    <body name="leaning" pos=".07 0 .224"><freejoint/><geom type="sphere" size=".06" mass=".1"/></body>
    <body name="robot_0/gripper/pad" mocap="true" pos="0 0 2"><geom type="sphere" size=".01"/></body>
    </worldbody></mujoco>'''
WALL='<geom type="box" pos="{x} 0 {z}" size=".005 .14 .05"/>'


def scored(tmp_path,*,backsplash=False,leaning='drink_bottle',at=None):
    """A ball wedged between a lying bottle's top edge and a wall, clear of the basket floor."""
    inner,outer=('','<body name="backsplash">'+WALL.format(x=.135,z=.16)+'</body>') if backsplash else (WALL.format(x=.135,z=.06),'')
    m=mujoco.MjModel.from_xml_string(SCENE.format(inner=inner,outer=outer));d=mujoco.MjData(m)
    if at is not None:d.qpos[m.jnt_qposadr[m.body_jntadr[m.body('leaning').id]]:][:3]=at
    for _ in range(1000):mujoco.mj_step(m,d)
    bindings=[dict(role='drink_bottle_01',body='lower',category='drink_bottle'),dict(role='drink_bottle_02',body='leaning',category=leaning),
              dict(role='purple_basket',body='purple_basket_prop',category='basket'),dict(role='orange_basket',body='orange_basket_prop',category='basket')]
    initial=dict(objects={b['role']:dict(category=b['category'],**({'color':b['role'].split('_')[0]} if b['category']=='basket' else {})) for b in bindings})
    for name,value in (('config.json',dict(task='unpack_and_sort_groceries',scene={})),('initial_snapshot.json',initial),
                       ('task_manifest.json',dict(bindings=bindings,supports=dict(kitchen='counter',dining='table'),surfaces={}))):
        (tmp_path/name).write_text(json.dumps(value))
    return m,d,CompositeReplayScorer(m,d,tmp_path),initial


def test_leaning_item_held_up_only_by_the_basket_counts_as_inside(tmp_path):
    m,d,scorer,initial=scored(tmp_path)
    ball=m.body('leaning').id
    normals=[abs(c.frame[2]) for c in d.contact if ball in m.geom_bodyid[[c.geom1,c.geom2]]]
    # No near-vertical support normal: the earlier rule left this item unsupported.
    assert len(normals)>=2 and max(normals)<.7
    state=scorer.snapshot(d)['objects']['drink_bottle_02']
    assert (state['support'],state['region'])==('purple_basket','interior')
    assert scorer.score(initial,d)['drinks_in_purple_basket']


def test_item_touching_counted_contents_counts_even_when_it_also_leans_outside(tmp_path):
    m,d,scorer,initial=scored(tmp_path,backsplash=True)
    state=scorer.snapshot(d)['objects']['drink_bottle_02']
    assert (state['support'],state['region'])==('purple_basket','interior')


def test_item_on_the_counter_against_the_outside_of_a_wall_does_not_count(tmp_path):
    m,d,scorer,initial=scored(tmp_path,at=(-.3,0,.14))
    d.qpos[m.jnt_qposadr[m.body_jntadr[m.body('leaning').id]]]=-.1998;d.qvel[:]=0;mujoco.mj_forward(m,d)
    ball=m.body('leaning').id;basket=m.body('purple_basket_prop').id
    assert any({ball,basket}=={*m.geom_bodyid[[c.geom1,c.geom2]]} and c.dist<=.002 for c in d.contact)
    assert scorer.snapshot(d)['objects']['drink_bottle_02']['region']=='default'
    assert not scorer.score(initial,d)['drinks_in_purple_basket']


def test_item_standing_on_the_counter_and_touching_contents_does_not_count(tmp_path):
    """Leaning on a counted item is not enough while the counter carries it."""
    m,d,scorer,initial=scored(tmp_path,backsplash=True,at=(.2,0,.14))
    d.qpos[m.jnt_qposadr[m.body_jntadr[m.body('lower').id]]]=.065;mujoco.mj_forward(m,d)
    for _ in range(300):mujoco.mj_step(m,d)
    state=scorer.snapshot(d)['objects']
    assert state['drink_bottle_01']['region']=='interior' and state['drink_bottle_02']['support']=='kitchen_counter'


def test_moving_or_gripped_item_does_not_count(tmp_path):
    m,d,scorer,initial=scored(tmp_path)
    saved=d.qvel.copy();adr=m.jnt_dofadr[m.body_jntadr[m.body('leaning').id]]
    d.qvel[adr+2]=-.2;mujoco.mj_forward(m,d)
    assert scorer.snapshot(d)['objects']['drink_bottle_02']['support'] is None
    d.qvel[:]=saved;d.mocap_pos[0]=d.body('leaning').xpos+np.array([0,0,.069]);mujoco.mj_forward(m,d)
    state=scorer.snapshot(d)
    assert state['held']=='drink_bottle_02' and state['objects']['drink_bottle_02']['support'] is None


def test_leaning_item_of_the_wrong_category_is_a_wrong_basket_item(tmp_path):
    m,d,scorer,initial=scored(tmp_path,leaning='snack_box')
    checks=scorer.score(initial,d)
    assert not checks['no_wrong_basket_items'] and not checks['snacks_in_orange_basket']
