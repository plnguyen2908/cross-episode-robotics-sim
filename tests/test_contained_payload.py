from types import SimpleNamespace

import mujoco
import numpy as np

from cross_episode_sim.control.contained_payload import ContainedPayloadMixin


def test_stationary_loaded_carry_keeps_measured_pose_without_ik():
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    attempts=[];checked=[];postures=[]
    def rejected(stage,pose):
        attempts.append(pose)
        raise RuntimeError('No nearby placement IK')
    def forbidden_move(*args,**kwargs):raise AssertionError('No movement requested')
    object_pose=np.eye(4);object_pose[0,3]=.5
    c=SimpleNamespace(object_name='basket',tcp=lambda:np.eye(4),
        bread_pose=lambda:object_pose,base_pose=lambda:np.zeros(3),
        plan_contact_path=rejected,check_loaded_tuck_path=lambda path:checked.append(path),
        mesh_contact_move=forbidden_move,record=lambda **kw:None,
        data=SimpleNamespace(joint=lambda name:SimpleNamespace(qpos=[.2])),
        profile=SimpleNamespace(namespace='',arm_joints=['joint']),
        planner=SimpleNamespace(names=['joint']),
        embodiment=SimpleNamespace(set_loaded_travel_posture=lambda q:postures.append(q)))
    BreakfastEpisode.tuck_vessel_for_navigation(c)
    assert len(attempts)==4
    assert checked==[[[.2]]]
    assert postures==[[.2]]
    assert c._filled_carry_body=='basket'


def test_projected_contents_collide_with_world_without_moving_live_objects():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="basket"><freejoint/><geom type="box" size=".1 .1 .01"/></body>
      <body name="bottle" pos="0 0 .12"><freejoint/><geom type="sphere" size=".05"/></body>
      <body name="obstacle" pos=".5 0 .12"><geom type="sphere" size=".04"/></body>
      </worldbody></mujoco>''')
    live=mujoco.MjData(model);mujoco.mj_forward(model,live)
    c=object.__new__(ContainedPayloadMixin)
    c.model=model;c.data=live;c.object_name='basket'
    c._contained_payload=['bottle'];c._contained_payload_bids={model.body('bottle').id}
    c.bread_bids={model.body('basket').id}
    mask=np.zeros(model.nbody,dtype=bool);mask[model.body('basket').id]=True
    c._contact_floor_mask=np.zeros(model.nbody,dtype=bool)
    c.contact_body_masks=lambda:(np.zeros(model.nbody,dtype=bool),None,mask)
    initial=live.qpos.copy()
    probe=mujoco.MjData(model);mujoco.mj_copyData(probe,model,live)
    probe.qpos[0]=.5;mujoco.mj_forward(model,probe)
    assert c.navigation_penetration(probe,True)>.05
    np.testing.assert_allclose(probe.body('bottle').xpos,[.5,0,.12])
    np.testing.assert_array_equal(live.qpos,initial)
    assert c.loaded_payload_bodies()=={model.body('basket').id,model.body('bottle').id}
    c.project_payload_contents(live)
    np.testing.assert_array_equal(live.qpos,initial)
