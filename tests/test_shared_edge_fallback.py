"""Check shared-controller dispatch and opt-in isolation for edge access."""
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from cross_episode_sim.manipulation.edge_access import ThinEdgeAccessMixin
from cross_episode_sim.manipulation.recovery import RecoveringManipulation


def test_shared_collision_dispatch_only_during_push():
    controller=object.__new__(RecoveringManipulation)
    controller.edge_pushing=True
    with patch.object(ThinEdgeAccessMixin, 'navigation_penetration', return_value=.123) as check:
        assert controller.navigation_penetration(None,False)==.123
        check.assert_called_once_with(controller,None,False)
    controller.edge_pushing=False
    # Outside a push the vectorized contact filter runs instead; with no
    # contacts it reports zero penetration and never calls the edge check.
    controller.model=SimpleNamespace(geom_bodyid=np.zeros(1,dtype=int))
    controller._contact_floor_mask=np.zeros(1,dtype=bool)
    masks=(np.zeros(1,dtype=bool),None,np.zeros(1,dtype=bool))
    empty=SimpleNamespace(contact=SimpleNamespace(geom1=np.zeros(0,dtype=int),geom2=np.zeros(0,dtype=int),dist=np.zeros(0)))
    with patch.object(ThinEdgeAccessMixin, 'navigation_penetration') as check, \
            patch.object(RecoveringManipulation, 'contact_body_masks', return_value=masks):
        assert controller.navigation_penetration(empty,False)==0.
        check.assert_not_called()


def test_disabled_fallback_delegates_without_geometry():
    class Parent:
        def prepare_pickup(self):return 'ordinary pickup'
    class Controller(ThinEdgeAccessMixin,Parent):pass
    controller=Controller();controller.args=SimpleNamespace()
    assert controller.prepare_pickup()=='ordinary pickup'


@pytest.mark.parametrize('fail',[False,True])
def test_pick_restores_aperture_without_commanding_held_gripper(fail):
    @dataclass
    class Profile:gripper_open:float=0.
    class Parent:
        def pick_payload(self):
            self._edge_original_gripper_open=0.
            self.profile=Profile(180.)
            if fail:raise RuntimeError('physical failure')
            return 'holding'
    class Controller(ThinEdgeAccessMixin,Parent):pass
    controller=Controller();controller.profile=Profile();controller.embodiment=SimpleNamespace();controller.args=SimpleNamespace()
    if fail:
        with pytest.raises(RuntimeError,match='physical failure'):controller.pick_payload()
    else:assert controller.pick_payload()=='holding'
    assert controller.profile.gripper_open==0.
    assert controller.embodiment.profile is controller.profile


def test_flat_placement_rotates_with_destination_stance_not_travel_arm():
    import numpy as np
    from scipy.spatial.transform import Rotation
    controller=object.__new__(RecoveringManipulation)
    controller.args=SimpleNamespace(thin_edge_fallback=True)
    controller._edge_flat_rotation=Rotation.from_euler('z',12,degrees=True).as_matrix()
    controller._edge_pick_yaw=-np.pi/2
    carried=np.eye(4);carried[:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()
    controller.bread_pose=lambda:carried.copy()
    controller.base_pose=lambda:np.array([2.35,-.97,np.pi/2])
    controller.record=lambda **kw:None
    flat=controller.placement_reference_pose()
    np.testing.assert_allclose(flat[:3,2],[0,0,1],atol=1e-12)
    np.testing.assert_allclose(flat[:3,:3],Rotation.from_euler('z',192,degrees=True).as_matrix(),atol=1e-12)
    controller.args.thin_edge_fallback=False
    np.testing.assert_array_equal(controller.placement_reference_pose(),carried)


def test_regenerated_grasps_keep_previous_failed_contacts(tmp_path):
    import json
    import numpy as np
    old=np.eye(4);new=np.eye(4);new[0,3]=.02
    prior=tmp_path/'previous/trial_000';prior.mkdir(parents=True)
    (prior/'attempted_grasp.json').write_text(json.dumps({'planned_object_T_tcp':old.tolist()}))
    controller=object.__new__(RecoveringManipulation)
    controller.args=SimpleNamespace(previous_trials=prior.parent)
    controller.output=tmp_path
    controller.bread_vertices=lambda:None
    controller.bread_pose=lambda:np.eye(4)
    controller.record=lambda **kw:None
    with patch('cross_episode_sim.manipulation.edge_access.side_annotations',return_value=np.array([old,new])):
        controller.refresh_edge_grasps([0,1])
    assert controller.physically_rejected_annotation_variants=={0}
    assert controller.args.annotation_source=='geometry_hypotheses'
    assert controller.annotation_path.is_file()


@pytest.mark.parametrize('supported',[False,True])
def test_physical_retry_requires_remaining_table_support(supported):
    import numpy as np
    calls=[]
    class Parent:
        def pick_payload(self):
            calls.append('pick')
            if len(calls)==1:raise RuntimeError('Object did not clear the counter during vertical lift')
            return 'held'
    class Controller(ThinEdgeAccessMixin,Parent):pass
    c=Controller();c.args=SimpleNamespace(thin_edge_fallback=True)
    c.support_contacts=lambda **kw:supported
    c.report={'annotation_selection':{'local_transform':np.eye(4).tolist()}}
    c.record=lambda **kw:None
    c.embodiment=SimpleNamespace(open_gripper=lambda owner:calls.append('open'))
    c.tick=lambda dt:None;c.tcp=lambda:np.eye(4)
    c.mesh_contact_move=lambda *args:None
    if supported:
        assert c.pick_payload()=='held'
        assert c._edge_force_push
        assert len(c._edge_failed_poses)==1
        assert calls==['pick','open','pick']
    else:
        with pytest.raises(RuntimeError,match='did not clear'):c.pick_payload()
        assert calls==['pick']


@pytest.mark.parametrize('flat,fingers,supported,expected',[(True,2,False,2),(True,1,False,1),(True,2,True,1),(False,2,False,1)])
def test_complete_short_lift_only_for_flat_objects_held_clear(flat,fingers,supported,expected):
    import numpy as np
    moves=[]
    class Parent:
        def move(self,stage,pose):moves.append((stage,pose.copy()))
    class Controller(ThinEdgeAccessMixin,Parent):pass
    c=Controller();c.args=SimpleNamespace(thin_edge_fallback=True);c.holding_loaf=True
    if flat:c._edge_flat_rotation=np.eye(3)
    c.pickup_start_height=.8
    pose=np.eye(4);pose[2,3]=.8966
    c.bread_pose=lambda:pose;c.tcp=lambda:pose.copy()
    c.support_contacts=lambda **kw:supported
    c.finger_object_contact=lambda:{'fingers':list(range(fingers))}
    c.record=lambda **kw:None
    c.move('lift bread',pose)
    assert len(moves)==expected
    if expected==2:assert moves[-1][1][2,3]==pytest.approx(.92)


@pytest.mark.parametrize('initial_lift,supported,efficiency,fingers',[
    (.012671,False,1.,2), (.004,True,.55,2), (.025,False,1.,2),
    (.004,True,0.,2), (.004,True,1.,1)])
def test_initial_clearance_uses_measured_feedback(initial_lift,supported,efficiency,fingers):
    import numpy as np
    moves=[]
    state={'lift':initial_lift,'gap':-.0004 if supported else .012}
    class Parent:
        def move(self,stage,pose):
            moves.append((stage,pose.copy()))
            if stage=='lift bread clearance correction':
                dz=pose[2,3]-self.tcp()[2,3]
                state['lift']+=efficiency*dz
                state['gap']+=efficiency*dz
    class Controller(ThinEdgeAccessMixin,Parent):pass
    c=Controller();c.args=SimpleNamespace(thin_edge_fallback=True);c.holding_loaf=True
    c.model=c.data=None;c.support_bids={1};c.pickup_start_height=.85
    c.support_contacts=lambda **kw:state['gap']<=0
    c.finger_object_contact=lambda:{'fingers':list(range(fingers))}
    c.bread_vertices=lambda:np.array([[0.,0.,.8+state['gap']]])
    def pose():
        p=np.eye(4);p[2,3]=c.pickup_start_height+state['lift'];return p
    c.bread_pose=pose;c.tcp=pose;c.record=lambda **kw:None
    with patch('cross_episode_sim.manipulation.edge_access.tabletop_bounds',return_value=np.array([[0,0,.8]])):
        c.move('lift bread vertically',pose())
    if efficiency and fingers==2:
        assert state['lift']>=.020
        assert state['gap']>=.010
        if efficiency<1:assert len(moves)>2
    elif fingers==1:assert len(moves)==1
    else:assert len(moves)==5


@pytest.mark.parametrize('lift,supported,fingers',[(.06,False,2),(.016,False,2),(.004,True,1)])
def test_successful_lift_or_lost_grip_does_not_require_tabletop(lift,supported,fingers):
    import numpy as np
    c=object.__new__(RecoveringManipulation)
    c.pickup_start_height=.8
    pose=np.eye(4);pose[2,3]=.8+lift
    c.bread_pose=lambda:pose
    c.support_contacts=lambda **kw:supported
    c.finger_object_contact=lambda:{'fingers':list(range(fingers))}
    with patch.object(c,'pickup_support_top',side_effect=ValueError('No physical horizontal tabletop')) as surface:
        c.clear_held_pickup_support()
    surface.assert_not_called()


def test_breakfast_clearance_support_accepts_thick_furniture():
    import mujoco
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c=object.__new__(BreakfastEpisode)
    c.source='living_table'
    c.model=mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="living_table">
      <geom type="box" pos="0 0 .65" size=".5 .4 .15"/>
    </body></worldbody></mujoco>""")
    c.data=mujoco.MjData(c.model);mujoco.mj_forward(c.model,c.data)
    assert c.pickup_support_top()==pytest.approx(.8)
