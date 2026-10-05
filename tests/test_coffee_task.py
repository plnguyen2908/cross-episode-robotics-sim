"""Coffee task never substitutes a symbolic pour or premature brew success."""
import numpy as np
import pytest

from cross_episode_sim.tasks.coffee.grounds import BrewState, grain_offsets, grain_inventory, pour_pose


def test_box_has_open_cavity_and_retains_free_grains_at_rotated_initial_pose():
    import xml.etree.ElementTree as ET
    import mujoco
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.tasks.coffee.grounds import make_dosing_box, add_grounds
    root=ET.Element('mujoco');world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'geom',type='plane',size='1 1 .1')
    body=ET.SubElement(world,'body',name='box',pos='0 0 .04')
    ET.SubElement(body,'freejoint')
    cavity=make_dosing_box(body)
    rotation=Rotation.from_euler('z',37,degrees=True)
    body.set('quat',' '.join(map(str,rotation.as_quat(scalar_first=True))))
    grains=add_grounds(world,[0,0,.04],cavity,32,rotation.as_matrix())
    model=mujoco.MjModel.from_xml_string(ET.tostring(root).decode());data=mujoco.MjData(model)
    mujoco.mj_step(model,data,nstep=1000)
    spec=dict(dosing_body='box',dosing_cavity=cavity,grains=grains,
              hopper=dict(center_xy=[2.,2.],half_width=.054,bottom_z=0.,top_z=.1))
    assert len(grain_inventory(model,data,spec)['vessel'])==32
    assert np.max(data.xpos[[model.body(n).id for n in grains],2])<.035
    # A corner is inside the rectangular walls but outside the old circular
    # mug approximation. It must stay with the payload during planning.
    pos=data.body('box').xpos+data.body('box').xmat.reshape(3,3)@np.array([.029,.024,0.])
    jid=model.body_jntadr[model.body(grains[0]).id]
    data.qpos[model.jnt_qposadr[jid]:model.jnt_qposadr[jid]+3]=pos
    mujoco.mj_forward(model,data)
    assert grains[0] in grain_inventory(model,data,spec)['vessel']


def test_button_requires_all_prerequisites_and_a_fresh_press():
    b=BrewState()
    assert b.update(0,True,False,True,True)=='start_rejected_missing_prerequisite'
    assert b.update(1,True,True,True,True) is None
    assert b.started_at is None
    b.update(2,False,True,True,True)
    assert b.update(3,True,True,True,True)=='brew_started'
    assert b.update(9,True,True,True,True) is None
    assert not b.completed  # must release button
    assert b.update(10,False,True,True,True)=='brew_complete'


@pytest.mark.parametrize('missing',(0,1,2))
def test_brew_aborts_if_grounds_lid_or_mug_prerequisite_is_lost(missing):
    b=BrewState();b.update(0,True,True,True,True)
    ready=[True]*3;ready[missing]=False
    assert b.update(1,False,*ready)=='brew_aborted'
    assert not b.completed and b.started_at is None


def test_completed_brew_survives_serving_mug():
    b=BrewState();b.update(0,True,True,True,True);b.update(6,False,True,True,True)
    b.update(7,False,True,True,False)
    assert b.completed


def test_grains_fit_without_overlap_and_pour_lip_stays_above_hopper():
    cavity=dict(center_xy=[0.,0.],radius=.035,bottom_z=-.05,rim_z=.05,reference_rotation=np.eye(3).tolist())
    points=grain_offsets(cavity)
    distances=np.linalg.norm(points[:,None]-points[None,:],axis=2)+np.eye(32)
    assert distances.min()>.007
    spec=dict(dosing_cavity=cavity,hopper=dict(center_xy=[1.,2.],top_z=1.))
    for angle in (0,30,60,90,120):
        p=pour_pose(spec,angle)
        lip=p[:3,3]+p[:3,:3]@np.array([0.,.035,.05])
        np.testing.assert_allclose(lip[:2],[1.,2.])
        assert 1.0249<=lip[2]<=1.1251
        if angle>=75:assert lip[2]==pytest.approx(1.035)
    with pytest.raises(ValueError,match='narrow'):
        grain_offsets(dict(cavity,radius=.01))


def test_lid_counter_targets_keep_grasp_rotation_and_supported_height():
    from types import SimpleNamespace
    from unittest.mock import patch
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
    c=CoffeeEpisode.__new__(CoffeeEpisode)
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('zy',[37,4],degrees=True).as_matrix();pose[:3,3]=[2.,-.5,1.15]
    local=np.array([[x,y,z] for x in (-.07,.07) for y in (-.07,.07) for z in (-.012,.05)])
    c.bread_pose=lambda:pose.copy();c.bread_vertices=lambda:local@pose[:3,:3].T+pose[:3,3]
    c.model=SimpleNamespace(ngeom=0);c.data=None;c.counter=c.destination='counter'
    c.table_bids={'counter':set()};c.lid_placement_offsets=[(0.,0.),(.06,0.)]
    c.spec={'lid_parking_xy':[1.82,-.5]};c.load_world=lambda:None
    with patch('cross_episode_sim.tasks.coffee.native_task.support_bounds',
               return_value=(np.array([.25,-.65,.89]),np.array([2.75,0.,.92]))):
        c.prepare_lid_placement()
    for target in [c.destination_pose,*c.placement_pose_options]:
        np.testing.assert_allclose(target[:3,:3],pose[:3,:3])
        vertices=local@target[:3,:3].T+target[:3,3]
        assert abs(vertices[:,2].min()-.92)<1e-9
        assert vertices[:,1].min()>-.65


@pytest.mark.parametrize('grains,brewed,object_name,upright', [
    ({1}, False, 'dose', True),
    (set(), False, 'dose', False),
    (set(), False, 'mug', False),
    (set(), True, 'mug', True),
    (set(), True, 'lid', False),
])
def test_only_loaded_coffee_vessels_use_orientation_preserving_carry(grains, brewed, object_name, upright):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c=CoffeeEpisode.__new__(CoffeeEpisode)
    c.object_name=object_name;c.brew=SimpleNamespace(completed=brewed)
    c.info=lambda role:{'body':'mug'};c.active_grains=lambda:grains
    c.tuck_vessel_for_navigation=Mock()
    with patch.object(BreakfastEpisode, 'tuck_loaded_for_navigation') as ordinary:
        c.tuck_loaded_for_navigation()
    assert c.tuck_vessel_for_navigation.called == upright
    assert ordinary.called != upright


@pytest.mark.parametrize('execution_error', [False, True])
def test_filled_carry_preserves_rotation_and_does_not_retry_execution_failure(execution_error):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c=BreakfastEpisode.__new__(BreakfastEpisode)
    pose=np.eye(4);pose[:3,3]=[.6,0.,1.]
    pose[:3,:3]=Rotation.from_euler('zy',[37,4],degrees=True).as_matrix()
    c.object_name='vessel';c.tcp=lambda:pose.copy();c.bread_pose=lambda:pose.copy()
    c.base_pose=lambda:np.zeros(3)
    c.profile=SimpleNamespace(namespace='robot/',arm_joints=['j1','j2'])
    c.data=SimpleNamespace(joint=lambda name:SimpleNamespace(qpos=[.2]))
    c.embodiment=SimpleNamespace(set_loaded_travel_posture=Mock())
    c.record=Mock()
    c.plan_contact_path=Mock(side_effect=[RuntimeError('blocked long retraction'), [[.2,.2]]])
    c.check_loaded_tuck_path=Mock()
    c.mesh_contact_move=Mock(side_effect=RuntimeError('lost grip') if execution_error else None)
    if execution_error:
        with pytest.raises(RuntimeError,match='lost grip'):
            c.tuck_vessel_for_navigation()
        c.embodiment.set_loaded_travel_posture.assert_not_called()
    else:
        c.tuck_vessel_for_navigation()
        c.embodiment.set_loaded_travel_posture.assert_called_once_with([.2,.2])
    assert c.plan_contact_path.call_count==2
    c.mesh_contact_move.assert_called_once()
    target=c.mesh_contact_move.call_args.args[1]
    np.testing.assert_allclose(target[:3,:3],pose[:3,:3])
    np.testing.assert_allclose(target[:3,3],[.42,0.,1.])


def test_dispenser_insertion_keeps_mug_below_spout_and_withdraws_horizontally():
    from unittest.mock import Mock
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode, COFFEE
    c=CoffeeEpisode.__new__(CoffeeEpisode);c.destination=COFFEE
    c.move=Mock();c.mesh_contact_move=Mock();c.record=Mock()
    candidate=np.eye(4);candidate[:3,3]=[2.4,-.4,1.04]
    staging=candidate.copy();staging[2,3]+=.12
    retreat=c.approach_table_placement(candidate,staging)
    target=c.mesh_contact_move.call_args.args[1]
    np.testing.assert_allclose(target[:3,3],[2.4,-.4,1.004])
    np.testing.assert_allclose(retreat[:3,3],[2.4,-.54,1.004])
    np.testing.assert_allclose(c.move.call_args.args[1],retreat)
    np.testing.assert_allclose(candidate[:3,3],[2.4,-.4,1.04])


def test_dispenser_targets_restore_upright_mug_with_handle_toward_aisle():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode, COFFEE
    c=CoffeeEpisode.__new__(CoffeeEpisode);c.object_name='mug';c.destination=COFFEE
    c.table_bids={COFFEE:set()};c.model=SimpleNamespace(ngeom=0)
    c.prepare_loaded_manipulation=Mock();c.record=Mock()
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('zy',[37,22],degrees=True).as_matrix()
    local=np.array([[x,y,z] for x in (-.04,.04) for y in (-.04,.04) for z in (-.05,.05)])
    c.bread_pose=lambda:pose.copy();c.bread_vertices=lambda:local@pose[:3,:3].T
    c.info=lambda role:dict(initial_quaternion=[1.,0.,0.,0.])
    c.local_annotations=np.eye(4)[None];c.local_annotations[0,0,3]=.08
    c.spec=dict(dispenser_target=[2.4,-.4,1.],tray_z=.95)
    c.prepare_destination()
    for target in [c.destination_pose,*c.placement_pose_options]:
        np.testing.assert_allclose(target[:2,3],[2.4,-.4])
        np.testing.assert_allclose(target[:3,2],[0.,0.,1.])
        vertices=local@target[:3,:3].T+target[:3,3]
        assert abs(vertices[:,2].min()-.952)<1e-9
    np.testing.assert_allclose(c.destination_pose[:3,0],[0.,-1.,0.],atol=1e-12)


@pytest.mark.parametrize('cached', [False, True])
def test_inventory_respects_vessel_rotation_and_disjoint_hopper_priority(cached):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    reference=Rotation.from_euler('y',90,degrees=True).as_matrix()
    current=Rotation.from_euler('x',70,degrees=True).as_matrix()
    cup=SimpleNamespace(xpos=np.array([2.,3.,4.]),xmat=current.ravel())
    positions=np.zeros((4,3))
    positions[3]=cup.xpos+current@reference.T@np.array([0.,0.,.02])
    positions[1]=[-1.,1.,.03];positions[2]=[5.,5.,0.]
    ids={'contained':3,'captured':1,'lost':2}
    model=SimpleNamespace(body=lambda name:SimpleNamespace(id=ids[name]))
    data=SimpleNamespace(xpos=positions,body=lambda name:cup)
    spec=dict(grains=list(ids),dosing_body='cup',
              dosing_cavity=dict(center_xy=[0.,0.],radius=.035,bottom_z=0.,rim_z=.05,
                                 reference_rotation=reference.tolist()),
              hopper=dict(center_xy=[-1.,1.],half_width=.054,bottom_z=0.,top_z=.10))
    cache=np.array(list(ids.values())) if cached else None
    assert grain_inventory(model,data,spec,cache)==dict(hopper=['captured'],vessel=['contained'],spilled=['lost'])
    p=positions[3]
    spec['hopper'].update(center_xy=p[:2],bottom_z=p[2]-.02,top_z=p[2]+.02)
    assert grain_inventory(model,data,spec,cache)==dict(hopper=['contained'],vessel=[],spilled=['captured','lost'])


def test_dosing_cup_requires_side_approach_and_docks_on_rotated_handle_side():
    from unittest.mock import Mock, patch
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
    from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
    c=CoffeeEpisode.__new__(CoffeeEpisode);c.object_name='dose';c.spec={'dosing_body':'dose'}
    with patch.object(CrossRoomManipulation,'annotation_approach_allowed',return_value=True):
        for angle,allowed in ((0,False),(30,False),(75,True),(120,False)):
            pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('x',180-angle,degrees=True).as_matrix()
            assert c.annotation_approach_allowed(pose)==allowed
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    c.bread_pose=lambda:pose
    c.local_annotations=np.repeat(np.eye(4)[None],2,axis=0);c.local_annotations[:,0,3]=.08
    c.docks_on_surface_side=Mock(return_value=iter([np.array([1.,2.,3.])]))
    docks=list(c.dock_candidates('living',[1.,1.,1.],pickup=True))
    np.testing.assert_allclose(c.docks_on_surface_side.call_args.args[2],[0.,1.])
    np.testing.assert_allclose(docks[0],[1.,2.,3.])


def test_reachable_initial_pose_with_unreachable_tilt_does_not_move_robot():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
    c=CoffeeEpisode.__new__(CoffeeEpisode)
    c.lid_closed=lambda:False;c.grasp_relative=np.eye(4)
    c.spec=dict(pour_yaws_deg=[0.],pour_angles_deg=[0.,15.,30.],
                dosing_cavity=dict(center_xy=[0.,0.],radius=.035,bottom_z=-.05,rim_z=.05,
                                   reference_rotation=np.eye(3).tolist()),
                hopper=dict(center_xy=[1.,2.],top_z=1.))
    c.profile=SimpleNamespace(namespace='robot/')
    c.data=SimpleNamespace(joint=lambda name:SimpleNamespace(qpos=[0.]))
    c.planner=SimpleNamespace(names=['j1'],plan=Mock(return_value=[[0.]]))
    c.embodiment=SimpleNamespace(planner_tool_offset=lambda:0.)
    c.nearby_ik=Mock(side_effect=RuntimeError('No nearby tilt IK'))
    c.move=Mock();c.preplanned_moves={}
    with pytest.raises(RuntimeError,match='No reachable pouring approach'):
        c.pour_grounds()
    c.planner.plan.assert_called_once()
    c.move.assert_not_called()
    assert c.preplanned_moves=={}


def test_return_from_pour_compensates_for_measured_grip_shift():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
    c=CoffeeEpisode.__new__(CoffeeEpisode)
    c.lid_closed=lambda:False;c.grasp_relative=np.eye(4)
    c.spec=dict(pour_yaws_deg=[0.],pour_angles_deg=[0.,15.],pour_hold_seconds=.35)
    c.profile=SimpleNamespace(namespace='robot/')
    c.data=SimpleNamespace(joint=lambda name:SimpleNamespace(qpos=[0.]))
    c.planner=SimpleNamespace(names=['j1'],plan=Mock(return_value=[[0.]]))
    c.embodiment=SimpleNamespace(planner_tool_offset=lambda:0.)
    c.nearby_ik=Mock(return_value=[0.]);c.check_loaded_tuck_path=Mock()
    c.preplanned_moves={};c.report={};c.record=Mock();c.move=Mock();c.tick=Mock()
    c.mesh_contact_move=Mock();c.grounds_ready=lambda:True
    c.inventory=lambda:dict(hopper=list(range(32)),vessel=[],spilled=[])
    shifted=np.eye(4);shifted[0,3]=.03
    c.tcp=lambda:np.eye(4);c.bread_pose=lambda:shifted.copy()
    with patch('cross_episode_sim.tasks.coffee.native_task.pour_pose',return_value=np.eye(4)):
        c.pour_grounds()
    c.check_loaded_tuck_path.assert_called_once()
    stage,target=c.mesh_contact_move.call_args.args
    assert stage=='return dosing cup upright'
    np.testing.assert_allclose(target[:3,3],[-.03,0.,0.])
    assert c.report['pour_evidence']['counts']==dict(hopper=32,vessel=0,spilled=0)
    assert c.poured and not c.pouring
