from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from cross_episode_sim.env.household import PhysicalCompositeEnvironment
from cross_episode_sim.generation.household_bindings import physical_floor


def test_compartment_alignment_clears_open_door_and_held_object():
    from cross_episode_sim.env.household_surfaces import compartment_front
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="cabinet"><geom name="floor" type="box" size=".3 .1 .01" pos="0 0 .8"/></body>
      <body name="door" pos=".3 -.4 .9"><joint name="hinge" axis="0 0 1"/>
        <geom type="box" size=".05 .15 .2"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    local=np.array([[x,y,z] for x in (-.05,.05) for y in (-.05,.05) for z in (-.03,.03)])
    grip=np.eye(4);grip[1,3]=-.02
    c=SimpleNamespace(model=model,data=data,
        _surface_specs={'cabinet':{'floor_geom':'floor'}},task_outward={'cabinet':[0.,-1.]},
        manifest={'fixture_bindings':{'cabinet':{'body':'door'}}},
        descendants=lambda name:{model.body(name).id},bread_pose=lambda:np.eye(4),
        bread_vertices=lambda:local,grasp_relative=grip)
    tcp=np.eye(4);tcp[2,3]=.85
    front=compartment_front(c,'cabinet',tcp)
    assert front[1,3]<-.6
    placed=front @ grip
    # Every point of the held object is in front of the door with 3 cm clearance.
    assert (local+placed[:3,3])[:,1].max()<=-.55-.03+1e-12
    np.testing.assert_array_equal(front[:3,:3],tcp[:3,:3])
    assert front[2,3]==tcp[2,3]
    c.manifest['fixture_bindings']['cabinet']['joint']='hinge'
    side=compartment_front(c,'cabinet',tcp)
    assert side[0,3]<-.3  # Approach around the side opposite the hinge.
    assert side[1,3]==-.4  # No need to retreat behind the base when laterally clear.
    data.qpos[model.joint('hinge').qposadr[0]]=np.pi/2
    mujoco.mj_forward(model,data)
    straight=compartment_front(c,'cabinet',tcp)
    assert straight[0,3]==tcp[0,3]  # Fully open door need not force an edge detour.
    assert straight[1,3]==pytest.approx(-.16)


def test_compartment_placement_prefers_containment_and_wrist_clearance(monkeypatch):
    from cross_episode_sim.env import household_surfaces
    from cross_episode_sim.env.household import CompositeController
    c=object.__new__(CompositeController)
    c.object_name='box';c.object_info={'box':{'destination':'microwave'}}
    c._surface_specs={'microwave':{}}
    c.destination='microwave';c.table_bids={'microwave':{1}}
    c.prepare_loaded_manipulation=lambda:None
    c.bread_pose=lambda:np.eye(4)
    local=np.array([[x,y,z] for x in (-.04,.04) for y in (-.06,.06) for z in (-.03,.03)])
    c.bread_vertices=lambda:local.copy()
    c.transfer_start=np.array([0.,0.,0.,1.,0.,0.,0.])
    c.grasp_relative=np.eye(4);c.grasp_relative[0,3]=.08
    c.task_outward={'microwave':np.array([0.,-1.])}
    lo=np.array([-.2,-.25,.79]);hi=np.array([.2,.25,.8])
    monkeypatch.setattr(household_surfaces,'surface',lambda *args:(2,lo,hi))
    monkeypatch.setattr(household_surfaces,'candidates',lambda *args:[
        np.array([0.,0.,.8]),np.array([0.,-.18,.8]),np.array([0.,-.24,.8])])
    c.prepare_destination()
    pose=c.destination_pose
    assert pose[1,3]==-.18
    tcp=pose @ np.linalg.inv(c.grasp_relative)
    assert (tcp[:3,3]-.30*tcp[:3,2])[1]<pose[1,3]-.079
    for p in [pose,*c.placement_pose_options]:
        vertices=local @ p[:3,:3].T+p[:3,3]
        assert np.all(vertices[:,:2]>=lo[:2]+.025)
        assert np.all(vertices[:,:2]<=hi[:2]-.025)
        assert p[2,2]>=np.cos(np.radians(40))-1e-12


def test_fixture_dock_faces_handle_arc_consistently_for_open_and_close():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="root">
      <body name="door"><joint name="hinge" axis="0 0 1"/>
        <geom name="handle" type="sphere" size=".01" pos="1 0 .5"/>
      </body></body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    goals=[]
    c=SimpleNamespace(model=model,data=data,tuck_for_navigation=lambda:None,
        plan_route=lambda xy,carrying,face: [xy.copy()],
        task_navigate=lambda xy,carrying,face:goals.append((xy.copy(),face)))
    env=object.__new__(PhysicalCompositeEnvironment)
    env.controller=c;env.held=lambda:None
    env.fixture_bindings={'cabinet':dict(handle_geom='handle',joint='hinge',
        support='root',dock_face_swing=True,open_position=np.pi/2)}
    for angle in (0.,np.pi/2):
        data.joint('hinge').qpos[0]=angle;mujoco.mj_forward(model,data)
        env.navigate_fixture('cabinet')
    np.testing.assert_allclose(goals[0][0],[1,-.5])
    np.testing.assert_allclose(goals[1][0],goals[0][0],atol=1e-12)
    expected=np.arctan2(np.sqrt(.5)+.5,np.sqrt(.5)-1)
    np.testing.assert_allclose([g[1] for g in goals],[expected,expected])


def test_handle_grasp_preflight_rejects_unreachable_later_stroke():
    import pytest

    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    requested=[]
    def ik(pose,q,**kwargs):
        angle=np.arctan2(pose[1,3],pose[0,3])
        requested.append(angle)
        if angle>np.radians(50):raise RuntimeError('wrist range exhausted')
        return np.asarray(q).copy()
    c=SimpleNamespace(angle=lambda:0.,door_joint=0,nearby_ik=ik,
        data=SimpleNamespace(xanchor=np.zeros((1,3)),xaxis=np.array([[0.,0.,1.]])))
    grasp=np.eye(4);grasp[0,3]=1.
    with pytest.raises(RuntimeError,match='wrist range exhausted'):
        CabinetDoorTest.preflight_hinge_ik(c,grasp,np.zeros(7),np.pi/2)
    assert max(requested)>np.radians(50)
    np.testing.assert_array_equal(grasp[:3,3],[1,0,0])


def test_placement_footprint_uses_upright_pose_not_tilted_carry():
    from cross_episode_sim.env.household_surfaces import candidates
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="basket"><geom name="floor" type="box" size=".12 .10 .008" pos="0 0 .8"/></body>
      <body name="bottle" pos="0 0 1" quat=".70710678 0 .70710678 0">
        <freejoint/><geom type="box" size=".03 .03 .08"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    c=SimpleNamespace(model=model,data=data,manifest={'bindings':[]},
        _surface_specs={'blue_basket':{'floor_geom':'floor','body':'basket'}},
        object_info={'bottle':{'initial_quaternion':[1,0,0,0]}},
        descendants=lambda name:{model.body(name).id})
    positions=np.asarray(candidates(c,'blue_basket','bottle'))
    np.testing.assert_allclose([positions[:,0].min(),positions[:,0].max()],[-.078,.078],atol=1e-6)
    assert positions[0,0]==positions[:,0].min()


def test_compartment_release_gap_does_not_change_tabletop_clearance(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    monkeypatch.setattr(BreakfastEpisode,'prepare_destination',lambda self:None)
    c=object.__new__(CompositeController)
    c.object_name='box';c.object_info={'box':{'destination':'microwave'}}
    c._surface_specs={};c.placement_release_clearance=.03
    c.lower_vessel_release_fallback=True;c.preserve_placement_candidate_order=False
    c.prepare_destination()
    assert c.placement_release_clearance==.005
    assert not c.lower_vessel_release_fallback
    assert c.preserve_placement_candidate_order
    c.object_info['box']['destination']='dining'
    c.prepare_destination()
    assert c.placement_release_clearance==.03
    assert c.lower_vessel_release_fallback
    assert not c.preserve_placement_candidate_order


def test_compartment_search_deadline_stops_before_more_planning():
    import pytest

    from cross_episode_sim.env.household import CompositeController
    c=SimpleNamespace(object_name='box',object_info={'box':{'destination':'microwave'}},
                      _compartment_plan_deadline=0.)
    with pytest.raises(RuntimeError,match='search budget exhausted') as caught:
        CompositeController.approach_table_placement(c,np.eye(4),np.eye(4))
    assert not CompositeController.retryable_placement(caught.value)


def test_compartment_withdrawal_uses_empty_hand_native_mesh_path(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    seen=[]
    monkeypatch.setattr('cross_episode_sim.control.compartment_entry.plan_empty_withdrawal',
                        lambda c, points: [(p, []) for p in points])
    monkeypatch.setattr(BreakfastEpisode,'move',lambda *args:seen.append('ordinary'))
    c=object.__new__(CompositeController)
    c.object_name='box';c.object_info={'box':{'destination':'microwave'}}
    c.mesh_contact_move=lambda *args, **kwargs:seen.append('native_mesh')
    c.move('withdraw from released table object',np.eye(4))
    c.object_info['box']['destination']='dining'
    c.move('withdraw from released table object',np.eye(4))
    assert seen==['native_mesh','ordinary']


def test_cabinet_entry_and_withdrawal_follow_the_same_aperture_corridor(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    front=np.eye(4);front[:3,3]=[-.2,-.5,.7]
    mouth=np.eye(4);mouth[:3,3]=[0.,-.2,.7]
    destination=np.eye(4);destination[:3,3]=[0.,0.,.7]
    monkeypatch.setattr('cross_episode_sim.env.household_surfaces.compartment_front',lambda *a:front)
    monkeypatch.setattr('cross_episode_sim.env.household_surfaces.compartment_mouth',lambda *a:mouth)
    events=[]
    def preflight(c,alignment,target,via=()):
        np.testing.assert_array_equal(via[0],mouth)
        events.append('preflight')
        return [np.zeros(7)]
    monkeypatch.setattr('cross_episode_sim.control.compartment_entry.plan_entry',preflight)
    monkeypatch.setattr('cross_episode_sim.control.compartment_entry.plan_empty_withdrawal',
                        lambda c, points: [(p, []) for p in points])
    c=SimpleNamespace(object_name='bowl',object_info={'bowl':{'destination':'kitchen_cabinet'}},
        preplanned_moves={},move=lambda *a:events.append('align'),
        mesh_contact_move=lambda stage,pose,**kwargs:events.append(pose.copy()))
    CompositeController.approach_table_placement(c,destination,destination)
    assert events[:2]==['preflight','align']
    np.testing.assert_array_equal(events[2],mouth)
    np.testing.assert_array_equal(events[3],destination)
    events.clear()
    CompositeController.move(c,'withdraw from released table object',front)
    np.testing.assert_array_equal(events[0],mouth)
    np.testing.assert_array_equal(events[1],front)


def test_basket_docking_uses_measured_parent_table(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    seen=[]
    def parent_docks(self,room,point,pickup=False):
        seen.append(room)
        yield np.array([1.,2.,3.])
    monkeypatch.setattr(BreakfastEpisode,'dock_candidates',parent_docks)
    c=object.__new__(CompositeController)
    c._surface_specs={'basket':{'parent_room':'dining'}}
    result=list(c.dock_candidates('basket',np.zeros(3)))
    assert seen==['dining']
    np.testing.assert_allclose(result[0],[1,2,3])


def test_basket_lift_settings_do_not_leak_after_failure(monkeypatch):
    import pytest

    from cross_episode_sim.skills.household_transfer import HouseholdTransferSkills
    c=SimpleNamespace(object_name='basket_body',args=SimpleNamespace(lift_height=.12),
                      configure_pickup_approach=lambda:None,
                      task_supports={'kitchen':'counter'},initial_lift_height=.03)
    env=SimpleNamespace(controller=c,bind_contained_payload=lambda target:None,roles={'basket':dict(body='basket_body',
                         source='kitchen',category='basket')})
    def failed_pick(controller):
        assert controller.args.lift_height==.08
        raise RuntimeError('grasp rejected')
    monkeypatch.setattr('cross_episode_sim.skills.household_transfer.pick_vessel',failed_pick)
    with pytest.raises(RuntimeError,match='grasp rejected'):
        HouseholdTransferSkills(env).pick('basket')
    assert c.args.lift_height==.12
    assert c.initial_lift_height==.03
    assert not hasattr(c,'minimum_pickup_lift_m')


@pytest.mark.parametrize('fallback', [False, True])
def test_compartment_preflight_restores_live_data_on_insertion_failure(fallback):
    import pytest

    from cross_episode_sim.control.compartment_entry import plan_entry
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="arm"><joint name="robot_0/joint" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".01"/></body>
      <body name="object" pos="0 0 1"><freejoint name="object_joint"/>
        <geom type="sphere" size=".01"/></body>
      </worldbody></mujoco>''')
    live=mujoco.MjData(model);mujoco.mj_forward(model,live)
    initial=live.qpos.copy()
    c=SimpleNamespace(model=model,data=live,object_joint='object_joint',object_name='object',
        planner=SimpleNamespace(names=['joint'],plan=lambda *args:[[.1]]),
        embodiment=SimpleNamespace(planner_tool_offset=lambda:0.),
        navigation_penetration=lambda *args:0.,robot_self_penetration=lambda *args:0.)
    def tcp():
        pose=np.eye(4);pose[0,3]=c.data.qpos[0];return pose
    def bread():
        pose=np.eye(4);pose[:3,3]=c.data.body('object').xpos;return pose
    def reject(*args):raise RuntimeError('insertion blocked')
    c.tcp=tcp;c.bread_pose=bread;c.plan_contact_path=reject
    radii=[]
    if fallback:
        def blocked(*args):raise RuntimeError('conservative box blocks cavity')
        def local_ik(pose, positions, trust_radius, **kwargs):
            radii.append(trust_radius)
            if trust_radius == 2.:raise RuntimeError('No nearby placement IK')
            return [.1]
        c.planner.plan=blocked
        c.nearby_ik=local_ik
        c.record=lambda **kwargs: None
    destination=np.eye(4);destination[1,3]=.1
    with pytest.raises(RuntimeError,match='insertion blocked'):
        plan_entry(c,np.eye(4),destination)
    assert c.data is live
    np.testing.assert_array_equal(live.qpos,initial)
    assert live.time==0
    assert radii == ([2.,6.] if fallback else [])


def test_compartment_rejection_does_not_execute_alignment(monkeypatch):
    import pytest

    from cross_episode_sim.env.household import CompositeController
    monkeypatch.setattr('cross_episode_sim.env.household_surfaces.surface',
        lambda *args: (0,np.array([0.,0.,.7]),np.array([.4,.4,.72])))
    movements=[]
    def reject(controller, front, destination):
        assert not np.allclose(front,destination)
        raise RuntimeError('Actual-mesh collision in contact move')
    monkeypatch.setattr('cross_episode_sim.control.compartment_entry.plan_entry',reject)
    controller=SimpleNamespace(object_name='box',manifest={},
        object_info={'box':{'destination':'microwave'}},
        task_outward={'microwave':[0.,-1.]},tcp=lambda:np.eye(4),
        record=lambda **kwargs:None,
        _checked_contact_targets=reject,
        mesh_contact_move=lambda *a,**kw:movements.append(a))
    candidate=np.eye(4);candidate[:3,3]=[.2,.2,.8]
    with pytest.raises(RuntimeError,match='Actual-mesh collision'):
        CompositeController.approach_table_placement(controller,candidate,candidate)
    assert movements==[]


def test_microwave_relocation_replaces_existing_euler_orientation(tmp_path):
    import pytest
    from cross_episode_sim.generation.appliance_setup import place_countertop_microwave
    scene=tmp_path/'scene.xml'
    scene.write_text('''<mujoco><worldbody>
      <body name="microwave_main_group_main" pos="3 3 1.5" euler="0 0 90">
        <geom type="box" size=".2 .2 .15"/>
      </body></worldbody></mujoco>''')
    with pytest.raises(ValueError, match='cart fallback is disabled'):
        place_countertop_microwave(scene)
    result=place_countertop_microwave(scene, allow_cart=True)
    model=mujoco.MjModel.from_xml_path(str(scene))
    assert result['support']=='microwave_cart'
    np.testing.assert_allclose(model.body('microwave_main_group_main').quat,[1,0,0,0])


def test_storage_binding_reserves_accessible_width_for_all_bowls():
    import pytest
    from cross_episode_sim.generation.household_bindings import fixture_bindings
    bodies=[]
    for number,half_width in [(3,.145),(4,.35)]:
        prefix=f'stack_{number}_main_group_1'
        bodies.append(f'''<body name="{prefix}_hingedoor">
          <joint name="{prefix}_doorhinge" axis="0 0 1" range="0 90"/>
          <geom type="box" size=".1 .02 .2" mass="1"/>
          <body name="{prefix}_door_handle_main"><geom name="{prefix}_door_handle_g1"
            type="box" size=".01 .01 .04"/></body></body>
          <body name="{prefix}_level2_main" pos="0 0 .6">
          <geom name="{prefix}_level2_shelf" type="box" size="{half_width} .265 .015"/></body>''')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+''.join(bodies)+'</worldbody></mujoco>')
    spec={'fixtures':['kitchen_cabinet']}
    assert fixture_bindings(model,spec)['kitchen_cabinet']['body'].startswith('stack_4_')
    chosen=fixture_bindings(model,spec,{'kitchen_cabinet':.60})['kitchen_cabinet']
    assert chosen['body'].startswith('stack_4_')
    assert chosen['door_stroke_degrees']==15.
    assert chosen['native_contact_refinement'] is True
    with pytest.raises(ValueError,match='No reachable native fixture'):
        fixture_bindings(model,spec,{'kitchen_cabinet':1.0})


def test_burner_uses_top_surface_not_internal_slab():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="stove">
      <geom name="internal" type="box" size=".3 .3 .01" pos="0 0 .94"/>
      <geom name="top" type="box" size=".3 .3 .01" pos="0 0 1.04"/>
      <site name="burner" pos="0 0 1.05"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    gid,_,high=physical_floor(model,data,dict(support='stove',burner_site='burner'))
    assert model.geom(gid).name=='top'
    assert np.isclose(high[2],1.05)


def test_initial_state_serializes_actual_spec_model_not_last_xml(tmp_path):
    spec=mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="robot"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    model=spec.compile();data=mujoco.MjData(model)
    data.qpos[0]=.7;mujoco.mj_forward(model,data)
    mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    env=object.__new__(PhysicalCompositeEnvironment)
    env.controller=SimpleNamespace(model=model,data=data)
    env.output=tmp_path;env.snapshot=lambda:dict(held=None)
    env.save_initial_state(tmp_path/'initial_state.npz')
    restored=mujoco.MjModel.from_binary_path(str(tmp_path/'model.mjb'))
    assert restored.body('robot').id>0
    restored_data=mujoco.MjData(restored)
    with np.load(tmp_path/'initial_state.npz') as saved:
        mujoco.mj_setState(restored,restored_data,saved['state'],int(saved['state_type']))
    assert restored_data.qpos[0]==.7


def test_fridge_shelf_in_static_child_excludes_moving_drawer():
    model=mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="fridge">
      <body name="glass"><geom name="shelf" type="box" size=".2 .2 .005" pos="0 0 .825"/></body>
      <body name="drawer"><joint type="slide" axis="0 1 0"/>
        <geom name="drawer_bottom" type="box" size=".2 .2 .005" pos="0 0 .83" mass="1"/>
      </body></body></worldbody></mujoco>""")
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    gid,_,high=physical_floor(model,data,dict(support='fridge',shelf_height=.835))
    assert model.geom(gid).name=='shelf'
    assert np.isclose(high[2],.83)


def test_fridge_place_requires_selected_shelf_not_any_fridge_support():
    from types import SimpleNamespace
    from cross_episode_sim.env.household_surfaces import supported_by_floor
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="low" type="box" size=".3 .3 .01" pos="0 0 .69"/>
      <geom name="shelf" type="box" size=".3 .3 .01" pos="0 0 1.08"/>
      <body name="drink" pos="0 0 .7499"><freejoint/><geom type="box" size=".02 .02 .05"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    c=SimpleNamespace(model=model,data=data);bodies={model.body('drink').id}
    assert supported_by_floor(c,bodies,model.geom('low').id)
    assert not supported_by_floor(c,bodies,model.geom('shelf').id)
    data.qpos[2]=1.1399;mujoco.mj_forward(model,data)
    assert supported_by_floor(c,bodies,model.geom('shelf').id)
    assert not supported_by_floor(c,bodies,model.geom('low').id)


def test_microwave_placement_binds_turntable_not_slab_under_side_walls():
    from cross_episode_sim.generation.household_bindings import fixture_bindings
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <body name="microwave_main_group_main">
        <geom name="bottom_slab" type="box" size=".4 .2 .01" pos="0 0 .94"/>
        <geom name="microwave_main_group_tray" type="cylinder" size=".16 .003" pos="0 0 .97"/>
        <body name="microwave_main_group_door">
          <joint name="microwave_main_group_microjoint" axis="0 0 1" limited="true" range="-1.57 0"/>
          <geom name="microwave_main_group_door_handle_main" type="box" size=".01 .01 .1" pos="0 -.2 1"/>
        </body>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    binding=fixture_bindings(model,{'fixtures':['microwave']})['microwave']
    gid,lo,hi=physical_floor(model,data,binding)
    assert model.geom(gid).name=='microwave_main_group_tray'
    np.testing.assert_allclose(hi[:2]-lo[:2],[.32,.32])
    assert hi[2]==pytest.approx(.973)


def test_carry_back_to_table_restores_allocated_slot_after_appliance():
    from cross_episode_sim.skills.household_transfer import HouseholdTransferSkills
    info = dict(body='box', source='kitchen', destination='microwave',
                position=[1., 2., .9], destination_position=[2., 3., 1.])
    seen = []
    c = SimpleNamespace(_surface_specs={}, task_supports={'dining': 'table'},
        _public_navigation_goal=([0., 0.], 0.),
        transport_payload=lambda: seen.append(list(info['destination_position'])))
    env = SimpleNamespace(controller=c, roles={'food': info}, held=lambda:'food',
        supports={'dining_table':'dining'}, dining_positions={'food':[-1., -2., .8]})
    HouseholdTransferSkills(env).navigate('food', destination='dining_table')
    assert seen == [[-1., -2., .8]]
    assert info['destination'] == 'dining'
    info['destination_position'][0] = 99.
    assert env.dining_positions['food'] == [-1., -2., .8]


def test_cabinet_docks_include_opposite_approach_side(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    monkeypatch.setattr('cross_episode_sim.env.household_surfaces.surface',
        lambda c,role:(0,np.array([0.,0.,.5]),np.array([.6,.4,.6])))
    c=SimpleNamespace(_surface_specs={'kitchen_cabinet':{}},
                      task_outward={'kitchen_cabinet':[0.,-1.]})
    docks=list(CompositeController.dock_candidates(c,'kitchen_cabinet',np.zeros(3)))
    assert any(d[0]<.3 for d in docks)
    assert any(d[0]>.6 for d in docks)
    assert all(d[1]<0. for d in docks)


def test_burner_approach_tries_lower_waypoint_before_executing(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.control import compartment_entry
    attempts, moves = [], []
    target = np.eye(4)
    target[2,3] = 1.
    def plan(c, front, destination):
        attempts.append(float(front[2,3]))
        assert not moves
        if front[2,3] > 1.05:
            raise RuntimeError('overhead waypoint unreachable')
        return np.zeros((1,7))
    monkeypatch.setattr(compartment_entry,'plan_entry',plan)
    c=SimpleNamespace(object_name='pot',object_info={'pot':{'destination':'burner_1','category':'pot'}},
                      _surface_specs={'burner_1':{'burner_site':'site'}},preplanned_moves={},
                      record=lambda **kwargs:None,
                      move=lambda *args:moves.append(args),mesh_contact_move=lambda *args:moves.append(args))
    front=CompositeController.approach_table_placement(c,target,None)
    np.testing.assert_allclose(attempts,[1.08,1.04])
    assert len(moves)==2
    assert front[2,3]==1.04

@pytest.mark.parametrize('collision_kind', ['environment', 'self'])
def test_basket_ik_fallback_still_rejects_native_collisions(collision_kind):
    from cross_episode_sim.control.compartment_entry import plan_entry
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="robot_0/joint" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".01"/></body>
      <body name="basket" pos="0 0 1"><freejoint name="basket_joint"/>
        <geom type="sphere" size=".01"/></body>
      </worldbody></mujoco>''')
    live=mujoco.MjData(model);mujoco.mj_forward(model,live)
    initial=live.qpos.copy()
    def blocked(*args):raise RuntimeError('conservative attached geometry')
    checks=[]
    def ik(pose,positions,trust_radius,preserve_self_clearance):
        checks.append(preserve_self_clearance)
        if preserve_self_clearance:raise RuntimeError('conservative self overlap')
        return [.1]
    c=SimpleNamespace(model=model,data=live,object_joint='basket_joint',object_name='basket',
        object_info={'basket':{'category':'basket'}},
        planner=SimpleNamespace(names=['joint'],plan=blocked),nearby_ik=ik,
        embodiment=SimpleNamespace(planner_tool_offset=lambda:0.),record=lambda **kw:None,
        navigation_penetration=lambda *args:.01 if collision_kind=='environment' else 0.,
        robot_self_penetration=lambda *args:.01 if collision_kind=='self' else 0.,
        contact_pair_detail=lambda *args:['native collision'])
    def tcp():
        pose=np.eye(4);pose[0,3]=c.data.qpos[0];return pose
    def basket():
        pose=np.eye(4);pose[:3,3]=c.data.body('basket').xpos;return pose
    c.tcp=tcp;c.bread_pose=basket
    with pytest.raises(RuntimeError,match='Actual-mesh collision'):
        plan_entry(c,np.eye(4),np.eye(4))
    assert checks==[True,True,False]
    assert c.data is live
    np.testing.assert_array_equal(live.qpos,initial)
    assert live.time==0


def test_meal_counter_search_preserves_preferred_sites_and_compact_fallback():
    from cross_episode_sim.generation.household_scene import meal_counter_fractions
    fractions=np.linspace(.05,.95,61)
    result=meal_counter_fractions(fractions,.25,2.75,2.25)
    preferred=[f for f in fractions if .9 <= .25+2.5*f <= 2.1]
    preferred.sort(key=lambda f:-abs(.25+2.5*f-2.25))
    np.testing.assert_allclose(result[:len(preferred)],preferred)
    assert any(.7 <= .25+2.5*f < .9 for f in result)
    assert all(.7 <= .25+2.5*f <= 2.3 for f in result)


def test_extra_room_navigation_uses_shared_controller():
    from types import SimpleNamespace
    events=[]
    c=SimpleNamespace(tuck_for_navigation=lambda:events.append('tuck'),
                      task_navigate=lambda xy,carrying,face:events.append((xy.tolist(),carrying,face)))
    env=object.__new__(PhysicalCompositeEnvironment);env.controller=c
    env.room_targets={'extra_1':[10.,-3.]};env.held=lambda:None
    env.navigate_fixture('extra_1')
    assert events==['tuck',([10.,-3.],False,0.)]
