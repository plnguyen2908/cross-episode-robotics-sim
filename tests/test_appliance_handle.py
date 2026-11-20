"""A handle is accepted for the microwave only when it sits on the pointed microwave body."""
import inspect
import numpy as np
from cross_episode_sim.perception.fixture_pointing import on_appliance_body, appliance_body_points


def test_microwave_handle_on_the_body_is_kept_and_cabinet_handle_below_is_dropped():
    body=[np.array([.82,-1.28,1.10])]          # fix102 saved view, pointed microwave body
    assert on_appliance_body([.95,-1.20,1.14],body)   # the microwave's own door handle
    assert not on_appliance_body([.45,-1.16,.70],body) # cabinet-door handle under the counter


def test_a_handle_far_along_the_counter_is_not_on_the_body():
    assert not on_appliance_body([1.9,-1.28,1.12],[np.array([.82,-1.28,1.10])])


def test_no_pointed_body_accepts_nothing():
    assert not on_appliance_body([.9,-1.2,1.1],[])


def test_body_points_skip_pixels_without_depth():
    class Camera:
        rgb=np.zeros((4,4,3),np.uint8);depth_m=np.ones((4,4));K=np.eye(3);T_base_optical=np.eye(4)
    class Pointer:
        def point(self,image,target):return dict(points=[(1,1),(2,2)])
    valid=np.zeros((4,4),bool);valid[2,2]=True
    assert len(appliance_body_points(Pointer(),Camera(),valid,'microwave'))==1


def test_remembered_appliance_doors_are_pushed_shut_like_cabinets():
    from cross_episode_sim.code_policy import fixture_profiles
    assert fixture_profiles.PUSH_CLOSED_APPLIANCES==('microwave','fridge')
    source=inspect.getsource(fixture_profiles.HouseholdFixtureControls._execute)
    assert "target in PUSH_CLOSED_APPLIANCES" in source
    assert "PUSH_CLOSE_FAILED_TRY_HANDLE" in inspect.getsource(fixture_profiles.HouseholdFixtureControls._closing_stroke)


def test_handle_redock_faces_the_microwave_when_no_handle_on_it_is_seen():
    from cross_episode_sim.runtime import qwen_pipeline
    assert 'microwave' in qwen_pipeline.FACE_APPLIANCE_FIRST
    source=inspect.getsource(qwen_pipeline.PipelineRedocking.__call__)
    assert "session.redock(face_fixture,'approach',FACE_APPLIANCE_DISTANCE)" in source


def _panel_with_bar(vertical):
    """A flat panel 0.7 m ahead (camera at the origin) with one protruding bar."""
    rng=np.random.default_rng(3)
    panel=np.c_[np.full(4000,.70),rng.uniform(-.2,.2,4000),rng.uniform(.8,1.2,4000)]
    long=rng.uniform(-.15,.15,1500);thin=rng.uniform(-.01,.01,1500)
    bar=np.c_[np.full(1500,.67),thin,1.+long] if vertical else np.c_[np.full(1500,.67),long,1.+thin]
    return np.r_[panel,bar],np.array([.67,0.,1.])


def test_fridge_door_handle_must_be_a_vertical_bar():
    from cross_episode_sim.perception.handle_surface import vertical_bar_handle
    cloud,target=_panel_with_bar(vertical=True)
    assert vertical_bar_handle(cloud,target)
    cloud,target=_panel_with_bar(vertical=False)   # the freezer pull under the door
    assert not vertical_bar_handle(cloud,target)


def test_no_handle_surface_is_not_a_vertical_bar():
    from cross_episode_sim.perception.handle_surface import vertical_bar_handle
    assert not vertical_bar_handle(np.zeros((10,3)),np.array([.7,0.,1.]))


def test_grasp_point_slides_down_a_long_vertical_bar_across_a_hidden_stretch():
    from cross_episode_sim.perception.handle_surface import slide_along_vertical_bar
    rng=np.random.default_rng(5)
    panel=np.c_[np.full(20000,.70),rng.uniform(-.2,.2,20000),rng.uniform(.5,1.7,20000)]
    z=rng.uniform(.68,1.58,6000);z=z[(z<1.17)|(z>1.28)]            # 11 cm hidden by the arm
    bar=np.c_[np.full(len(z),.66),rng.uniform(-.01,.01,len(z)),z]
    moved=slide_along_vertical_bar(np.r_[panel,bar],np.array([.66,0.,1.40]),1.05)
    assert abs(moved[2]-1.05)<.02 and abs(moved[1])<.01


def test_grasp_point_stays_when_there_is_no_bar():
    from cross_episode_sim.perception.handle_surface import slide_along_vertical_bar
    target=np.array([.66,0.,1.40])
    assert np.allclose(slide_along_vertical_bar(np.zeros((10,3)),target,1.05),target)


def test_microwave_redock_distance_follows_the_measured_floor_centre():
    from cross_episode_sim.runtime.household_components import centre_redock_distance
    assert centre_redock_distance(.78) is None                      # already inside reach
    assert abs(centre_redock_distance(.857)-.513)<1e-3             # fix104: 8.7 cm closer
    assert centre_redock_distance(1.2)==.45                         # never closer than the minimum
    assert centre_redock_distance(None) is None


def test_navigation_hands_objects_inside_an_appliance_to_their_own_pick_dock():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    script=(root/'scripts/demo_semantic_navigation.py').read_text()
    assert "confined_target=a.target.endswith((' inside fridge',' inside microwave'))" in script
    assert "'confined_target_found'" in script
    navigator=(root/'cross_episode_sim/navigation/hybrid_navigation.py').read_text()
    assert "self.terminal='confined_target_found'" in navigator


def test_pick_proposals_take_the_shelf_under_the_object_not_the_dense_shelf_above():
    from scripts.target_pick_proposals import target_context
    rng=np.random.default_rng(7)
    shelf=np.c_[rng.uniform(.4,.8,3000),rng.uniform(-.2,.2,3000),np.full(3000,1.07)]
    above=np.c_[rng.uniform(.4,.8,9000),rng.uniform(-.2,.2,9000),np.full(9000,1.665)]
    angle=rng.uniform(0,2*np.pi,2000)
    bottle=np.c_[.6+.03*np.cos(angle),.03*np.sin(angle),rng.uniform(1.075,1.24,2000)]
    _,mask,support=target_context(np.r_[shelf,above,bottle],np.array([.6,0.,1.13]))
    assert abs(support-1.07)<.006 and mask.sum()>1500


def test_pick_proposals_keep_the_densest_level_on_an_open_surface():
    from scripts.target_pick_proposals import target_context
    rng=np.random.default_rng(8)
    table=np.c_[rng.uniform(.4,.8,6000),rng.uniform(-.2,.2,6000),np.full(6000,.75)]
    angle=rng.uniform(0,2*np.pi,1500)
    cup=np.c_[.6+.04*np.cos(angle),.04*np.sin(angle),rng.uniform(.755,.85,1500)]
    _,mask,support=target_context(np.r_[table,cup],np.array([.6,0.,.80]))
    assert abs(support-.75)<.006 and mask.sum()>1000


def test_level_entry_pick_offers_mid_height_when_all_proposals_sit_at_the_foot():
    from cross_episode_sim.control.manipulation_approach import front_pick_candidates
    surface=np.c_[np.full(50,.62),np.zeros(50),np.linspace(1.075,1.235,50)]
    low=np.eye(4);low[:3,3]=[.62,0.,1.09]
    rows=front_pick_candidates([dict(tcp=low.tolist(),group=0)],np.array([-1.,0.,0.]),surface,1.07)
    heights=[np.asarray(r['tcp'])[2,3] for r in rows]
    assert abs(heights[0]-1.155)<1e-6 and rows[0]['proposal_source'].endswith('mid_height')


def test_level_entry_pick_is_unchanged_when_a_proposal_is_already_at_mid_height():
    from cross_episode_sim.control.manipulation_approach import front_pick_candidates
    surface=np.c_[np.full(50,.62),np.zeros(50),np.linspace(1.075,1.235,50)]
    mid=np.eye(4);mid[:3,3]=[.62,0.,1.15]
    rows=front_pick_candidates([dict(tcp=mid.tolist(),group=0)],np.array([-1.,0.,0.]),surface,1.07)
    assert len(rows)==2 and all(not r['proposal_source'].endswith('mid_height') for r in rows)


def test_mid_stroke_docks_stand_off_the_leaf_turned_toward_closed():
    from cross_episode_sim.code_policy.closing_redock import ClosingRedocking, closing_face
    # Fridge of the meal scene (world frame): hinge on the right, door open 90 deg toward -y.
    memory=dict(hinge=np.array([4.975,-.783,1.09]),closed_handle=np.array([4.29,-.83,1.09]),
                open_handle=np.array([5.02,-1.47,1.09]),closed_normal=np.array([0.,-1.,0.]))
    dock=ClosingRedocking.__new__(ClosingRedocking)
    dock.expected,dock.approach_normal_map=closing_face(memory);dock.memory=memory
    assert dock.approach_normal_map[0]>.95                       # fully open: the outer face looks at the wall (+x)
    poses=np.asarray(dock.mid_stroke_docks((.6,)))
    assert len(poses)==12
    assert poses[0,0]>5.54-.31                                   # the oracle's own spot is inside the docking margin here
    assert np.sum(poses[:,0]<5.54-.31)>=4                        # the further turned spots clear the wall at x=5.54
    assert np.all(poses[:,1]<-1.4)                               # all out in the room, level with or past the leaf's free edge


def test_placement_outcome_fields_are_read_from_dicts_and_objects():
    from cross_episode_sim.runtime.household_components import outcome_field
    from cross_episode_sim.skills.m2t2_interface import SkillOutcome
    failed=dict(success=False,reason='stage_failed:test_aperture_place: placement_plan_unreachable')
    assert outcome_field(failed,'success',False) is False and 'unreachable' in outcome_field(failed,'reason','')
    assert outcome_field(SkillOutcome(True,'ok'),'success') is True
    assert outcome_field({}, 'reason','')==''


def test_partial_close_turns_the_remembered_handle_toward_closed():
    from cross_episode_sim.code_policy.panel_closing import advance_open_memory
    memory=dict(hinge=np.array([0.,0.,1.]),closed_handle=np.array([-.7,0.,1.]),open_handle=np.array([0.,-.7,1.]))
    remaining=advance_open_memory(memory,45.)
    assert abs(remaining-45.)<1e-6
    assert np.allclose(memory['open_handle'],[-.7*np.sin(np.pi/4),-.7*np.cos(np.pi/4),1.],atol=1e-6)
    assert advance_open_memory(memory,80.)<1e-6                       # never past closed
    assert np.allclose(memory['open_handle'],memory['closed_handle'],atol=1e-6)


def test_close_door_is_pushed_in_strokes_before_the_handle_fallback():
    from cross_episode_sim.code_policy import fixture_profiles
    source=inspect.getsource(fixture_profiles.HouseholdFixtureControls._execute)
    assert 'for stroke in range(CLOSE_STROKES)' in source and 'PartialPanelClose' in source
    assert 'self.index+=1' in inspect.getsource(fixture_profiles.HouseholdFixtureControls._closing_stroke)


def test_depth_returns_on_the_robot_itself_are_not_kept_as_the_open_door():
    from types import SimpleNamespace
    from cross_episode_sim.navigation.aperture_dock import outside_robot
    arm=SimpleNamespace(center=np.array([.08,0.,1.0]),axes=np.eye(3),half=np.array([.06,.06,.2]))
    points=np.array([[.117,.08,.977],[.10,.45,1.0],[.60,.26,1.1]])   # elbow return, door leaf, hinge side
    kept=outside_robot(points,[arm])
    assert len(kept)==2 and not np.any(np.all(np.isclose(kept,[.117,.08,.977]),axis=1))


def test_closer_microwave_dock_shifts_away_from_the_open_leaf():
    from cross_episode_sim.runtime.household_components import away_from_open_leaf
    # fix111, base frame: door plane ahead facing the robot (normal -x), hinge on the left, leaf swung out on the left.
    aperture=dict(hinge=[.628,.274,1.1],closed_handle=[.565,-.281,1.1],closed_normal=[-1.,-.023,0.],open_handle=[.074,.337,1.1])
    side=away_from_open_leaf(aperture)
    normal=np.array([-1.,-.023]);tangent=np.array([-normal[1],normal[0]])
    assert (side*tangent)[1]<0                                  # the shift goes to the right, away from the leaf on the left
    assert away_from_open_leaf(None)==1.


def test_held_panel_is_found_above_the_hand_when_hidden_at_hand_height():
    from cross_episode_sim.control.sensor_hinged import fit_held_panel,PanelObservation
    rng=np.random.default_rng(11)
    # A tall door plane at x=0.70 (normal toward the robot, -x); the band around the hand (z 0.9-1.1) is hidden.
    y=rng.uniform(-.3,.3,40000);z=rng.uniform(.6,1.6,40000);keep=(z<.88)|(z>1.12)
    door=np.c_[np.full(keep.sum(),.70),y[keep],z[keep]]
    reference=PanelObservation(np.array([.70,0.,1.05]),np.array([-1.,0.,0.]),1000,0.)
    tcp=np.array([.655,0.,1.05])
    offset,panel=fit_held_panel(door,tcp,np.array([0.,1.,0.]),reference,tcp,np.array([-1.,0.,0.]))
    assert abs(panel.anchor[0]-.70)<.01 and panel.normal[0]<-.98


def test_fridge_close_gets_one_far_side_push_when_the_closing_push_cannot_start():
    from cross_episode_sim.code_policy import fixture_profiles
    from cross_episode_sim.runtime import fixture_controls
    assert fixture_profiles.FAR_SIDE_PUSH_DOORS==('fridge',)
    source=inspect.getsource(fixture_profiles.HouseholdFixtureControls._execute)
    assert 'turned,_=self.door_hook_push(target,root)' in source and "raise PartialPanelClose(turned,'far-side push')" in source
    assert fixture_controls.DOOR_HOOK_CONTACT_AFTER<fixture_controls.DOOR_HOOK_TRAVEL
    # 0.44 m of hand travel at a 0.67 m handle radius was measured to turn the leaf 24.7 deg.
    turned=np.degrees(np.arcsin((.44-fixture_controls.DOOR_HOOK_CONTACT_AFTER)/.67))
    assert abs(turned-24.7)<1.


def test_remembered_handle_can_also_be_turned_further_open():
    from cross_episode_sim.code_policy.panel_closing import advance_open_memory
    memory=dict(hinge=np.array([0.,0.,1.]),closed_handle=np.array([-.7,0.,1.]),open_handle=np.array([-.7*np.cos(np.radians(47)),-.7*np.sin(np.radians(47)),1.]))
    remaining=advance_open_memory(memory,-43.)
    assert abs(remaining-90.)<1e-6
    assert np.allclose(memory['open_handle'],[0.,-.7,1.],atol=1e-6)


def test_a_partial_handle_pull_is_finished_by_pushing_from_inside():
    from cross_episode_sim.runtime import fixture_controls
    source=inspect.getsource(fixture_controls.LiveFixtureControls.execute)
    assert 'PARTIAL_OPENING_MIN_DEGREES' in source and 'self._push_rest_open(target,root' in source
    push=inspect.getsource(fixture_controls.LiveFixtureControls._push_rest_open)
    assert "opening=True" in push and "self.observed_door_angle(target,total)" in push


def test_leaf_angle_is_read_from_points_lined_up_through_the_hinge():
    from cross_episode_sim.code_policy.panel_closing import leaf_angle_from_points,set_open_memory
    rng=np.random.default_rng(13)
    hinge=np.array([0.,0.,1.1]);closed=np.array([.57,0.,1.1]);remembered=np.array([.57*np.cos(np.radians(-46)),.57*np.sin(np.radians(-46)),1.1])
    true=np.radians(-59.)                                   # the leaf really stands at 59 deg, opened toward -y
    r=rng.uniform(.05,.57,3000);z=rng.uniform(.95,1.3,3000)
    leaf=np.c_[r*np.cos(true),r*np.sin(true),z]
    front=np.c_[rng.uniform(.0,.6,3000),np.zeros(3000),rng.uniform(.95,1.3,3000)]   # the appliance's closed front
    seen=leaf_angle_from_points(np.r_[leaf,front],hinge,closed,remembered,46.)
    assert abs(seen-59.)<3.5                                # an exact line is spread over the 5 deg smoothing window
    memory=dict(hinge=hinge,closed_handle=closed,open_handle=remembered)
    set_open_memory(memory,seen)
    assert abs(np.degrees(np.arctan2(memory['open_handle'][1],memory['open_handle'][0]))+seen)<1e-6
    assert leaf_angle_from_points(front[:10],hinge,closed,remembered,46.) is None


def test_leaf_angle_is_read_at_the_far_edge_of_a_thick_door():
    from cross_episode_sim.code_policy.panel_closing import leaf_angle_from_points
    rng=np.random.default_rng(17)
    hinge=np.array([0.,0.,1.1]);closed=np.array([.67,0.,1.1]);remembered=np.array([.67*np.cos(np.radians(62)),.67*np.sin(np.radians(62)),1.1])
    def at(deg,n,lo=.2,hi=.67):
        r=rng.uniform(lo,hi,n);a=np.radians(deg+rng.normal(0.,.6,n));return np.c_[r*np.cos(a),r*np.sin(a),rng.uniform(1.,1.2,n)]
    # Inner bins dominate 7 deg behind the leaf; the leaf's own skin ends at 84 deg (fix118 shape).
    cloud=np.r_[at(77.,9000),at(80.,5000),at(83.,4000)]
    seen=leaf_angle_from_points(cloud,hinge,closed,remembered,62.)
    assert 81.<=seen<=86.


def test_failed_call_dump_links_the_episode_model_instead_of_copying_it(tmp_path):
    from types import SimpleNamespace
    from cross_episode_sim.runtime import artifacts
    shared=tmp_path/'recording_model.mjb';shared.write_bytes(b'model'*1000)
    target=tmp_path/'handoff'/'private_checkpoint'/'model.mjb'
    runtime=SimpleNamespace(artifacts={str(target.resolve()):'live episode model'})
    artifacts.persist_under(runtime,[tmp_path],None,None,model_link=shared)
    assert target.exists() and target.stat().st_ino==shared.stat().st_ino     # one file, two names


def _microwave_front(bar=True):
    """Door panel at x=0.70 facing the robot, countertop lip under it, bar handle beside the point."""
    rng=np.random.default_rng(3)
    y,z=np.meshgrid(np.arange(-.25,.25,.006),np.arange(.92,1.36,.006));panel=np.c_[np.full(y.size,.70),y.ravel(),z.ravel()]
    y,x=np.meshgrid(np.arange(-.25,.25,.006),np.arange(.62,.70,.006));lip=np.c_[x.ravel(),y.ravel(),np.full(x.size,.915)]
    parts=[panel,lip]
    if bar:
        y,z=np.meshgrid(np.arange(.09,.12,.004),np.arange(.96,1.32,.004));parts.append(np.c_[np.full(y.size,.66),y.ravel(),z.ravel()])
    cloud=np.concatenate(parts);return cloud+rng.normal(0,.0005,cloud.shape)


def test_microwave_point_on_the_countertop_lip_moves_to_the_bar_handle():
    from cross_episode_sim.perception.handle_surface import vertical_bar_handle,nearby_vertical_bar,vertical_bar_middle
    cloud=_microwave_front();lip=np.array([.64,.0,.915])
    assert not vertical_bar_handle(cloud,lip)
    seed=nearby_vertical_bar(cloud,lip);assert seed is not None and .08<seed[1]<.13
    grasp=vertical_bar_middle(cloud,seed)
    assert .08<grasp[1]<.13 and 1.05<grasp[2]<1.25      # on the bar, clear of its ends and of anything below


def test_no_bar_beside_the_point_leaves_it_alone():
    from cross_episode_sim.perception.handle_surface import nearby_vertical_bar
    assert nearby_vertical_bar(_microwave_front(bar=False),np.array([.64,.0,.915])) is None


def test_point_on_the_bare_door_far_from_the_bar_finds_it_with_a_wide_reach():
    from cross_episode_sim.perception.handle_surface import nearby_vertical_bar
    cloud=_microwave_front();door=np.array([.70,-.20,1.10])        # bare door, 0.3 m from the bar
    assert nearby_vertical_bar(cloud,door,reach=.20) is None
    seed=nearby_vertical_bar(cloud,door,reach=.70);assert seed is not None and .08<seed[1]<.13


def test_bar_search_can_refuse_a_bar():
    from cross_episode_sim.perception.handle_surface import nearby_vertical_bar
    cloud=_microwave_front();lip=np.array([.64,.0,.915])
    assert nearby_vertical_bar(cloud,lip,accept=lambda seed:False) is None


def test_bar_must_reach_the_pointed_height():
    from cross_episode_sim.perception.handle_surface import nearby_vertical_bar,bar_spans_height
    cloud=_microwave_front();seed=nearby_vertical_bar(cloud,np.array([.64,.0,.915]))
    assert bar_spans_height(cloud,seed,.915) and bar_spans_height(cloud,seed,1.29)
    assert not bar_spans_height(cloud,seed,.70) and not bar_spans_height(cloud,seed,1.55)


def _door_memory(open_degrees):
    a=np.radians(open_degrees)
    return dict(hinge=np.zeros(3),closed_handle=np.array([.6,0.,0.]),open_handle=np.array([.6*np.cos(a),.6*np.sin(a),0.]))


def test_remembered_open_angle_and_short_close_rule():
    from cross_episode_sim.code_policy import panel_closing as pc
    assert abs(pc.remembered_open_degrees(_door_memory(89.2))-89.2)<1e-6
    # finished closes measured 1-3.5 deg short; the unfinished one 10.6
    assert 89.2-78.6>pc.UNFINISHED_CLOSE_DEGREES>3.5
    assert issubclass(pc.ShortPanelClose,pc.PartialPanelClose)


def test_finishing_stroke_that_cannot_start_keeps_the_completed_push(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.code_policy import fixture_profiles as fp
    monkeypatch.setattr(fp,'CLOSED_PRESS_DOORS',('microwave',))
    from cross_episode_sim.code_policy.panel_closing import ShortPanelClose
    controls=fp.HouseholdFixtureControls.__new__(fp.HouseholdFixtureControls)
    controls.runtime=SimpleNamespace(door_push_memory=None);controls.hinges=dict(microwave=_door_memory(89.2))
    strokes=[];pressed=[];controls._press_closed=lambda target,progress:pressed.append(target)
    def stroke(target,progress):
        strokes.append(1)
        if len(strokes)==1:raise ShortPanelClose(78.6,'push ended 10.6 deg short of closed')
        return None
    controls._closing_stroke=stroke
    outcome=controls.execute('close_door',target='microwave',progress=lambda *a:None)
    assert outcome.success and len(strokes)==2 and pressed==['microwave']      # a closed microwave door is pressed home
    # and a finishing stroke that works returns its own outcome
    strokes.clear();controls.hinges=dict(microwave=_door_memory(89.2))
    done=fp.SkillOutcome(True,'second') if hasattr(fp,'SkillOutcome') else None
    def stroke2(target,progress):
        strokes.append(1)
        if len(strokes)==1:raise ShortPanelClose(78.6,'short')
        return done
    controls._closing_stroke=stroke2
    assert controls.execute('close_door',target='microwave',progress=lambda *a:None) is done


def test_press_is_off_by_default():
    from cross_episode_sim.code_policy import fixture_profiles as fp
    assert fp.CLOSED_PRESS_DOORS==()


def test_only_a_pushed_shut_named_door_is_pressed_home(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.code_policy import fixture_profiles as fp
    monkeypatch.setattr(fp,'CLOSED_PRESS_DOORS',('microwave',))
    controls=fp.HouseholdFixtureControls.__new__(fp.HouseholdFixtureControls)
    pressed=[];controls._press_closed=lambda target,progress:pressed.append(target)
    done=fp.SkillOutcome(True,'observed_panel_push_completed; final goal scored separately')
    for target,skill,outcome in (('fridge','close_door',done),('microwave','open_door',done),
            ('microwave','close_door',fp.SkillOutcome(True,'commands_completed_fixture_goal_not_scored')),
            ('microwave','close_door',fp.SkillOutcome(False,'x'))):
        controls.hinges={target:_door_memory(89.)};controls._execute=lambda skill,target,progress,o=outcome:o
        assert controls.execute(skill,target=target,progress=lambda *a:None) is outcome
    assert pressed==[]
    controls.hinges=dict(microwave=_door_memory(89.));controls._execute=lambda skill,target,progress:done
    controls.execute('close_door',target='microwave',progress=lambda *a:None);assert pressed==['microwave']


def test_past_edge_dock_stands_beyond_the_free_edge_on_the_outer_side():
    from types import SimpleNamespace
    from cross_episode_sim.code_policy.closing_redock import PastEdgeRedocking,closing_face
    # hinge at origin, closed leaf along +x, closed face looking toward -y; door opened 25 deg (toward -y)
    a=np.radians(-25.);memory=dict(hinge=np.zeros(3),closed_handle=np.array([.68,0.,0.]),closed_normal=np.array([0.,-1.,0.]),
        open_handle=np.array([.68*np.cos(a),.68*np.sin(a),0.]))
    dock=PastEdgeRedocking.__new__(PastEdgeRedocking)
    dock.expected,dock.approach_normal_map=closing_face(memory);dock.memory=memory
    PastEdgeRedocking.__init__.__wrapped__ if hasattr(PastEdgeRedocking.__init__,'__wrapped__') else None
    hinge=memory['hinge'][:2];along=dock.expected[:2]-hinge;along/=np.linalg.norm(along)
    outer=np.asarray(dock.approach_normal_map)[:2].copy()
    turn=np.radians(30.);expected=np.cos(turn)*along+np.sin(turn)*outer
    # build through the real constructor path without sensors
    import cross_episode_sim.code_policy.closing_redock as module
    original=module.ClosingRedocking.__init__
    module.ClosingRedocking.__init__=lambda self,runtime,sensors,mapper,memory:(setattr(self,'memory',memory),
        setattr(self,'expected',closing_face(memory)[0]),setattr(self,'approach_normal_map',closing_face(memory)[1])) and None
    try:
        real=PastEdgeRedocking(None,None,None,memory)
        inner=PastEdgeRedocking(None,None,None,memory,turn_degrees=-30.)
    finally:module.ClosingRedocking.__init__=original
    assert np.allclose(real.approach_normal_map[:2],expected) and real.mid_stroke_docks((.65,))==[]
    assert real.strict_frontal and real.no_closing_tiers
    assert (inner.approach_normal_map[:2])@outer<0                    # a negative turn stands on the inner side
    assert module.PUSH_OPEN_STAND_TURNS[0]<0                          # inner side is tried first (fix129)
    stand=real.expected[:2]+.63*real.approach_normal_map[:2]
    assert np.linalg.norm(stand-hinge)>np.linalg.norm(real.expected[:2]-hinge)+.4      # beyond the free edge
    assert (stand-real.expected[:2])@outer>0                                           # on the outer (room) side


def test_controls_on_a_face_first_appliance_also_face_it_first():
    import inspect
    from cross_episode_sim.runtime import qwen_pipeline
    source=inspect.getsource(qwen_pipeline.PipelineRedocking.__call__)
    assert "face_fixture is not None and 'not visible' in str(exc)" in source
    assert "session.redock(face_fixture,'approach',FACE_APPLIANCE_DISTANCE)" in source
    names=qwen_pipeline.FACE_APPLIANCE_FIRST
    pick=lambda target:next((n for n in names if n in target.lower() and any(w in target.lower() for w in ('key','button','knob'))),None)
    assert pick('Start key at the bottom of the microwave keypad')=='microwave'
    assert pick('Stop key on the microwave keypad')=='microwave' and pick('food container inside microwave') is None


def _microwave_over_cabinet():
    """Microwave front with its bar, the countertop lip, and a cabinet door with a vertical handle below."""
    rng=np.random.default_rng(5);parts=[_microwave_front()]
    y,z=np.meshgrid(np.arange(-.25,.25,.006),np.arange(.45,.90,.006));parts.append(np.c_[np.full(y.size,.64),y.ravel(),z.ravel()])   # cabinet door
    y,z=np.meshgrid(np.arange(-.14,-.11,.004),np.arange(.65,.86,.004));parts.append(np.c_[np.full(y.size,.60),y.ravel(),z.ravel()]) # its handle
    cloud=np.concatenate(parts);return cloud+rng.normal(0,.0005,cloud.shape)


def test_cabinet_handle_under_the_microwave_is_not_its_bar_handle():
    """fix133: a point 1.00 m up on the microwave was moved to the cabinet handle (middle 0.75 m)."""
    from cross_episode_sim.perception.handle_surface import nearby_vertical_bar,vertical_bar_middle,bar_matches_point
    cloud=_microwave_over_cabinet();point=np.array([.70,-.12,1.00])        # on the microwave door, right above the cabinet handle
    seed=nearby_vertical_bar(cloud,point,reach=.70,accept=lambda s:bar_matches_point(cloud,s,float(point[2])))
    assert seed is not None
    middle=vertical_bar_middle(cloud,seed);assert middle[2]>.95 and .08<middle[1]<.13      # the microwave's own bar
    low=np.array([.60,-.125,.75]);assert not bar_matches_point(cloud,low,1.00)             # the cabinet handle is refused


def test_microwave_approach_falls_back_to_spots_away_from_the_open_leaf():
    import inspect
    from cross_episode_sim.code_policy import household_session
    assert all(k>0 for k in household_session.MICROWAVE_APPROACH_SHIFTS)
    source=inspect.getsource(household_session)
    assert "'No loaded aperture-facing dock' not in str(exc)" in source and 'away_from_open_leaf(aperture)' in source


def test_point_near_the_top_of_the_bar_moves_to_its_middle():
    from cross_episode_sim.perception.handle_surface import bar_middle_point,bar_height_range
    cloud=_microwave_front();top=np.array([.66,.105,1.29])          # 3 cm from the top end of a 0.96-1.32 m bar
    low,high=bar_height_range(cloud,top);assert low<1.0 and high>1.28
    middle=bar_middle_point(cloud,top)
    assert 1.08<middle[2]<1.20 and .08<middle[1]<.13
    centre=np.array([.66,.105,1.14]);assert abs(bar_middle_point(cloud,centre)[2]-1.14)<.04   # a middle point stays


def test_push_open_that_ends_on_resistance_near_the_request_is_at_the_stop():
    from cross_episode_sim.runtime.fixture_controls import opened_to_its_stop
    assert opened_to_its_stop(80.8,90.,'resisted_motion')                  # fix132: privately at 90
    assert not opened_to_its_stop(80.8,90.,'travel_budget_exhausted')
    assert not opened_to_its_stop(81.3,90.,'motion_path_blocked: No collision-clear IK solution: []')
    assert not opened_to_its_stop(60.,90.,'resisted_motion') and not opened_to_its_stop(None,90.,'resisted_motion')


def test_failed_pick_inside_the_microwave_stands_closer_for_the_next_look(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.runtime import household_components as hc
    redocks=[];looks=[]
    class Session:
        last_pick_output=None
        def execute_skill(self,name,options):
            looks.append(1);return dict(success=len(looks)==3,reason='observed_pickup_result')
        def redock(self,target,purpose,distance):redocks.append(round(distance,3));return dict(success=True,final_distance_m=distance)
    monkeypatch.setattr(hc,'live_session',lambda *a,**k:Session())
    monkeypatch.setattr(hc,'load_memory',lambda *a,**k:{},raising=False)
    try:hc.pick_confined(SimpleNamespace(qwen_docking=False),None,None,None,'food container inside microwave')
    except Exception:pass                      # what follows the retry loop needs real run files
    assert len(looks)==3 and redocks==[.75,.70]


def test_first_microwave_placement_dock_shifts_away_from_the_leaf_when_the_centre_has_no_route(monkeypatch):
    """fix128/fix134/fix136: the centred dock was refused and the placement was never tried."""
    from cross_episode_sim.runtime import household_components as hc
    docks=[]
    class Session:
        last_placement_output=None
        def aperture(self,name,map_frame=False):
            return dict(hinge=[0.,0.,1.],closed_handle=[.5,0.,1.],closed_normal=[0.,-1.,0.],open_handle=[0.,-.5,1.])
        def redock(self,target,purpose,distance,lateral_offset):
            docks.append(round(lateral_offset,3))
            if len(docks)<3:raise RuntimeError("No loaded aperture-facing dock: [{'event': 'placement_route_unverified'}]")
        def execute_skill(self,name,options):return dict(success=True)
    monkeypatch.setattr(hc,'live_session',lambda *a,**k:Session())
    outcome,_=hc.place_microwave(None,None,None,None,'food container')
    assert outcome['success'] and docks[0]==0. and abs(docks[1])==.06 and abs(docks[2])==.10
    assert docks[1]*docks[2]>0                                   # both on the same side: away from the leaf


def test_microwave_placement_looks_again_from_another_spot_when_the_floor_is_not_seen(monkeypatch):
    """fix139: 'support_not_visible' ended the placement with the food still held."""
    from cross_episode_sim.runtime import household_components as hc
    docks=[];places=[]
    class Session:
        last_placement_output=None
        def aperture(self,name,map_frame=False):
            return dict(hinge=[0.,0.,1.],closed_handle=[.5,0.,1.],closed_normal=[0.,-1.,0.],open_handle=[0.,-.5,1.])
        def redock(self,target,purpose,distance,lateral_offset):docks.append((distance,round(abs(lateral_offset),3)))
        def execute_skill(self,name,options):
            places.append(1)
            return dict(success=len(places)==3,reason='stage_failed:test_aperture_place: support_not_visible: redock to expose the support')
    monkeypatch.setattr(hc,'live_session',lambda *a,**k:Session())
    outcome,_=hc.place_microwave(None,None,None,None,'food container')
    assert outcome['success'] and len(places)==3 and docks==[(.6,0.),(.65,0.),(.6,.06)]


def test_front_normal_of_a_door_seen_at_a_grazing_angle_points_at_the_robot():
    from cross_episode_sim.perception.handle_surface import front_normal_toward_robot
    rng=np.random.default_rng(7)
    # door front in the plane y=1.0 (robot at the origin, looking along +x: a grazing view)
    x,z=np.meshgrid(np.arange(1.2,1.9,.01),np.arange(.6,1.5,.01));door=np.c_[x.ravel(),np.full(x.size,1.0),z.ravel()]+rng.normal(0,.002,(x.size,3))
    normal=front_normal_toward_robot(door,np.array([1.5,1.0,1.05]))
    assert normal is not None and abs(normal[0])<.1 and normal[1]<-.99
    # a corner (two faces) is not one flat front
    y,z=np.meshgrid(np.arange(.7,1.0,.01),np.arange(.6,1.5,.01));side=np.c_[np.full(y.size,1.2),y.ravel(),z.ravel()]
    assert front_normal_toward_robot(np.concatenate([door[door[:,0]<1.5],side]),np.array([1.22,.98,1.05])) is None
    assert front_normal_toward_robot(door[:50],np.array([1.5,1.0,1.05])) is None


def test_open_floor_direction_points_from_a_fixture_front_to_the_floor_before_it():
    from types import SimpleNamespace
    from cross_episode_sim.code_policy.redocking import SensorRedocking
    res=.05;n=80;floor=np.zeros((n,n),int);evidence=np.zeros((n,n,4),int)
    # map x right (cols), y up (rows); a fixture occupies y>=2.0, open floor seen for y<2.0
    floor[:40,:]=3;evidence[40:,:,1]=3
    grid=SimpleNamespace(floor_views=floor,evidence=evidence,config=SimpleNamespace(minimum_floor_views=2),
        cell_to_world=lambda ij:(np.asarray(ij,float)+.5)*res)
    dock=SensorRedocking.__new__(SensorRedocking);dock.runtime=SimpleNamespace(observed_navigation_map=grid)
    direction=dock.open_floor_direction(np.array([2.0,2.0]))
    assert direction is not None and abs(direction[0])<.1 and direction[1]<-.99
    floor[:,:]=3;evidence[:]=0                                     # floor all round: no front to speak of
    assert dock.open_floor_direction(np.array([2.0,2.0])) is None


def test_declined_key_is_retried_once_from_a_new_view():
    import inspect
    from cross_episode_sim.evaluation import sensor_household
    from cross_episode_sim.runtime import qwen_pipeline
    source=inspect.getsource(sensor_household)
    assert "name not in ('push_button','rotate_knob') or not retriable" in source
    assert "'no_reachable_m2t2_knob_grasp' in str(exc)" in source
    assert "interface.redock(target=kw['target'],purpose='handle',face_first=True)" in source and "attempt('_retry')" in source
    call=inspect.getsource(qwen_pipeline.PipelineRedocking.__call__)
    assert 'face_first=False' in call and "if face_first and face_fixture is not None:raise RuntimeError('Target not visible" in call


def test_microwave_bar_grasp_slides_from_the_top_end_to_the_preferred_height():
    from cross_episode_sim.perception.handle_surface import slide_along_vertical_bar
    from cross_episode_sim.runtime import fixture_controls
    assert .95<fixture_controls.MICROWAVE_HANDLE_GRASP_HEIGHT<1.25
    cloud=_microwave_front();top=np.array([.66,.105,1.29])
    lower=slide_along_vertical_bar(cloud,top,fixture_controls.MICROWAVE_HANDLE_GRASP_HEIGHT)
    assert abs(lower[2]-fixture_controls.MICROWAVE_HANDLE_GRASP_HEIGHT)<.02 and .08<lower[1]<.13


def test_short_close_threshold_catches_fix145_second_microwave_close():
    import cross_episode_sim.code_policy.panel_closing as pc
    # 81.2 of 86.9 deg measured (door privately 4.2 deg open, fix145) and 84.3 of
    # about 89 (4.7 deg open, fix162) must get the finishing stroke, which now
    # pushes past closed; a door the robot measures within 1.5 deg is left.
    assert 86.9-81.2>89.-84.3>pc.FINISH_SHORT_CLOSE_DEGREES>=1.


def test_microwave_grasp_slide_must_move_toward_the_requested_height():
    from cross_episode_sim.runtime.fixture_controls import slides_toward,MICROWAVE_HANDLE_GRASP_HEIGHT as H
    assert not slides_toward(H,.973,.84)        # fix148: slid down off the bar toward the cabinet handle
    assert slides_toward(H,1.29,1.10)           # top end of the bar down to the grasp height
    assert slides_toward(H,.98,1.10)
