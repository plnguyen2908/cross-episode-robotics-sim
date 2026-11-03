from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation.onering import OneRingLocalNavigator,BASE_ACTIONS
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.local_scan import ScanConfig
from cross_episode_sim.navigation.readiness import ManipulationReadiness
from cross_episode_sim.mapping.room_memory import ObservedRoomMemory


def packet(stamp=0.):
    c=NS(rgb=np.zeros((8,8,3),np.uint8),depth_m=np.ones((8,8))*8,valid_depth=np.zeros((8,8),bool),robot_mask=None,K=np.eye(3),T_base_optical=np.eye(4))
    return NS(timestamp=stamp,arm=NS(position=np.array(ScanConfig().stowed_arm)),cameras={'navigation':c,'wrist':c})

def tracking():return NS(accepted=True,T_map_base=np.eye(4))

def test_ring_action_uses_feedback_and_not_one_prediction_per_physics_step():
    class Policy:
        calls=0
        def predict(self,*args):self.calls+=1;return 'm'
    grid=ObservedNavigationMap(MapConfig(extent_m=6));grid.floor_views[:]=2
    p=Policy();nav=OneRingLocalNavigator(grid,p,'find a bowl');t=tracking()
    assert nav.step(packet(),t,0.)['forward']>0
    assert nav.step(packet(.02),t,.02)['forward']>0 and p.calls==1
    t.T_map_base[0,3]=.2
    assert nav.step(packet(.04),t,.04)['status']=='ring_action_complete'
    assert p.calls==1

def test_ring_stop_and_unsupported_action_are_not_readiness():
    grid=ObservedNavigationMap(MapConfig(extent_m=6));grid.floor_views[:]=2
    for action,status in [('end','local_done'),('p','unsupported_ring_action')]:
        nav=OneRingLocalNavigator(grid,NS(predict=lambda *args:action),'find a bowl')
        r=nav.step(packet(),tracking(),0.)
        assert r['status']==status and r['forward']==r['yaw']==0
    assert BASE_ACTIONS['l'][1]>0 and BASE_ACTIONS['r'][1]<0

def test_lost_tracking_never_calls_ring():
    nav=OneRingLocalNavigator(ObservedNavigationMap(MapConfig(extent_m=6)),NS(predict=lambda *args:1/0),'bowl')
    t=tracking();t.accepted=False
    assert nav.step(packet(),t,0.)['status']=='tracking_lost'

def test_readiness_requires_fresh_evidence_and_collision_checked_reach():
    t=NS(position=np.array([.6,0,1]),last_seen=0.,observations=2)
    assert ManipulationReadiness().check(t,packet(),tracking(),None).reason=='reachability_unverified'
    assert not ManipulationReadiness(lambda *args:(True,False)).check(t,packet(),tracking(),None).ready
    assert ManipulationReadiness(lambda *args:(True,True)).check(t,packet(),tracking(),None).ready
    assert not ManipulationReadiness(lambda *args:(True,True)).check(t,packet(5),tracking(),None).ready

def test_room_labels_need_agreement_and_do_not_invent_region_ids():
    vlm=NS(label_with_prompt=lambda *args:'{"label":"kitchen","confidence":0.95,"evidence":"stove"}')
    rooms=ObservedRoomMemory(vlm,period=1.)
    region={'id':'R(2)','position':[0,0,0],'places':[[0,0,0]]}
    rooms.update(packet(),tracking(),[region]);assert rooms.current_label=='unknown'
    rooms.update(packet(1),tracking(),[region]);assert rooms.current_label=='kitchen'
    assert rooms.label_for('R(2)')=='kitchen' and len(rooms.destinations('kitchen',np.zeros(2)))==1
    rooms.update(packet(2),tracking(),[]);assert rooms.current_region is None and rooms.destinations('kitchen',np.zeros(2))==[]


def test_reverse_guard_checks_the_path_behind_the_robot():
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    grid=ObservedNavigationMap(MapConfig(extent_m=6));grid.floor_views[:]=2
    guard=HouseExplorer(grid);guard.layers(np.zeros(2))
    guard.near_points=np.array([[-.65,0.,.5]])
    assert guard._command_clear(np.zeros(2),0.,np.array([.4,0.]))
    assert not guard._command_clear(np.zeros(2),0.,np.array([-.4,0.]))


def test_room_transition_invalidates_old_visual_consensus():
    vlm=NS(label_with_prompt=lambda *args:'{"label":"kitchen","confidence":0.95}')
    rooms=ObservedRoomMemory(vlm,period=1.)
    first={'id':'r1','places':[[0,0,0]]};second={'id':'r2','places':[[2,0,0]]}
    t=tracking();rooms.update(packet(),t,[first,second]);rooms.update(packet(1),t,[first,second])
    assert rooms.current_label=='kitchen'
    t.T_map_base[0,3]=2
    rooms.update(packet(1.1),t,[first,second])
    assert rooms.current_label=='unknown'


def test_pending_room_vote_confirms_before_normal_five_second_scan_interval():
    calls=[]
    def label(*args):
        calls.append(1)
        return '{"label":"kitchen","confidence":0.95}'
    rooms=ObservedRoomMemory(NS(label_with_prompt=label))
    rooms.update(packet(),tracking(),[])
    rooms.update(packet(.1),tracking(),[])
    assert len(calls)==1 and rooms.current_label=='unknown'
    rooms.update(packet(.5),tracking(),[])
    assert len(calls)==2 and rooms.current_label=='kitchen'
    rooms.update(packet(1.),tracking(),[])
    assert len(calls)==2


def test_stale_nearby_object_does_not_hide_fresh_target():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    stale=NS(label='bowl',position=np.array([.5,0,1]),observations=20,last_seen=0.)
    fresh=NS(label='bowl',position=np.array([.7,0,1]),observations=3,last_seen=10.)
    nav=HybridNavigator.__new__(HybridNavigator)
    nav.semantic=NS(targets=NS(targets=[stale,fresh]),target_label='bowl')
    nav.readiness=NS(max_age=2.)
    assert nav._target(np.zeros(2),10.) is fresh
    assert nav._target(np.zeros(2),13.) is None


def test_blocked_ring_command_waits_for_new_observation():
    class Policy:
        calls=0
        def predict(self,*args):self.calls+=1;return 'b'
    grid=ObservedNavigationMap(MapConfig(extent_m=6));policy=Policy()
    nav=OneRingLocalNavigator(grid,policy,'bowl')
    grid.evidence[30,30,3]=2
    assert nav.step(packet(),tracking(),0.)['status']=='ring_collision_stop'
    assert nav.step(packet(),tracking(),.02)['status']=='await_fresh_ring_observation'
    assert policy.calls==1


def test_hybrid_stop_requests_redock_but_never_claims_success_by_distance():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer,HouseExplorationConfig
    grid=ObservedNavigationMap(MapConfig(extent_m=6));explorer=HouseExplorer(grid,HouseExplorationConfig())
    target=NS(position=np.array([1.2,0.,1.]),last_seen=0.,observations=2,label='bowl')
    follower=HouseExplorer(grid)
    semantic=NS(grid=grid,explorer=explorer,follower=follower,targets=NS(targets=[target]),target_label='bowl',last_mapping={},detection_rounds=3,observe=lambda *args:None)
    grid.floor_views[:]=2
    local=NS(guard=follower,reset_motion=lambda:None,policy=NS(reset=lambda:None),step=lambda *args:dict(status='local_done',forward=0.,yaw=0.))
    rooms=NS(current_label='kitchen',update=lambda *args:None)
    nav=HybridNavigator(semantic,local,rooms,ManipulationReadiness())
    assert nav.step(packet(),tracking(),0.)['status']=='confirm_found_object'
    assert not nav.object_search_success
    assert nav.step(packet(.1),tracking(),.1)['status']=='confirm_found_object'
    target.history=[{'timestamp':.2},{'timestamp':.3}];target.last_seen=.3
    assert nav.step(packet(.3),tracking(),.3)['status']=='redock_requested'
    assert nav.mode=='redock' and nav.object_search_success and nav.terminal is None
    semantic.step=lambda *args:dict(status='redock_needs_observation',forward=0.,yaw=0.)
    assert nav.step(packet(.4),tracking(),.4)['status']=='redock_route_unverified'
    # The search is verified, but docking is not. Repeated ticks never restart it.
    local.step=lambda *args:(_ for _ in ()).throw(AssertionError('search restarted'))
    assert nav.step(packet(.5),tracking(),.5)['status']=='redock_route_unverified'
    assert nav.terminal!='manipulation_ready'
    # A new done signal cannot reuse old observations as fresh confirmation.
    local.step=lambda *args:dict(status='local_done',forward=0.,yaw=0.)
    other=HybridNavigator(semantic,local,rooms,ManipulationReadiness())
    assert other.step(packet(1.),tracking(),1.)['status']=='confirm_found_object'
    assert other.step(packet(4.3),tracking(),4.3)['status']=='object_found_unconfirmed'
    assert not other.object_search_success


def test_incremental_guard_observes_forward_corridor_without_full_surroundings():
    from cross_episode_sim.navigation.onering import IncrementalMotionGuard
    grid=ObservedNavigationMap(MapConfig(extent_m=6));guard=IncrementalMotionGuard(grid)
    yy,xx=np.indices(grid.floor_views.shape)
    world=grid.origin+(np.stack((xx,yy),-1)+.5)*grid.config.resolution
    grid.floor_views[(world[:,:,0]>0)&(np.abs(world[:,:,1])<.8)]=2
    guard.layers(np.zeros(2))
    assert not guard.free[guard.cell(np.zeros(2))]  # Old full-disk erosion fails.
    assert guard._command_clear(np.zeros(2),0.,np.array([.2,0.]))
    assert not guard._command_clear(np.zeros(2),0.,np.array([-.2,0.]))


def test_incremental_guard_prior_never_erases_obstacles_or_expands_with_motion():
    from cross_episode_sim.navigation.onering import IncrementalMotionGuard
    grid=ObservedNavigationMap(MapConfig(extent_m=6));guard=IncrementalMotionGuard(grid)
    guard.layers(np.zeros(2))
    assert guard._command_clear(np.zeros(2),0.,np.array([0.,.1]))
    assert not guard._command_clear(np.array([.2,0.]),0.,np.array([0.,.1]))
    r,c=guard.cell(np.zeros(2));grid.evidence[r,c,3]=2;guard.layers(np.zeros(2))
    assert not guard._command_clear(np.zeros(2),0.,np.array([0.,.1]))


def test_redocking_does_not_silently_start_full_scan():
    from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
    class Explorer:
        terminal=None
        def step(self,*args):raise AssertionError('Local redocking must not start exploration')
    nav=SemanticNavigator(ObservedNavigationMap(MapConfig(extent_m=6)),NS(detect=lambda *args:[]),target='bowl',explorer=Explorer(),allow_exploration=False)
    assert nav.step(packet(),tracking(),0.)['status']=='redock_needs_observation'


def test_shape_docking_is_not_misreported_as_manipulation_readiness():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    g=ObservedNavigationMap(MapConfig(extent_m=6));follower=HouseExplorer(g)
    target=NS(position=np.array([.6,0.,1.]),last_seen=0.,observations=2,label='bowl')
    semantic=NS(grid=g,explorer=HouseExplorer(g),follower=follower,targets=NS(targets=[target]),target_label='bowl',last_mapping={},detection_rounds=3,observe=lambda *args:None)
    local=NS(reset_motion=lambda:None,policy=NS(reset=lambda:None))
    docking=NS(target=target,terminal='shape_docked',step=lambda *args:follower._stop('shape_docked'))
    nav=HybridNavigator(semantic,local,NS(current_label='unknown',update=lambda *args:None),ManipulationReadiness(),goal_room='',docking=docking)
    nav.mode='redock';nav.object_search_success=True
    result=nav.step(packet(),tracking(),0.)
    assert nav.docking_success and not nav.last_readiness.ready
    assert result['status']=='docked_reachability_unverified'


def test_visible_target_handoff_does_not_wait_for_ring_done():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    g=ObservedNavigationMap(MapConfig(extent_m=6));f=HouseExplorer(g)
    target=NS(position=np.array([3.3,0.,1.]),last_seen=0.,observations=3,label='bowl',history=[])
    semantic=NS(grid=g,explorer=HouseExplorer(g),follower=f,targets=NS(targets=[target]),target_label='bowl',last_mapping={},detection_rounds=3,observe=lambda *args:None)
    local=NS(reset_motion=lambda:None,policy=NS(reset=lambda:None),step=lambda *args:(_ for _ in ()).throw(AssertionError('unnecessary search')))
    nav=HybridNavigator(semantic,local,NS(current_label='kitchen',update=lambda *args:None),ManipulationReadiness(),docking=NS(reset=lambda t:None))
    assert nav.step(packet(),tracking(),0.)['status']=='confirm_visible_target'
    assert not nav.object_search_success
    assert nav.step(packet(.1),tracking(),.1)['status']=='confirm_found_object'
    target.history=[{'timestamp':.2},{'timestamp':.3}];target.last_seen=.3
    assert nav.step(packet(.3),tracking(),.3)['status']=='redock_requested'
    assert nav.object_search_success and not nav.docking_success


def test_unobserved_dock_corridor_resumes_search_without_false_success():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    g=ObservedNavigationMap(MapConfig(extent_m=6));f=HouseExplorer(g)
    target=NS(position=np.array([3.3,0.,1.]),last_seen=0.,observations=3,label='bowl',history=[])
    semantic=NS(grid=g,explorer=HouseExplorer(g),follower=f,targets=NS(targets=[target]),target_label='bowl',last_mapping={},detection_rounds=3,observe=lambda *args:None)
    local=NS(reset_motion=lambda:None,policy=NS(reset=lambda:None),step=lambda *args:f._stop('ring_l'))
    docking=NS(terminal='shape_route_unverified',step=lambda *args:f._stop('shape_route_unverified'))
    nav=HybridNavigator(semantic,local,NS(current_label='kitchen',update=lambda *args:None),ManipulationReadiness(),docking=docking)
    nav.mode='redock'
    assert nav.step(packet(),tracking(),0.)['status']=='observe_approach_corridor'
    assert nav.mode=='local' and not nav.docking_success
    assert nav.step(packet(.1),tracking(),.1)['status']=='ring_l'
    t=tracking();t.T_map_base[0,3]=.3;target.last_seen=.2
    assert nav.step(packet(.2),t,.2)['status']=='confirm_visible_target'


def test_ring_shape_guard_checks_rotation_braking_envelope():
    g=ObservedNavigationMap(MapConfig(extent_m=6));g.floor_views[:]=2
    nav=OneRingLocalNavigator(g,NS(predict=lambda *args:'l'),'bowl')
    seen=[]
    class Check:
        detail={'reason':'rotation_obstacle'}
        def swept(self,a,b):
            seen.append(b.copy())
            return b[2]<.12
    nav.shape_checker_factory=lambda *args:Check()
    assert not nav._shape_command_clear(packet(),np.zeros(2),0.,np.array([0.,.6]))
    assert max(p[2] for p in seen)>.12
    assert nav.guard.stop_detail['reason']=='rotation_obstacle'


def test_closer_instance_switch_requires_stop_and_fresh_repeated_evidence():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    g=ObservedNavigationMap(MapConfig(extent_m=6));f=HouseExplorer(g)
    old=NS(position=np.array([1.,0.,1.]),last_seen=1.,observations=3,label='bowl',track_id=0)
    new=NS(position=np.array([.6,0.,1.]),last_seen=1.,observations=3,label='bowl',track_id=1,history=[{'timestamp':.9},{'timestamp':1.}])
    semantic=NS(grid=g,explorer=HouseExplorer(g),follower=f,targets=NS(targets=[old,new]),target_label='bowl',last_mapping={},detection_rounds=3,observe=lambda *args:None)
    local=NS(reset_motion=lambda:None,policy=NS(reset=lambda:None))
    docking=NS(target=old,terminal=None,velocity=np.array([.2,0.]),step=lambda *args:f._stop('shape_docking'))
    docking.reset=lambda target:setattr(docking,'target',target)
    nav=HybridNavigator(semantic,local,NS(current_label='unknown',update=lambda *args:None),ManipulationReadiness(),goal_room='',docking=docking)
    nav.mode='redock';nav.found_target=old
    assert nav.step(packet(1.),tracking(),1.)['status']=='shape_docking'
    assert docking.target is old
    docking.velocity[:]=0
    new.history=[{'timestamp':-10.},{'timestamp':-9.}]
    assert nav.step(packet(1.05),tracking(),1.05)['status']=='shape_docking'
    assert docking.target is old
    new.history=[{'timestamp':.9},{'timestamp':1.}]
    assert nav.step(packet(1.1),tracking(),1.1)['status']=='redock_requested'
    assert docking.target is new


def test_ring_carry_posture_requires_payload_shape_checks():
    from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
    grid=ObservedNavigationMap(MapConfig(extent_m=6));grid.floor_views[:]=2
    observation=packet();observation.arm.position=observation.arm.position.copy();observation.arm.position[0]+=.2
    nav=OneRingLocalNavigator(grid,NS(predict=lambda *args:'m'),'find dining table',HouseExplorationConfig(arm_posture=tuple(observation.arm.position)))
    assert nav.step(observation,tracking(),0.)['status']=='carrying_requires_shape_checker'
    nav.shape_checker_factory=lambda *args:NS(swept=lambda *args:False,detail={'reason':'test_payload_collision'})
    assert nav.step(observation,tracking(),0.)['status']=='ring_collision_stop'


def test_explicit_unknown_floor_requests_wrist_observation():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    g=ObservedNavigationMap(MapConfig(extent_m=6));f=HouseExplorer(g)
    target=NS(position=np.array([3.3,0.,1.]),last_seen=0.,observations=3,label='bowl',history=[])
    semantic=NS(grid=g,explorer=HouseExplorer(g),follower=f,targets=NS(targets=[target]),target_label='bowl',last_mapping={},detection_rounds=3,observe=lambda *args:None)
    local=NS(reset_motion=lambda:None,policy=NS(reset=lambda:None),step=lambda *args:f._stop('ring_l'))
    docking=NS(events=[dict(event='pickup_candidates_rejected',rejected=[dict(detail=dict(reason='unobserved_base_support'))])],terminal='shape_route_unverified',step=lambda *args:f._stop('shape_route_unverified'))
    nav=HybridNavigator(semantic,local,NS(current_label='kitchen',update=lambda *args:None),ManipulationReadiness(),docking=docking)
    nav.mode='redock'
    assert nav.step(packet(),tracking(),0.)['status']=='needs_floor_observation'
    assert nav.terminal=='needs_floor_observation'
    assert not nav.docking_success


def test_unconfirmed_room_waypoint_is_not_replanned_forever():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    nav=HybridNavigator.__new__(HybridNavigator)
    nav.goal_room='kitchen';nav.visited_room_waypoints=[np.array([1.,0.])]
    nav.rooms=NS(destinations=lambda *args:[np.array([0.,0.]),np.array([1.1,0.]),np.array([4.,0.])])
    destinations=nav._room_destinations(np.zeros(2))
    assert len(destinations)==1 and np.array_equal(destinations[0],[4.,0.])


def test_search_stall_detects_circling_but_not_forward_progress():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    nav=HybridNavigator.__new__(HybridNavigator);nav.search_window=[]
    for t in range(15):
        assert not nav._search_stalled(np.array([.1*np.sin(t),0]),t)
    assert nav._search_stalled(np.array([0.,0.]),15.)
    nav.search_window=[]
    for t in range(40):
        assert not nav._search_stalled(np.array([.1*t,0.]),t)


def test_recovery_hypothesis_does_not_become_fresh_target():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator,RoomWaypointFollower
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    grid=ObservedNavigationMap(MapConfig(extent_m=8));grid.floor_views[:]=2
    explorer=HouseExplorer(grid)
    hint=NS(label='cup',position=np.array([2.,0.,1.]),observations=5,last_seen=-100.)
    nav=HybridNavigator.__new__(HybridNavigator)
    nav.semantic=NS(grid=grid,explorer=explorer,remembered_object_hypotheses=[hint],
                    targets=NS(targets=[]),target_label='cup')
    nav.local=NS(reset_motion=lambda:None)
    nav.mode='local';nav.events=[];nav.searched_viewpoints=[]
    nav.readiness=NS(max_age=2.)
    nav._start_search_recovery(np.zeros(2),20.)
    assert nav.mode=='search_recovery'
    assert isinstance(nav.search_recovery,RoomWaypointFollower)
    assert nav.search_recovery.path
    assert nav._target(np.zeros(2),20.) is None
    assert hint.last_seen==-100.


def room_entry_navigator():
    from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
    from cross_episode_sim.navigation.house_exploration import HouseExplorer
    grid=ObservedNavigationMap(MapConfig(extent_m=8));grid.floor_views[:]=2
    follower=HouseExplorer(grid)
    semantic=NS(grid=grid,explorer=HouseExplorer(grid),follower=follower,
                targets=NS(targets=[]),target_label='cup',last_mapping={},
                detection_rounds=3,observe=lambda *args:None)
    local=NS(reset_motion=lambda:None,policy=NS(reset=lambda:None),
             step=lambda *args:(_ for _ in ()).throw(AssertionError('premature One-Ring')))
    rooms=NS(current_label='kitchen',update=lambda *args:None,
             destinations=lambda *args:[np.array([.4,0]),np.array([2.,0])])
    return HybridNavigator(semantic,local,rooms,ManipulationReadiness())


def test_room_label_approaches_deeper_observed_viewpoint_before_ring():
    nav=room_entry_navigator()
    result=nav.step(packet(),tracking(),0.)
    assert result['status']=='room_viewpoint_approach_requested'
    assert nav.mode=='search_recovery' and nav.room_approach
    assert np.allclose(nav.search_recovery.goal,[2.,0])
    assert not nav.object_search_success
    nav.search_recovery.step=lambda *args:dict(status='arrived_frontier',forward=0.,yaw=0.)
    assert nav.step(packet(.1),tracking(),.1)['status']=='search_viewpoint_observed'
    assert nav.mode=='local' and not nav.object_search_success


def test_failed_room_approach_does_not_hand_off_to_ring():
    nav=room_entry_navigator();nav.step(packet(),tracking(),0.)
    nav.search_recovery.terminal='departure_blocked'
    assert nav.step(packet(.1),tracking(),.1)['status']=='room_approach_departure_blocked'
    assert nav.mode!='local' and not nav.object_search_success


def test_observed_hint_facing_checks_collision_and_stale_sensors():
    grid=ObservedNavigationMap(MapConfig(extent_m=6));grid.floor_views[:]=2
    nav=OneRingLocalNavigator(grid,NS(),'cup')
    r=nav.face_point(packet(),tracking(),0.,[0,2])
    assert r['forward']==0 and 0<r['yaw']<=.02
    nav.shape_checker_factory=lambda *args:None
    nav._shape_command_clear=lambda *args:False
    assert nav.face_point(packet(.02),tracking(),.02,[0,2])['status']=='viewpoint_turn_blocked'
    assert nav.face_point(packet(),tracking(),10.,[0,2])['status']=='stale_observation'


def test_single_sighting_pauses_but_never_authorizes_grasp():
    nav=room_entry_navigator();nav.goal_room=''
    sighting=NS(track_id=8,label='cup',position=np.array([1.,0,1]),observations=1,last_seen=0.)
    nav.semantic.targets.targets=[sighting]
    assert nav.step(packet(),tracking(),0.)['status']=='confirm_tentative_target'
    assert not nav.object_search_success and nav.found_target is None
    assert nav.step(packet(1),tracking(),1.)['status']=='confirm_tentative_target'
    nav.local.step=lambda *args:dict(status='ring_m',forward=.1,yaw=0.)
    assert nav.step(packet(2.3),tracking(),2.3)['status']=='ring_m'
    assert not nav.object_search_success

