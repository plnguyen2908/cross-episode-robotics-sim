import numpy as np
from types import SimpleNamespace
from cross_episode_sim.navigation.semantic_navigation import astar_observed
from cross_episode_sim.perception.visual_targets import Detection,locate_detection,VisualTargetMap


def test_astar_requires_observed_connection_and_does_not_cut_corners():
    free=np.ones((7,7),bool);free[3,:]=False
    assert not astar_observed(free,(1,1),(5,5))
    free[3,3]=True;path=astar_observed(free,(1,1),(5,5))
    assert (3,3) in path
    for a,b in zip(path,path[1:]):
        if a[0]!=b[0] and a[1]!=b[1]:assert free[a[0],b[1]] and free[b[0],a[1]]


def camera():
    return SimpleNamespace(depth_m=np.ones((20,20))*2,valid_depth=np.ones((20,20),bool),robot_mask=np.zeros((20,20),bool),K=np.array([[10,0,9.5],[0,10,9.5],[0,0,1]]),T_base_optical=np.eye(4),frame_id='navigation')


def test_depth_localization_uses_camera_calibration_and_slam():
    c=camera();T=np.eye(4);T[0,3]=3
    assert np.allclose(locate_detection(c,Detection('cup',.9,(0,0,20,20)),T),[3,0,2])
    c.robot_mask[:]=True
    assert locate_detection(c,Detection('cup',.9,(0,0,20,20)),T) is None


def test_target_confirmation_requires_distinct_accepted_observations():
    targets=VisualTargetMap();c=camera();d=[Detection('cup',.9,(0,0,20,20))]
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    targets.update(c,d,tracking,0.);targets.update(c,d,tracking,0.)
    assert targets.targets[0].observations==1
    targets.update(c,d,tracking,1.);assert targets.targets[0].observations==2
    tracking.accepted=False;targets.update(c,d,tracking,2.)
    assert targets.targets[0].observations==2


def test_unknown_target_continues_exploration_and_tracking_loss_stops():
    from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    class Detector:
        calls=0
        def detect(self,rgb):self.calls+=1;return []
    class Explorer:
        terminal=None
        calls=0
        def step(self,*args):self.calls+=1;return dict(status='exploring',forward=.1,yaw=0.)
    detector=Detector();explorer=Explorer()
    nav=SemanticNavigator(ObservedNavigationMap(MapConfig(extent_m=4)),detector,explorer=explorer)
    from cross_episode_sim.navigation.local_scan import ScanConfig
    packet=SimpleNamespace(timestamp=0.,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)),cameras={'navigation':SimpleNamespace(rgb=np.zeros((2,2,3),np.uint8))})
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    assert nav.step(packet,tracking,0.)['status']=='exploring'
    assert nav.mode=='explore' and explorer.calls==1 and detector.calls==1
    tracking.accepted=False
    assert nav.step(packet,tracking,.02)['forward']==0
    assert explorer.calls==1 and detector.calls==1
    tracking.accepted=True
    assert nav.step(packet,tracking,1.)['status']=='stale_observation'
    assert explorer.calls==1


def test_floor_patch_is_not_accepted_as_a_dining_table():
    c=camera();T=np.eye(4);T[2,3]=-2
    assert locate_detection(c,Detection('dining table',.9,(0,0,20,20)),T) is None
    T[2,3]=-1.2
    assert locate_detection(c,Detection('dining table',.9,(0,0,20,20)),T) is not None


def test_tentative_target_is_inspected_but_retries_need_new_evidence():
    from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    nav=SemanticNavigator(ObservedNavigationMap(MapConfig(extent_m=4)),None,target='cup')
    t=SimpleNamespace(track_id=7,label='cup',observations=1)
    assert nav._candidate_allowed(t,0.)
    t.observations=0
    assert not nav._candidate_allowed(t,0.)
    t.observations=1;nav.attempts[7]=3;nav.attempt_observations[7]=1
    assert not nav._candidate_allowed(t,0.)
    t.observations=2
    assert nav._candidate_allowed(t,0.)
    nav.retry_after[7]=10.
    assert not nav._candidate_allowed(t,9.)


def test_failed_confirmation_replans_view_without_claiming_success():
    from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    nav=SemanticNavigator(ObservedNavigationMap(MapConfig(extent_m=4)),None,target='cup')
    nav.follower.target=SimpleNamespace(track_id=7)
    nav.mode='align';nav.follower.terminal='failed';nav.follower.path=[np.zeros(2)]
    result=nav._recover_target(np.array([1.,2.]),5.,'confirmation_failed')
    assert result['forward']==0 and result['yaw']==0
    assert nav.terminal is None and nav.mode=='explore'
    assert nav.follower.target is None and not nav.follower.path
    assert np.array_equal(nav.follower.rejected_views[7][0],[1.,2.])
    assert nav.retry_after[7]>5.


def test_blocked_departure_resenses_without_commanding_motion():
    from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    from cross_episode_sim.navigation.local_scan import ScanConfig
    class Detector:
        def detect(self,rgb):return []
    class Explorer:
        terminal=None
        def step(self,*args):
            self.terminal='departure_blocked'
            return dict(status='departure_blocked',forward=0.,yaw=0.)
    nav=SemanticNavigator(ObservedNavigationMap(MapConfig(extent_m=4)),Detector(),explorer=Explorer())
    packet=SimpleNamespace(timestamp=0.,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)),cameras={'navigation':SimpleNamespace(rgb=np.zeros((2,2,3),np.uint8))})
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    assert nav.step(packet,tracking,0.)['status']=='resense_blocked_departure'
    assert nav.terminal is None
    for stamp in [1.,3.,4.,6.]:
        packet.timestamp=stamp
        result=nav.step(packet,tracking,stamp)
        assert result['forward']==0 and result['yaw']==0
    assert nav.terminal=='departure_blocked'  # No endless retry or clearance bypass.


def test_close_approach_prefers_distance_and_does_not_cross_unknown():
    from cross_episode_sim.navigation.semantic_navigation import TargetFollower
    from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    grid=ObservedNavigationMap(MapConfig(extent_m=6))
    grid.floor_views[:]=grid.config.minimum_floor_views
    follower=TargetFollower(grid,HouseExplorationConfig())
    follower.target=SimpleNamespace(track_id=1,position=np.array([1.5,0.,1.]))
    assert follower._plan(np.zeros(2),0.) is None
    assert np.linalg.norm(follower.goal-follower.target.position[:2])<.85
    assert np.linalg.norm(follower.goal)>.5  # Does not accept the shorter distant view.
    grid.floor_views[:]=0  # Unknown floor cannot become a shortcut to the object.
    follower.support[:]=False
    assert follower._plan(np.zeros(2),1.) is not None


def test_refined_target_outside_standoff_replans_instead_of_spinning():
    from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    from cross_episode_sim.navigation.local_scan import ScanConfig
    nav=SemanticNavigator(ObservedNavigationMap(MapConfig(extent_m=6)),SimpleNamespace(detect=lambda rgb:[]),preferred_distance=1.1)
    target=SimpleNamespace(position=np.array([2.,0.,.8]),observations=3,last_seen=1.,track_id=0)
    nav.targets.targets=[target];nav.follower.target=target;nav.mode='align';nav.alignment_started=0.
    nav.observe=lambda *args:None
    calls=[];nav.follower._plan=lambda xy,now:calls.append((xy.copy(),now))
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)),cameras={'navigation':camera()})
    result=nav.step(packet,SimpleNamespace(accepted=True,T_map_base=np.eye(4)),1.)
    assert calls and nav.mode=='navigate'
    assert result['status']=='target_geometry_updated'
    assert result['forward']==result['yaw']==0.
    assert nav.terminal is None
