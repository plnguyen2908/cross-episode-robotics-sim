from types import SimpleNamespace
import numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap
from cross_episode_sim.navigation.semantic_navigation import SemanticNavigator
from cross_episode_sim.navigation.station_exploration import StationExplorer


def landmark(label,observations,**extra):
    return dict(label=label,observations=observations,position=[1.,0.,.8],last_seen=1.,
                evidence=[dict(timestamp=0.),dict(timestamp=1.)],viewpoints=[[5.,5.,0.]],**extra)


def explorer_at(xy,landmarks,monkeypatch,statuses):
    """Station policy whose follower and frontier explorer are scripted."""
    furniture=SimpleNamespace(landmarks=landmarks,observe=lambda *args:None)
    policy=StationExplorer(ObservedNavigationMap(),furniture)
    policy.started=0.
    monkeypatch.setattr(SemanticNavigator,'step',lambda self,packet,tracking,now:(self.observe(packet,tracking),dict(status=statuses.pop(0),forward=0.,yaw=0.))[1])
    pose=np.eye(4);pose[:2,3]=xy
    return policy,furniture,SimpleNamespace(accepted=True,T_map_base=pose)


def test_counter_is_visited_as_soon_as_it_is_confirmed():
    furniture=SimpleNamespace(landmarks=[landmark('dining table',2),landmark('kitchen',1)],observe=lambda *args:None)
    policy=StationExplorer(ObservedNavigationMap(),furniture)
    policy.observe(None,None)
    assert not policy.targets.targets and policy.stage is None
    furniture.landmarks[1]['observations']=2
    policy.observe(None,None)
    target=policy.targets.targets[0]
    assert target.navigation_memory and target.label=='kitchen' and policy.target_label=='kitchen'
    policy.observe(None,None)
    assert policy.targets.targets==[target] and policy.targets.targets[0] is target
    assert policy.detection_calls==0 and policy.detector is None


def test_counter_does_not_wait_for_the_table():
    furniture=SimpleNamespace(landmarks=[landmark('kitchen',2)],observe=lambda *args:None)
    policy=StationExplorer(ObservedNavigationMap(),furniture)
    policy.observe(None,None)
    assert policy.stage=='kitchen' and policy.targets.targets[0].label=='kitchen'


def test_counter_standoff_is_scanned_then_the_table_ends_exploration(monkeypatch):
    statuses=['travel','at_target_standoff','travel','at_target_standoff']
    policy,furniture,tracking=explorer_at([2.,3.],[landmark('dining table',2),landmark('kitchen',2)],monkeypatch,statuses)
    scans=['scanning','scan_complete']
    def scan(packet,tracking,now):
        assert policy.explorer.phase=='scan'
        status=scans.pop(0)
        if status=='scan_complete':policy.explorer.phase='travel'
        return dict(status=status,forward=0.,yaw=.5)
    monkeypatch.setattr(policy.explorer,'step',scan)
    seen=[policy.step(None,tracking,t*.02)['status'] for t in range(6)]
    assert seen==['travel','kitchen_standoff_reached','scanning','scan_complete','travel','stations_explored']
    kitchen=furniture.landmarks[1]
    # The reached pose replaces the remote recognition viewpoint.
    assert kitchen['reached'] and kitchen['viewpoints']==[[2.,3.,0.]]
    assert policy.terminal=='stations_explored' and policy.targets.targets[0].label=='dining table'
    assert policy.target_label=='dining table' and not scans and not statuses
    assert [e['event'] for e in policy.follower.events]==['kitchen_standoff_reached','kitchen_scan_complete']


def test_scan_is_reused_when_exploration_already_scanned_at_the_standoff(monkeypatch):
    policy,furniture,tracking=explorer_at([2.,3.],[landmark('kitchen',2)],monkeypatch,['at_target_standoff','travel'])
    policy.explorer.scan_locations.append(np.array([2.2,3.1]))
    monkeypatch.setattr(policy.explorer,'step',lambda *args:(_ for _ in ()).throw(AssertionError('no second scan')))
    assert policy.step(None,tracking,0.)['status']=='kitchen_standoff_reached'
    assert policy.stage=='dining table' and policy.explorer.phase=='travel' and not policy.targets.targets
    # The table is still unconfirmed: frontier exploration goes on without a target.
    assert policy.step(None,tracking,.02)['status']=='travel' and policy.terminal is None
    furniture.landmarks.append(landmark('dining table',2))
    policy.observe(None,None)
    assert policy.targets.targets[0].label=='dining table'


def test_blocked_scan_does_not_end_exploration(monkeypatch):
    policy,furniture,tracking=explorer_at([2.,3.],[landmark('dining table',2),landmark('kitchen',2)],monkeypatch,['at_target_standoff','at_target_standoff'])
    def blocked(packet,tracking,now):
        policy.explorer.terminal='obstacle_near_scan_envelope'
        return dict(status=policy.explorer.terminal,forward=0.,yaw=0.)
    monkeypatch.setattr(policy.explorer,'step',blocked)
    seen=[policy.step(None,tracking,t*.02)['status'] for t in range(3)]
    assert seen==['kitchen_standoff_reached','kitchen_scan_abandoned','stations_explored']
    assert policy.explorer.terminal is None and furniture.landmarks[1]['reached']


def test_exploration_budget_still_ends_a_station_scan(monkeypatch):
    policy,furniture,tracking=explorer_at([2.,3.],[landmark('kitchen',2)],monkeypatch,['at_target_standoff'])
    policy.step(None,tracking,0.)
    def exhausted(packet,tracking,now):
        policy.explorer.terminal='budget_exhausted'
        return dict(status='budget_exhausted',forward=0.,yaw=0.)
    monkeypatch.setattr(policy.explorer,'step',exhausted)
    assert policy.step(None,tracking,.02)['status']=='budget_exhausted' and policy.terminal=='budget_exhausted'


def test_reached_counter_from_memory_is_not_revisited():
    furniture=SimpleNamespace(landmarks=[landmark('dining table',2),landmark('kitchen',2,reached=True)],observe=lambda *args:None)
    policy=StationExplorer(ObservedNavigationMap(),furniture)
    policy.observe(None,None)
    assert policy.stage=='dining table' and policy.targets.targets[0].label=='dining table'


def test_preset_table_targets_keep_the_single_stage_approach(monkeypatch):
    # The closed-door runner presets confirmed tables and disables observation.
    policy,furniture,tracking=explorer_at([0.,0.],[],monkeypatch,['at_target_standoff'])
    policy.observe=lambda packet,tracking:None
    assert policy.step(None,tracking,0.)['status']=='stations_explored'
