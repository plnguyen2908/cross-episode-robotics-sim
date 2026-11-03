from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.navigation.local_scan import LocalScanSkill,ScanConfig


def sample(t=0.,yaw=0.,depth=2.):
    T=np.eye(4);T[:2,:2]=[[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=1.
    c=SimpleNamespace(depth_m=np.array([[depth]]),valid_depth=np.ones((1,1),bool),robot_mask=np.zeros((1,1),bool),K=np.eye(3),T_base_optical=mount)
    packet=SimpleNamespace(timestamp=t,cameras={'navigation':c},arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)))
    track=SimpleNamespace(timestamp=t,accepted=True,T_map_base=T)
    return packet,track


def test_scan_does_not_need_a_free_map_or_astar():
    skill=LocalScanSkill();p,t=sample();r=skill.step(p,t,0.)
    assert r.status=='scanning' and 0<r.yaw_rps<=.8*.02+1e-10
    for i in range(1,127):
        p,t=sample(i*.1,i*.05);r=skill.step(p,t,i*.1)
        if r.complete:break
    assert r.complete and r.observed_sectors==24 and r.yaw_rps==0


def test_no_completion_without_actual_estimated_turn():
    skill=LocalScanSkill(ScanConfig(max_seconds=1.))
    for i in range(11):
        p,t=sample(i*.1);r=skill.step(p,t,i*.1)
    assert r.status=='scan_budget_exhausted' and not r.complete


def test_guard_stops_for_external_obstacle_and_masks_self():
    p,t=sample(depth=.4);skill=LocalScanSkill();assert skill.step(p,t,0.).status=='obstacle_near_scan_envelope'
    p,t=sample();p.cameras['navigation'].robot_mask[:]=True
    assert LocalScanSkill().step(p,t,0.).status=='insufficient_depth'


def test_lost_stale_unsynchronized_and_arm_state():
    p,t=sample();t.accepted=False
    assert LocalScanSkill().step(p,t,0.).status=='tracking_lost'
    p,t=sample();assert LocalScanSkill().step(p,t,1.).status=='stale_observation'
    p,t=sample();t.timestamp=.1
    assert LocalScanSkill().step(p,t,.1).status=='unsynchronized_observation'
    p,t=sample();p.arm.position[0]=1
    assert LocalScanSkill().step(p,t,0.).status=='arm_not_stowed'


def test_pose_jump_does_not_complete_scan():
    skill=LocalScanSkill();p,t=sample();skill.step(p,t,0.)
    p,t=sample(.1,2.);r=skill.step(p,t,.1)
    assert r.status=='pose_jump' and not r.complete
    with pytest.raises(ValueError):skill.step(p,t,.1)
