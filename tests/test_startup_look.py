from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.navigation.local_scan import ScanConfig
from cross_episode_sim.navigation.startup_look import StartupLook,StartupLookConfig

LOST=dict(status='tracking_lost',forward=0.,yaw=0.,scans=0,plans=0)


def sample(t=0.,accepted=False,depth=2.):
    mount=np.eye(4);mount[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];mount[2,3]=1.
    c=SimpleNamespace(depth_m=np.array([[depth]]),valid_depth=np.ones((1,1),bool),robot_mask=np.zeros((1,1),bool),K=np.eye(3),T_base_optical=mount)
    packet=SimpleNamespace(timestamp=t,cameras={'navigation':c},arm=SimpleNamespace(position=np.array(ScanConfig().stowed_arm)))
    track=SimpleNamespace(timestamp=t,accepted=accepted,status='tracking' if accepted else 'insufficient features',T_map_base=np.eye(4) if accepted else None)
    return packet,track


def test_turns_until_the_first_accepted_frame_then_never_again():
    look=StartupLook();p,t=sample();r=look.step(LOST,p,t,0.)
    assert r['status']=='startup_look' and r['forward']==0 and 0<r['yaw']<=.8*.02+1e-10 and r['scans']==0
    for i in range(1,100):
        p,t=sample(i*.02);r=look.step(LOST,p,t,i*.02)
    assert r['yaw']==pytest.approx(.3) and look.commanded_angle>.5
    scanning=dict(LOST,status='scanning',yaw=.01)
    p,t=sample(2.,accepted=True);assert look.step(scanning,p,t,2.) is scanning
    # A later loss first waits for the tracker's own recovery.
    p,t=sample(2.02);assert look.step(LOST,p,t,2.02) is LOST
    assert look.summary()['used'] and look.summary()['tracked'] and look.summary()['terminal'] is None


def test_does_nothing_when_the_first_frame_tracks():
    look=StartupLook();command=dict(LOST,status='scanning');p,t=sample(accepted=True)
    assert look.step(command,p,t,0.) is command and not look.summary()['used']


def test_navigator_budget_and_terminal_states_stay_authoritative():
    look=StartupLook();p,t=sample();spent=dict(LOST,status='budget_exhausted')
    assert look.step(spent,p,t,0.) is spent and look.commanded_angle==0


def test_refuses_to_turn_beside_an_obstacle_or_without_depth():
    p,t=sample(depth=.4);look=StartupLook();r=look.step(LOST,p,t,0.)
    assert r['status']=='startup_look_blocked' and r['yaw']==0 and look.events[-1]['reason']=='obstacle_near_scan_envelope'
    p,t=sample(.02);assert look.step(LOST,p,t,.02)['status']=='startup_look_blocked'
    p,t=sample();p.cameras['navigation'].robot_mask[:]=True;look=StartupLook()
    assert look.step(LOST,p,t,0.)['status']=='startup_look_blocked' and look.events[-1]['reason']=='insufficient_depth'


def test_requires_fresh_sensing_and_stowed_arm():
    p,t=sample();assert StartupLook().step(LOST,p,t,.3)['status']=='stale_observation'
    p,t=sample();p.arm.position=p.arm.position+.2
    assert StartupLook().step(LOST,p,t,0.)['status']=='arm_not_stowed'


def test_stops_after_one_commanded_turn():
    look=StartupLook(StartupLookConfig(max_angle=1.))
    for i in range(400):
        p,t=sample(i*.02);r=look.step(LOST,p,t,i*.02)
        if r['status']!='startup_look':break
    assert r['status']=='startup_look_exhausted' and r['yaw']==0 and look.commanded_angle==pytest.approx(1.,abs=.01)
    assert look.summary()['terminal']=='startup_look_exhausted' and not look.summary()['tracked']


def lose(look,start,turn=.5,steps=400):
    """Track while turning one way, then lose tracking; return the commands issued while lost."""
    p,t=sample(start,accepted=True);look.step(dict(LOST,status='travel',yaw=turn),p,t,start)
    out=[]
    for i in range(1,steps):
        now=start+i*.02;p,t=sample(now);out.append(look.step(LOST,p,t,now))
    return out


def test_later_loss_turns_back_the_way_the_base_came():
    look=StartupLook();out=lose(look,0.,turn=.5)
    waited=[r for r in out if r is LOST]
    assert len(waited)==pytest.approx(50,abs=1)  # the tracker's own recovery comes first
    turning=[r for r in out if r['status']=='recovery_look']
    assert turning and all(r['forward']==0 and -.3-1e-9<=r['yaw']<0 for r in turning)
    assert look.events[0]['event']=='recovery_look' and look.events[0]['direction']==-1
    look=StartupLook();assert all(r['yaw']>0 for r in lose(look,0.,turn=-.5) if r['status']=='recovery_look')


def test_recovered_view_returns_control_and_each_loss_gets_a_fresh_turn():
    look=StartupLook(StartupLookConfig(max_angle=1.));lose(look,0.,steps=150)
    travel=dict(LOST,status='travel',forward=.2,yaw=.5)
    p,t=sample(3.,accepted=True);assert look.step(travel,p,t,3.) is travel
    assert look.events[-1]['event']=='view_recovered' and look.events[-1]['commanded_angle']<0
    out=lose(look,3.,steps=150)
    assert look.recoveries==2 and out[-1]['status']=='recovery_look' and 0<look.commanded_angle<1.


def test_recovery_is_guarded_and_bounded():
    look=StartupLook(StartupLookConfig(max_angle=1.));out=lose(look,0.)
    assert out[-1]['status']=='recovery_look_exhausted' and out[-1]['yaw']==0 and look.summary()['terminal']=='recovery_look_exhausted'
    look=StartupLook();p,t=sample(0.,accepted=True);look.step(dict(LOST,status='travel'),p,t,0.)
    for i in range(1,80):
        p,t=sample(i*.02,depth=.4);r=look.step(LOST,p,t,i*.02)
    assert r['status']=='recovery_look_blocked' and look.commanded_angle==0
    look=StartupLook(StartupLookConfig(max_recoveries=1));lose(look,0.,steps=100)
    p,t=sample(2.,accepted=True);look.step(dict(LOST,status='travel'),p,t,2.)
    assert lose(look,2.,steps=100)[-1]['status']=='recovery_look_exhausted' and look.recoveries==1


def test_resumed_map_recovers_but_never_starts_a_first_look():
    look=StartupLook(tracked=True);p,t=sample();assert look.step(LOST,p,t,0.) is LOST
    for i in range(1,80):
        p,t=sample(i*.02);r=look.step(LOST,p,t,i*.02)
    assert r['status']=='recovery_look' and look.summary()['recoveries']==1
    p,t=sample(2.);p.arm.position=p.arm.position+.2
    assert look.step(LOST,p,t,2.)['status']=='arm_not_stowed'
