"""A loaded turn in place that would swing the held object into mapped points is replaced by a back-off."""
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.navigation.carry_departure import (LoadedTurnGuard,RecordedThenCheckedDeparture,loaded_turn_blocked,
    straight_back_clear,LOADED_TURN_BACK_LIMIT,LOADED_TURN_BACK_SPEED)


class DoorOnTheLeft:
    """Contacts as measured on fix144's saved map: growing with a left turn, gone 0.2 m further back."""
    def __init__(self,start=(0.,0.,0.)):self.start=np.asarray(start,float)
    def contact_count(self,pose):
        back=-(np.asarray(pose[:2])-self.start[:2])@np.array([np.cos(self.start[2]),np.sin(self.start[2])])
        if back>=.2:return 0
        turn=np.degrees(pose[2]-self.start[2])
        return int(19+4*turn) if 0<=turn<=25 else (19 if turn==0 else 0)


def test_turn_toward_the_door_is_blocked_and_the_other_way_is_not():
    checker=DoorOnTheLeft()
    assert loaded_turn_blocked(checker,[0.,0.,0.],1.)
    assert not loaded_turn_blocked(checker,[0.,0.,0.],-1.)
    assert straight_back_clear(checker,[0.,0.,0.])
    assert not loaded_turn_blocked(checker,[-.25,0.,0.],1.)        # after backing off


def test_guard_backs_off_until_the_turn_is_clear_then_lets_it_through():
    guard=LoadedTurnGuard();checker=DoorOnTheLeft();pose=np.zeros(3);stamp=0;turns=0
    for tick in range(600):
        if tick%5==0:stamp+=1
        command=guard.command(0.,.4,pose,stamp,lambda:checker)
        if command is None:turns+=1;break
        assert -LOADED_TURN_BACK_SPEED-1e-9<=command<0
        pose[0]+=command*.02
    assert turns==1 and .2<=-pose[0]<.35 and guard.announced


def test_guard_leaves_driving_and_clear_turns_alone():
    guard=LoadedTurnGuard();calls=[]
    def checker():calls.append(1);return DoorOnTheLeft()
    assert guard.command(.1,.4,np.zeros(3),1,checker) is None and not calls          # driving, not a turn in place
    assert guard.command(0.,-.4,np.zeros(3),2,checker) is None and len(calls)==1     # right turn is clear
    assert guard.command(0.,-.4,np.zeros(3),2,checker) is None and len(calls)==1     # one decision per packet


def test_guard_gives_up_after_its_limit_and_when_backing_is_not_clear():
    class Everywhere:
        def contact_count(self,pose):return int(10+np.degrees(abs(pose[2])))
    guard=LoadedTurnGuard();pose=np.zeros(3);stamp=0;backed=0.
    for tick in range(2000):
        if tick%5==0:stamp+=1
        command=guard.command(0.,.4,pose,stamp,Everywhere)
        if command is None:break
        backed-=command*.02
    assert LOADED_TURN_BACK_LIMIT-.01<=backed<=LOADED_TURN_BACK_LIMIT+.01
    class Walled:
        def contact_count(self,pose):return int(10+np.degrees(abs(pose[2]))+1000*max(0.,-pose[0]))
    assert LoadedTurnGuard().command(0.,.4,np.zeros(3),1,Walled) is None


def test_departure_applies_the_guard_to_its_command():
    class Turning:
        error=None;follower=None
        def __init__(self):self.docking=SimpleNamespace(events=[],checker=None);self.velocity=np.array([0.,.4])
        def step(self,packet,tracking,free,now):return dict(status='reverse_recorded_approach',forward=0.,yaw=.4)
    payload=SimpleNamespace(checker=lambda grid,q,pose,support:DoorOnTheLeft())
    departure=RecordedThenCheckedDeparture(Turning(),lambda:None,payload=payload,grid=SimpleNamespace())
    packet=SimpleNamespace(arm=SimpleNamespace(position=np.zeros(7)),timestamp=1.)
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    out=departure.step(packet,tracking,None,0.)
    assert out['yaw']==0. and out['forward']<0 and out['status']=='reverse_recorded_approach'
    assert departure.docking.events[-1]['event']=='loaded_turn_blocked_back_off' and departure.docking.events[-1]['turn']=='left'
    assert not departure.active.velocity.any()


def test_guard_holds_the_back_off_while_the_follower_restarts_its_turn_from_rest():
    # As in the departure: the follower's speed is zeroed during the back-off, so every tick it
    # asks again for the first small step of the same turn (0.016 rad/s). The base must never turn.
    guard=LoadedTurnGuard();checker=DoorOnTheLeft();pose=np.zeros(3);stamp=0;turned=0.
    for tick in range(600):
        if tick%5==0:stamp+=1
        command=guard.command(0.,.016,pose,stamp,lambda:checker)
        if command is None:
            if not loaded_turn_blocked(checker,pose,1.):break
            turned+=.016*.02;pose[2]+=.016*.02
        else:pose[0]+=command*.02
    assert turned==0. and -pose[0]>=.2
