"""The carry departure falls back to a checked route when the approach cannot be retraced."""
from types import SimpleNamespace
from cross_episode_sim.navigation.carry_departure import RecordedThenCheckedDeparture,make_carry_departure


class Stub:
    def __init__(self,error=None,status='moving'):
        self.error=error;self.follower=None;self.docking=SimpleNamespace(events=[dict(event='own')],checker=None);self.status=status;self.steps=0
    def step(self,packet,tracking,free,now):
        self.steps+=1;return dict(status=self.error or self.status,forward=0.,yaw=0.)


def test_unavailable_recorded_departure_switches_to_the_checked_one():
    recorded=Stub(error='recorded_departure_unavailable');checked=Stub(status='observed_undock_to_transit')
    departure=RecordedThenCheckedDeparture(recorded,lambda:checked)
    out=departure.step(None,None,None,0.)
    assert out['status']=='observed_undock_to_transit' and departure.error is None and checked.steps==1
    assert [e['event'] for e in departure.docking.events][-1]=='recorded_departure_unavailable'
    departure.step(None,None,None,1.);assert recorded.steps==1 and checked.steps==2       # stays on the checked one


def test_working_recorded_departure_is_left_alone():
    recorded=Stub(status='reverse_recorded_approach');made=[]
    departure=RecordedThenCheckedDeparture(recorded,lambda:made.append(1) or Stub())
    assert departure.step(None,None,None,0.)['status']=='reverse_recorded_approach' and not made


def test_other_errors_are_not_masked():
    recorded=Stub(error='tracking_lost');made=[]
    departure=RecordedThenCheckedDeparture(recorded,lambda:made.append(1) or Stub())
    departure.step(None,None,None,0.);assert departure.error=='tracking_lost' and not made


def test_known_map_carry_gets_the_fallback_only_with_geometry_and_a_frozen_map():
    frozen=SimpleNamespace(fixed_obstacle_points=[]);live=SimpleNamespace()
    assert isinstance(make_carry_departure(frozen,SimpleNamespace(geometry=object()),known_map_only=True),RecordedThenCheckedDeparture)
    assert not isinstance(make_carry_departure(live,SimpleNamespace(geometry=object()),known_map_only=True),RecordedThenCheckedDeparture)
    assert not isinstance(make_carry_departure(frozen,SimpleNamespace(),known_map_only=True),RecordedThenCheckedDeparture)
