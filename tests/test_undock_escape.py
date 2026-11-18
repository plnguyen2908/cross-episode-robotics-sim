"""Undock from a dock that starts inside the safety margin of an observed obstacle."""
import numpy as np
from types import SimpleNamespace
from cross_episode_sim.navigation.observed_undock import ObservedUndock


class Checker:
    """Obstacle line at x=+0.30 (just-opened door beside the base); contact grows as x grows."""
    detail={'reason':'observed_3d_collision'}
    def clear(self,p):return p[0]<-.05
    def contact_count(self,p):return max(0,int(round((p[0]+.051)*1000)))
    def swept(self,a,b):return True


def undock(direction,replans=2):
    grid=SimpleNamespace(config=SimpleNamespace(resolution=.05),world_to_cell=lambda xy:np.zeros((len(xy),2),int))
    dock=SimpleNamespace(events=[],checker=Checker(),make_checker=lambda pose,q:Checker(),grid=grid)
    u=ObservedUndock(dock);u.departure_start=np.zeros(2);u.started=0.;u.budget=30.;u.goal_region=np.zeros((1,1),bool)
    u.follower=SimpleNamespace(command=lambda pose:(direction*.2,0.),complete=False);u.replans=replans
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=None))
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    return u.step(packet,tracking,np.zeros((1,1),bool),1.),dock.events


def test_step_away_from_the_obstacle_is_allowed_while_in_contact():
    out,events=undock(-1.)
    assert out['status']=='observed_undock_to_transit' and out['forward']<0
    assert any(e['event']=='undock_escape_from_contact' for e in events)


def test_step_deeper_into_the_obstacle_is_still_stopped():
    out,events=undock(+1.)
    assert out['status']=='undock_clearance_stop'


def test_a_clearance_stop_first_replans_from_where_the_robot_stands():
    out,events=undock(+1.,replans=0)
    assert out['status']=='observed_undock_to_transit' and out['forward']==0.
    assert any(e['event']=='undock_replanned' for e in events)


def test_departure_is_planned_with_extra_clearance_first():
    from cross_episode_sim.navigation import observed_undock
    assert observed_undock.PLANNING_CLEARANCES[0]>0 and observed_undock.PLANNING_CLEARANCES[-1]==0.
    margins=[]
    class Dock:
        events=[]
        def make_checker(self,pose,q):return SimpleNamespace(margin=.035)
    dock=Dock();u=ObservedUndock(dock)
    def fake_plan(d,pose,packet,free,extra):
        margins.append(extra);return extra<=.02
    u._plan_departure=fake_plan
    for extra in observed_undock.PLANNING_CLEARANCES:
        if u._plan_departure(dock,None,None,None,extra):break
    assert margins==[.04,.02]


class MarginChecker:
    """Wall 0.40 m beside the base; the base needs half-width 0.33 m plus the margin."""
    detail={'reason':'observed_3d_collision'}
    def __init__(self):self.margin=.08
    def clear(self,p):return .33+self.margin<.40
    def contact_count(self,p):return 0 if self.clear(p) else 8
    def swept(self,a,b):return self.clear(a)


def plan_from_standing_contact(wall_margin):
    from cross_episode_sim.navigation import observed_undock
    made=[]
    class Dock:
        def __init__(self):
            self.events=[];self.path=None;self.goal=None
            self.grid=SimpleNamespace(config=SimpleNamespace(resolution=.05),
                world_to_cell=lambda xy:np.floor((np.atleast_2d(xy)+2.)/.05).astype(int),
                cell_to_world=lambda cells:(np.asarray(cells)+.5)*.05-2.)
        def make_checker(self,pose,q):
            c=MarginChecker();c.margin=wall_margin;made.append(c);return c
    dock=Dock();u=ObservedUndock(dock,planner=lambda *a:False)
    free=np.ones((80,80),bool)
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=None))
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    return u,u.step(packet,tracking,free,1.),dock.events


def test_standing_pose_inside_the_margin_plans_with_a_reduced_margin():
    # Nominal .08 puts the wall .01 inside the margin; -.015 clears it.
    u,out,events=plan_from_standing_contact(.08)
    planned=[e for e in events if e['event']=='undock_route_planned']
    assert out['status']=='observed_undock_to_transit' and u.error is None
    assert planned and planned[-1]['planning_clearance_m']==-.015 and u.margin_reduction==-.015
    assert any(e['event']=='undock_standing_pose_check' and e['inside_nominal_margin'] for e in events)


def test_a_real_obstruction_is_not_planned_through():
    # Wall .06 inside the margin: no allowed reduction clears it.
    u,out,events=plan_from_standing_contact(.13)
    assert out['status']=='undock_route_unverified' and u.error=='undock_route_unverified'
    assert not [e for e in events if e['event']=='undock_route_planned']


def test_a_shared_checker_gets_each_planning_clearance_from_its_nominal_margin():
    """The transit stage hands out one checker object; extras must not pile up."""
    from cross_episode_sim.navigation import observed_undock
    shared=MarginChecker();shared.margin=.035;seen=[]
    class Dock:
        def __init__(self):
            self.events=[];self.path=None;self.goal=None
            self.grid=SimpleNamespace(config=SimpleNamespace(resolution=.05),
                world_to_cell=lambda xy:np.floor((np.atleast_2d(xy)+2.)/.05).astype(int),
                cell_to_world=lambda cells:(np.asarray(cells)+.5)*.05-2.)
        def make_checker(self,pose,q):return shared
    shared.clear=lambda p:(seen.append(round(shared.margin,3)),False)[1]
    shared.swept=lambda a,b:False
    dock=Dock();u=ObservedUndock(dock,planner=lambda *a:False)
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=None))
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    u.step(packet,tracking,np.ones((80,80),bool),1.)
    assert sorted(set(seen),reverse=True)==[.075,.055,.035,.02,.005]
    assert shared.margin==.035


class Corner:
    """Counter front ahead-right of the base: inside the margin while x > -0.08; contact grows with x."""
    margin=.03;nominal_margin=.03;detail={'reason':'observed_3d_collision','part':'robot_0/base/collision'}
    def clear(self,p):return p[0]<=-.08
    def contact_count(self,p):return max(0,int(round((p[0]+.08)*1000)))
    def swept(self,a,b):return self.clear(a) and self.clear(b)


def test_departure_with_no_route_backs_straight_out_of_the_margin_first():
    """fix124: every departure plan was rejected from a pose 4 cm from a cabinet front."""
    grid=SimpleNamespace(config=SimpleNamespace(resolution=.05),world_to_cell=lambda xy:np.zeros((len(xy),2),int),
        cell_to_world=lambda cells:np.zeros((len(cells),2)))
    dock=SimpleNamespace(events=[],checker=Corner(),make_checker=lambda pose,q:Corner(),grid=grid,path=None,goal=None)
    u=ObservedUndock(dock,planner=lambda *a:False)
    packet=SimpleNamespace(timestamp=1.,arm=SimpleNamespace(position=None))
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    out=u.step(packet,tracking,np.zeros((1,1),bool),1.)
    events=[e['event'] for e in dock.events]
    assert 'undock_straight_escape' in events and u.error is None
    assert out['status']=='observed_undock_to_transit' and out['forward']<0      # reversing, away from the counter
    assert dock.goal[0]<=-.08 and abs(dock.goal[1])<1e-9


def test_straight_escape_function():
    from cross_episode_sim.navigation.observed_undock import straight_escape
    assert straight_escape(Corner(),np.array([-.2,0.,0.])) is None               # already clear
    goal=straight_escape(Corner(),np.zeros(3));assert goal is not None and -.12<=goal[0]<=-.08
