from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.navigation import edge_first as module
from cross_episode_sim.navigation.edge_first import edge_first


def track(x,y,z=1.,track_id=0):return NS(position=np.array([x,y,z]),track_id=track_id)

def depths(table):return lambda packet,T,position:table.get((round(float(position[0]),2),round(float(position[1]),2)))

def test_instance_nearer_the_front_edge_replaces_the_one_behind_it():
    rear=track(2.3,0.,track_id=0);front=track(2.05,.15,track_id=1)
    chosen,measured=edge_first(rear,[rear,front],None,None,depth=depths({(2.3,0.):.32,(2.05,.15):.05}))
    assert chosen is front
    assert [row['edge_depth_m'] for row in measured]==[.32,.05] and measured[1]['track_id']==1
    # Order of sighting does not matter: the front one stays chosen.
    assert edge_first(front,[rear],None,None,depth=depths({(2.3,0.):.32,(2.05,.15):.05}))[0] is front

def test_noise_sized_difference_or_missing_evidence_keeps_the_current_instance():
    current=track(2.1,0.);other=track(2.1,.2,track_id=1)
    assert edge_first(current,[other],None,None,depth=depths({(2.1,0.):.10,(2.1,.2):.06}))[0] is current
    assert edge_first(current,[other],None,None,depth=depths({(2.1,0.):.10}))[0] is current
    assert edge_first(current,[other],None,None,depth=depths({}))[0] is current
    # The current instance has no verified edge beside it; the other one does.
    assert edge_first(current,[other],None,None,depth=depths({(2.1,.2):.20}))[0] is other

def test_instances_elsewhere_are_not_compared():
    def forbidden(*args):raise AssertionError('Measured an instance outside the group')
    current=track(2.1,0.)
    far=track(2.1,.9,track_id=1);other_surface=track(2.1,.2,z=.75,track_id=2)
    assert edge_first(current,[current,far,other_surface],None,None,depth=forbidden)==(current,[])

def test_edge_depth_is_measured_from_the_observed_counter_front(monkeypatch):
    x,y=np.meshgrid(np.linspace(1.8,2.4,100),np.linspace(-.4,1.,100));top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.9)]
    y,z=np.meshgrid(np.linspace(-.4,1.,100),np.linspace(.32,.8,80));face=np.c_[np.full(y.size,1.8),y.ravel(),z.ravel()]
    cloud=np.concatenate([top,face])
    from cross_episode_sim.navigation import docking_candidates
    def observed(packet,T,target):return docking_candidates.fit_surface_frame(cloud,target,T[:2,3])
    monkeypatch.setattr(module,'observed_surface',observed)
    T=np.eye(4);T[:2,3]=(.8,.3)
    assert abs(module.edge_depth(None,T,[1.86,.2,.98])-.06)<.01
    assert abs(module.edge_depth(None,T,[2.12,.35,.98])-.32)<.01
    def unverified(*args):raise ValueError('Support edge not observed')
    monkeypatch.setattr(module,'observed_surface',unverified)
    assert module.edge_depth(None,T,[1.86,.2,.98]) is None
