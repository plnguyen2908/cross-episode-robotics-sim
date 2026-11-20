import numpy as np
import pytest
from types import SimpleNamespace
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.sensors.onboard import RGBDCamera,SensorPacket,JointObservation
class Geometry:
    tcp_body=1
    tcp_local=np.eye(4)
    def fk(self,q):
        T=np.eye(4);T[:3,3]=[q[0],0,1]
        return {1:T}
def test_payload_uses_robot_fk_and_keeps_scene_depth():
    rng=np.random.default_rng(1);p=rng.uniform(-.03,.03,(40,3));payload=ObservedPayload(Geometry(),p)
    q=np.zeros(7);b=payload.box(q)
    assert abs(b.center[2]-1)<.03
    depth=np.array([[1.,2.]],dtype=np.float32)
    c=RGBDCamera('cam',np.zeros((1,2,3),np.uint8),depth,np.ones((1,2),bool),np.array([[100.,0,0],[0,100.,0],[0,0,1]]),np.zeros(5),np.eye(4))
    packet=SensorPacket(0.,{'navigation':c},JointObservation(tuple(map(str,range(7))),q,q),SimpleNamespace())
    filtered=payload.filter_packet(packet)
    assert not filtered.cameras['navigation'].valid_depth[0,0]
    assert filtered.cameras['navigation'].valid_depth[0,1]
    assert packet.cameras['navigation'].valid_depth.all()
    q[0]=.2
    assert payload.box(q).center[0]==pytest.approx(b.center[0]+.2)
def test_payload_rejects_missing_geometry():
    with pytest.raises(ValueError):ObservedPayload(Geometry(),np.empty((0,3)))


def test_reconstruct_held_surface_without_detector_box():
    from types import SimpleNamespace as NS
    from cross_episode_sim.control.observed_payload import reconstruct_held_points
    geo=NS(fk=lambda q:{0:np.eye(4)},tcp_body=0,tcp_local=np.eye(4),boxes=lambda q:[])
    depth=np.full((60,60),.12);depth[:10]=1.
    camera=NS(depth_m=depth,valid_depth=np.ones_like(depth,bool),robot_mask=np.zeros_like(depth,bool),K=np.array([[100.,0.,30.],[0.,100.,30.],[0.,0.,1.]]),T_base_optical=np.eye(4))
    packet=NS(arm=NS(position=np.zeros(7)),cameras={'wrist':camera})
    points=reconstruct_held_points(packet,geo)
    assert len(points)>100
    assert np.allclose(points[:,2],.12)
    camera.robot_mask[:]=True
    import pytest
    with pytest.raises(ValueError,match='Insufficient'):reconstruct_held_points(packet,geo)


def test_loaded_docking_includes_payload_and_inherited_support():
    from cross_episode_sim.control.observed_payload import PayloadDocking
    captured={}
    def checker(grid,q,pose,support):
        captured['support']=support
        return 'payload-aware-checker'
    payload=SimpleNamespace(geometry=Geometry(),checker=checker)
    inherited=np.array([-.3,0.,0.])
    dock=PayloadDocking(SimpleNamespace(inherited_support_poses=[inherited]),payload)
    assert dock.make_checker(np.zeros(3),np.zeros(7))=='payload-aware-checker'
    assert any(np.array_equal(p,inherited) for p in captured['support'])
    assert any(np.array_equal(p,np.zeros(3)) for p in captured['support'])


def test_payload_clearance_rejects_obstacle_even_when_arm_is_clear():
    from cross_episode_sim.control.observed_payload import payload_clearance
    payload=ObservedPayload(Geometry(),np.random.default_rng(7).uniform(-.03,.03,(40,3)))
    q=np.zeros(7)
    assert not payload_clearance(payload,q,np.array([[0.,0.,1.08]]))
    assert payload_clearance(payload,q,np.array([[0.,0.,1.3]]))


def test_live_payload_refinement_retains_prior_and_rejects_other_surface(monkeypatch):
    from cross_episode_sim.control.observed_payload import refine_held_payload
    prior=np.random.default_rng(9).uniform(-.03,.03,(200,3))
    live=prior+np.array([.01,0.,0.])
    monkeypatch.setattr('cross_episode_sim.control.observed_payload.reconstruct_held_points',lambda *args:live)
    combined,info=refine_held_payload(None,None,prior)
    assert info['agreement']==1.
    assert combined[:,0].max()>prior[:,0].max()
    monkeypatch.setattr('cross_episode_sim.control.observed_payload.reconstruct_held_points',lambda *args:live+.4)
    with pytest.raises(ValueError,match='disagrees'):refine_held_payload(None,None,prior)


def test_refinement_accepts_bounded_matched_slip_without_shrinking(monkeypatch):
    from cross_episode_sim.control import observed_payload as module
    rng=np.random.default_rng(4)
    prior=rng.uniform(-.015,.015,(300,3))
    live=prior+np.array([0.,0.,.05])
    monkeypatch.setattr(module,'reconstruct_held_points',lambda *args:live)
    combined,info=module.refine_held_payload(None,None,prior)
    assert .04<np.linalg.norm(info['attachment_translation_m'])<.06
    assert combined[:,2].min()<=prior[:,2].min()+.005
    assert combined[:,2].max()>=live[:,2].max()-.005
    monkeypatch.setattr(module,'reconstruct_held_points',lambda *args:prior+[0,0,.1])
    import pytest
    with pytest.raises(ValueError,match='disagrees'):
        module.refine_held_payload(None,None,prior)


def turned_bowl(degrees):
    """Rim and wall samples of a 0.2 m bowl hanging beside the tool, turned about the tool x axis."""
    rng=np.random.default_rng(2);angle=rng.uniform(0,2*np.pi,900);height=rng.uniform(0,.07,900);radius=.06+.04*height/.07
    bowl=np.c_[radius*np.cos(angle),radius*np.sin(angle)-.12,height]
    c,s=np.cos(np.radians(degrees)),np.sin(np.radians(degrees))
    return bowl@np.array([[1,0,0],[0,c,-s],[0,s,c]]).T


def test_carried_estimate_is_widened_by_the_held_surface_seen_now(monkeypatch):
    from cross_episode_sim.control import observed_payload as module
    carried=turned_bowl(0);live=turned_bowl(10)
    monkeypatch.setattr(module,'reconstruct_held_points',lambda *args:live)
    # The carried box (2 cm + 1 cm) leaves part of the turned bowl in the scene.
    old=ObservedPayload(Geometry(),carried)
    outside=lambda payload:int((~np.all(np.abs(live-payload.center)<=payload.half+.01,axis=1)).sum())
    assert outside(old)>0
    points,detail=module.live_checked_payload(None,None,carried)
    assert detail['refined'] and max(detail['extent_growth_m'])>.01
    assert outside(ObservedPayload(Geometry(),points))==0
    # Nothing of the carried estimate is dropped.
    assert np.all(points.min(0)<=carried.min(0)+.005) and np.all(points.max(0)>=carried.max(0)-.005)


def test_unmatched_or_missing_live_surface_leaves_the_carried_estimate_unchanged(monkeypatch):
    from cross_episode_sim.control import observed_payload as module
    carried=turned_bowl(0)
    monkeypatch.setattr(module,'reconstruct_held_points',lambda *args:carried+[0,.3,0])
    points,detail=module.live_checked_payload(None,None,carried)
    assert not detail['refined'] and 'disagrees' in detail['reason'] and np.array_equal(points,carried)
    def nothing(*args):raise ValueError('Insufficient observed held-object surface')
    monkeypatch.setattr(module,'reconstruct_held_points',nothing)
    points,detail=module.live_checked_payload(None,None,carried)
    assert not detail['refined'] and np.array_equal(points,carried)
