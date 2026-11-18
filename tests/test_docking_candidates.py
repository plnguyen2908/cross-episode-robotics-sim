import numpy as np
import pytest
from cross_episode_sim.navigation.docking_candidates import household_surface_docks


@pytest.mark.parametrize('height,fridge,pickup,burner',[(.9,False,True,False),(.7,False,False,False),(.9,True,True,False),(.9,True,False,False),(.8,False,True,True)])
@pytest.mark.parametrize('outward',[(-1.,0.),(1.,0.),(0.,-1.),(0.,1.)])
def test_extraction_preserves_original_household_oracle_order(height,fridge,pickup,burner,outward):
    point=np.array([1.,2.,height+.08]);lo=np.array([.5,1.,height]);hi=np.array([1.5,3.,height]);outward=np.array(outward)
    distances=(((.50,.43,.35,.55) if pickup else (.43,.50,.35,.55)) if fridge else ((.26,.30,.35,.43) if height>=.88 or burner else (.35,.43,.50,.28)))
    if burner:distances=(*distances,.50,.60,.70)
    laterals=(0.,-.10,.10,-.20,-.30,-.40) if fridge else (0.,-.10,.10)
    expected=[];axis=int(np.argmax(np.abs(outward)))
    for distance in distances:
        for lateral in laterals:
            xy=point[:2].copy();xy[axis]=(hi if outward[axis]>0 else lo)[axis]+outward[axis]*distance;xy[1-axis]+=lateral
            expected.append([*xy,np.arctan2(point[1]-xy[1],point[0]-xy[0])])
    assert np.allclose(list(household_surface_docks(point,lo,hi,outward,fridge=fridge,pickup=pickup,burner=burner)),expected)


def test_frontal_pickup_checks_all_centered_standoffs_before_lateral():
    from cross_episode_sim.navigation.docking_candidates import frontal_surface_docks
    p=np.array([2.,.3,1.]);lo=np.array([1.8,-1.,.9]);hi=np.array([2.4,1.,.9]);out=np.array([-1.,0.])
    candidates=list(frontal_surface_docks(p,lo,hi,out))
    assert all(abs(c[1]-p[1])<1e-12 and abs(c[2])<1e-12 for c in candidates[:6])
    assert np.allclose([c[0] for c in candidates[:6]],lo[0]-np.array([.26,.30,.35,.43,.50,.55]))
    assert len(candidates)==7 and abs(candidates[6][1]-p[1])<1e-12


def test_standoff_docks_face_the_target_along_the_surface_normal():
    from cross_episode_sim.navigation.docking_candidates import standoff_surface_docks
    p=np.array([2.,.3,1.]);out=np.array([-3.,4.])
    docks=list(standoff_surface_docks(p,out,(.70,.75)))
    assert np.allclose([np.linalg.norm(c[:2]-p[:2]) for c in docks],[.70,.75])
    for c in docks:assert np.allclose(c[:2]+np.linalg.norm(c[:2]-p[:2])*np.array([np.cos(c[2]),np.sin(c[2])]),p[:2])


@pytest.mark.parametrize('angle',[0.,.3,-.7,1.4])
def test_observed_surface_frame_is_rotation_equivariant(angle):
    from cross_episode_sim.navigation.docking_candidates import fit_surface_frame,frontal_surface_docks
    x,y=np.meshgrid(np.linspace(1.8,2.4,100),np.linspace(-.4,1.,100));top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.9)]
    y,z=np.meshgrid(np.linspace(-.4,1.,100),np.linspace(.32,.8,80));face=np.c_[np.full(y.size,1.8),y.ravel(),z.ravel()]
    pts=np.concatenate([top,face]);target=np.array([2.,.3,.98]);base=np.array([.8,.3])
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);shift=np.array([3.,-2.])
    pts[:,:2]=pts[:,:2]@R.T+shift;target[:2]=R@target[:2]+shift;base=R@base+shift
    lo,hi,out=fit_surface_frame(pts,target,base)
    assert np.allclose(out,R@np.array([-1.,0.]),atol=.001)
    for candidate in frontal_surface_docks(target,lo,hi,out):
        delta=candidate[:2]-target[:2]
        assert abs(delta@np.array([out[1],-out[0]]))<1e-8


def legged_table(camera,angle=0.,shift=(0.,0.),floor=True):
    """Points one camera would measure on a top spanning x 1.8-2.6, y -.4-1. with nothing under its edges."""
    x,y=np.meshgrid(np.arange(1.805,2.6,.01),np.arange(-.395,1.,.01));top=np.c_[x.ravel(),y.ravel(),np.full(x.size,.75)]
    x,y=np.meshgrid(np.arange(.5,3.5,.01),np.arange(-1.5,2.,.01));ground=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    # Keep only floor the top does not hide from this camera.
    crossing=camera[:2]+(ground[:,:2]-camera[:2])*(camera[2]-.75)/camera[2]
    hidden=(crossing[:,0]>1.8)&(crossing[:,0]<2.6)&(crossing[:,1]>-.4)&(crossing[:,1]<1.)
    points=np.concatenate([top,ground[~hidden]]) if floor else top
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);origin=camera.copy()
    points[:,:2]=points[:,:2]@R.T+shift;origin[:2]=R@origin[:2]+shift
    return origin,points,R


@pytest.mark.parametrize('angle',[0.,.3,-.7,1.4])
def test_open_edge_fits_a_table_without_a_face(angle):
    from cross_episode_sim.navigation.docking_candidates import fit_open_edge,fit_surface_frame
    shift=np.array([3.,-2.]);origin,points,R=legged_table(np.array([.9,.3,1.3]),angle,shift)
    target=np.r_[R@np.array([2.,.3])+shift,.83]
    with pytest.raises(ValueError):fit_surface_frame(points,target,origin[:2])
    lo,hi,out=fit_open_edge([(origin,points)],target,origin[:2])
    assert np.allclose(out,R@np.array([-1.,0.]),atol=.02)
    assert np.allclose(lo[:2],R@np.array([1.8,.3])+shift,atol=.015) and abs(lo[2]-.75)<.003 and np.array_equal(lo,hi)


def test_open_edge_selects_the_edge_nearest_the_target():
    from cross_episode_sim.navigation.docking_candidates import fit_open_edge
    # From the corner both the x and the y edge are measured beside the target.
    origin,points,_=legged_table(np.array([1.1,-1.1,1.3]))
    _,_,out=fit_open_edge([(origin,points)],np.array([2.2,-.2,.83]),origin[:2])
    assert np.allclose(out,[0.,-1.],atol=.02)
    _,_,out=fit_open_edge([(origin,points)],np.array([2.,.2,.83]),origin[:2])
    assert np.allclose(out,[-1.,0.],atol=.02)


def test_open_edge_requires_sight_lines_past_the_top():
    from cross_episode_sim.navigation.docking_candidates import fit_open_edge
    # A top that ends at the image border has no measured edge.
    origin,points,_=legged_table(np.array([.9,.3,1.3]),floor=False)
    with pytest.raises(ValueError,match='Support edge not observed'):fit_open_edge([(origin,points)],np.array([2.,.3,.83]),origin[:2])
    # A camera under the top sees the floor without looking past an edge.
    origin,points,_=legged_table(np.array([.9,.3,1.3]));origin[2]=.6
    with pytest.raises(ValueError,match='Support edge not observed'):fit_open_edge([(origin,points)],np.array([2.,.3,.83]),origin[:2])


def test_observed_surface_uses_the_open_edge_only_when_no_face_fits(monkeypatch):
    from types import SimpleNamespace
    from cross_episode_sim.navigation import docking_candidates as module
    K=np.array([[100.,0.,1.],[0.,100.,1.],[0.,0.,1.]]);mount=np.eye(4);mount[:3,3]=[.1,0.,1.2]
    camera=SimpleNamespace(valid_depth=np.ones((2,2),bool),robot_mask=np.array([[True,False],[False,False]]),depth_m=np.full((2,2),2.),K=K,T_base_optical=mount)
    packet=SimpleNamespace(cameras=dict(navigation=camera));T=np.eye(4);T[:2,3]=[3.,4.];target=np.zeros(3);calls=[]
    def face(points,target,base):calls.append(('face',len(points)));return 'face'
    def edge(clouds,target,base):calls.append(('edge',clouds[0][0].tolist(),len(clouds[0][1]),base.tolist()));return 'edge'
    monkeypatch.setattr(module,'fit_surface_frame',face);monkeypatch.setattr(module,'fit_open_edge',edge)
    assert module.observed_surface(packet,T,target)=='face' and calls==[('face',3)]
    def no_face(points,target,base):raise ValueError('Counter orientation unverified')
    monkeypatch.setattr(module,'fit_surface_frame',no_face)
    assert module.observed_surface(packet,T,target)=='edge' and calls[-1]==('edge',[3.1,4.,1.2],3,[3.,4.])
    def no_edge(clouds,target,base):raise ValueError('Support edge not observed')
    monkeypatch.setattr(module,'fit_open_edge',no_edge)
    with pytest.raises(ValueError,match='Counter orientation unverified; Support edge not observed'):module.observed_surface(packet,T,target)
