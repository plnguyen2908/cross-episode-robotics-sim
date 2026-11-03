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
