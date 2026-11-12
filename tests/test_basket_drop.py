import numpy as np
from cross_episode_sim.control.basket_drop import basket_drop_candidates


def geometry():
    x,y=np.meshgrid(np.linspace(.4,.7,31),np.linspace(-.15,.15,31))
    basket=np.c_[x.ravel(),y.ravel(),np.full(x.size,1.)]
    tcp=np.eye(4);tcp[:3,3]=[.3,0,1.4]
    held=np.array([[x,y,z] for x in (.27,.33) for y in (-.03,.03) for z in (1.27,1.4)])
    return basket,tcp,held


def test_release_above_existing_contents_without_empty_floor():
    basket,tcp,held=geometry()
    contents=basket.copy();contents[:,2]=1.1
    rows=basket_drop_candidates(basket,contents,held,tcp)
    assert rows
    for row in rows:
        pose=np.asarray(row['tcp'])
        released=(held-tcp[:3,3])@pose[:3,:3].T+pose[:3,3]
        assert released[:,2].min()>=1.135-1e-9
        assert released[:,0].min()>=.425-1e-8 and released[:,0].max()<=.675+1e-8
        assert released[:,1].min()>=-.125-1e-8 and released[:,1].max()<=.125+1e-8
        assert not row['empty_floor_required'] and not row['learned_proposal']


def test_no_release_outside_observed_opening_or_for_oversized_payload():
    basket,tcp,held=geometry();held=(held-tcp[:3,3])*8+tcp[:3,3]
    assert basket_drop_candidates(basket,basket,held,tcp)==[]
    assert basket_drop_candidates(basket[:2],basket,held,tcp)==[]


def test_overhead_cabinet_is_not_mistaken_for_basket_contents():
    basket,tcp,held=geometry()
    contents=basket.copy();contents[:,2]=1.025
    ceiling=basket.copy();ceiling[:,2]=1.6
    rows=basket_drop_candidates(basket,np.vstack([contents,ceiling]),held,tcp)
    assert rows and all(row['contact'][2]<1.1 for row in rows)
