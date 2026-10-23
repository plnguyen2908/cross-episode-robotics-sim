import numpy as np
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union
from cross_episode_sim.generation.growth_layout import sample_growth_layout, grow


def test_growth_layout_contract_and_geometry():
    for seed in (0,1,2):
        layout=sample_growth_layout(seed,[0,-4.8,5.5,0],6)
        assert layout==sample_growth_layout(seed,[0,-4.8,5.5,0],6)
        rooms=layout['rooms'];kinds=[r['room_type'] for r in rooms]
        assert kinds.count('kitchen')==kinds.count('dining')==1
        assert kinds.count('bedroom')>=1
        polys={r['name']:Polygon(r['floor_polygon']) for r in rooms}
        assert polys['kitchen'].equals(box(0,-4.8,5.5,0))
        assert unary_union(list(polys.values())).geom_type=='Polygon'
        for i,p in enumerate(polys.values()):
            assert p.is_valid and not p.interiors
            for q in list(polys.values())[i+1:]:assert p.intersection(q).area<1e-7
        for r in rooms:
            assert unary_union([box(*b) for b in r['floor_rectangles']]).equals(polys[r['name']])
        for e in layout['connections']:
            axis=int(e['wall'][0]);a=e['center'].copy();b=a.copy();a[1-axis]-=.95;b[1-axis]+=.95
            assert all(polys[n].boundary.buffer(1e-6).covers(LineString([a,b])) for n in e['rooms'])


def test_growth_fills_mask_without_outside_cells():
    mask=np.ones((12,14),bool);mask[:3,:4]=False
    result=grow(mask,[2,1,1],np.random.default_rng(4))
    assert (result[mask]>0).all()
    assert (result[~mask]==-1).all()
