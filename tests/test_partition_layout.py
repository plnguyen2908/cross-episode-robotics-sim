import pytest
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union
from cross_episode_sim.generation.partition_layout import sample_partition_layout


def test_partition_geometry_and_doors():
    accepted=[]
    for seed in range(150):
        try:layout=sample_partition_layout(seed,[0.,-4.8,5.5,0.],7)
        except ValueError:continue
        accepted.append(layout)
        assert layout==sample_partition_layout(seed,[0.,-4.8,5.5,0.],7)
        rooms=layout['rooms'];polys={r['name']:Polygon(r['floor_polygon']) for r in rooms}
        assert polys['kitchen'].equals(box(0,-4.8,5.5,0))
        assert sum(r['room_type']=='dining' for r in rooms)==1
        assert any(r['room_type']=='bedroom' for r in rooms)
        assert any(r['shape']!='rectangle' for r in rooms)
        for r in rooms:
            p=polys[r['name']]
            assert p.is_valid and not p.interiors
            assert p.covers(box(*r['furnishing_bounds']))
            assert unary_union([box(*b) for b in r['floor_rectangles']]).equals(p)
        for i,p in enumerate(polys.values()):
            for q in list(polys.values())[i+1:]:assert p.intersection(q).area<1e-7
        for e in layout['connections']:
            a,b=e['rooms'];axis=int(e['wall'][0]);c=e['center'];u=c.copy();v=c.copy();u[1-axis]-=.95;v[1-axis]+=.95
            opening=LineString([u,v])
            assert polys[a].boundary.buffer(1e-6).covers(opening)
            assert polys[b].boundary.buffer(1e-6).covers(opening)
        if len(accepted)==8:break
    assert len(accepted)==8
