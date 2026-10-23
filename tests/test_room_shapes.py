import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from cross_episode_sim.generation.base_layout import sample_layout
from cross_episode_sim.generation.room_shapes import room_segments


def test_real_floor_polygons_and_doorway_preservation():
    valid=0
    for seed in range(80):
        try:layout=sample_layout(seed,[0.,-4.8,5.5,0.],9)
        except ValueError:continue
        valid+=1
        for room in layout['rooms']:
            polygon=Polygon(room['floor_polygon'])
            assert polygon.is_valid
            tiles=unary_union([box(*r) for r in room['floor_rectangles']])
            assert tiles.symmetric_difference(polygon).area<1e-8
            if room['room_type'] in ('living','bedroom'):
                assert polygon.area<box(*room['bounds']).area
                assert len(room['floor_polygon'])>=6
                for cut in room['structural_returns']:
                    assert polygon.intersection(box(*cut['bounds'])).area<1e-8
            for edge in layout['connections']:
                if room['name'] not in edge['rooms']:continue
                axis,c,_,_=edge['wall'];axis=int(axis);mid=edge['center'][1-axis]
                assert any(a==axis and abs(coord-c)<1e-6 and lo<=mid-.95 and hi>=mid+.95
                           for _,a,coord,lo,hi in room_segments(room))
    assert valid>=10
