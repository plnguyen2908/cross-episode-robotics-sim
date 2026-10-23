import pytest
from shapely.geometry import box
from cross_episode_sim.generation.circulation import validate_room_circulation


def fixture():
    return dict(name='living',floor_polygon=[[0,0],[6,0],[6,6],[0,6]]),[
        dict(door='left',rooms=['living','a'],center=[0,3],wall=[0,0,0,6]),
        dict(door='right',rooms=['living','b'],center=[6,3],wall=[0,6,0,6])]


def test_blocked_path_rejected_even_when_door_squares_clear():
    r,e=fixture()
    with pytest.raises(ValueError,match='room-local'):
        validate_room_circulation(r,e,[box(2.6,.1,3.4,5.9)])


def test_door_sweep_and_robot_margin():
    r,e=fixture()
    with pytest.raises(ValueError,match='sweep'):
        validate_room_circulation(r,e,[box(.2,3.95,.8,4.5)])
    with pytest.raises(ValueError,match='room-local'):
        validate_room_circulation(r,e,[box(2.6,0,3.4,5.2)])
    assert validate_room_circulation(r,e,[box(2.6,0,3.4,4.4)])['door_count']==2
