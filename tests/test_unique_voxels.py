import numpy as np
import pytest
from cross_episode_sim.control.sensor_drawer_geometry import unique_voxels

@pytest.mark.parametrize('dtype',[np.int32,np.int64])
@pytest.mark.parametrize('size',[0,1,1000,100000])
def test_exact_cells_order_and_first_indices(dtype,size):
    rng=np.random.default_rng(123)
    rows=rng.integers(-100,100,(size,3),dtype=dtype)
    rows=np.concatenate([rows,rows[:size//2]])
    expected,indices=np.unique(rows,axis=0,return_index=True)
    actual,first=unique_voxels(rows,return_index=True)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(first,indices)
    np.testing.assert_array_equal(unique_voxels(rows),expected)

def test_extreme_coordinates_fall_back_without_overflow():
    v=np.iinfo(np.int64)
    rows=np.array([[v.min,0,0],[v.max,0,0],[v.min,0,0]])
    expected,indices=np.unique(rows,axis=0,return_index=True)
    actual,first=unique_voxels(rows,return_index=True)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(first,indices)
