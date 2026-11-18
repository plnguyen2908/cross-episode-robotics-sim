"""Measured bottom of a drawer the robot opened itself."""
import numpy as np
from cross_episode_sim.perception.drawer_support import drawer_support_points, drawer_front_aperture

DRAWER=dict(open_handle=[.40,0.,.78],normal=[-1.,0.,0.],opening=.28)   # robot at the origin, drawer pulled toward it


def grid(x0,x1,y0,y1,z,step=.008):
    g=np.mgrid[x0:x1:step,y0:y1:step].reshape(2,-1).T
    return np.column_stack([g,np.full(len(g),z)])


def test_bottom_is_found_and_rim_counter_and_floor_are_not():
    bottom=grid(.45,.64,-.18,.18,.72)             # exposed part of the drawer bottom
    counter=grid(.70,1.2,-.5,.5,.90)              # counter top behind and above
    floor=grid(.0,.35,-.5,.5,.0)                  # floor in front
    rim=grid(.40,.42,-.2,.2,.80)                  # top edge of the drawer front
    surface=drawer_support_points(np.concatenate([bottom,counter,floor,rim]),DRAWER)
    assert len(surface)>200 and np.allclose(surface[:,2],.72)


def test_no_surface_when_the_bottom_is_hidden():
    assert len(drawer_support_points(np.concatenate([grid(.70,1.2,-.5,.5,.90),grid(.0,.35,-.5,.5,.0)]),DRAWER))==0


def test_front_aperture_faces_the_robot_and_is_centred_on_the_handle():
    a=drawer_front_aperture(DRAWER)
    assert np.allclose((np.array(a['hinge'])+np.array(a['closed_handle']))/2,DRAWER['open_handle'])
    assert np.allclose(a['closed_normal'],[-1.,0.,0.])


def test_bottom_under_an_overhanging_counter_is_left_out():
    bottom=grid(.43,.70,-.18,.18,.72)             # bottom visible to 30 cm behind the handle
    counter=grid(.58,1.2,-.5,.5,.90)              # counter edge overhangs from 18 cm behind the handle
    surface=drawer_support_points(np.concatenate([bottom,counter]),DRAWER)
    assert len(surface)>100 and surface[:,0].max()<.58


def test_held_cup_above_the_drawer_is_not_taken_for_a_cover():
    bottom=grid(.43,.70,-.18,.18,.72)
    counter=grid(.62,1.2,-.5,.5,.90)
    cup=grid(.46,.54,-.04,.04,1.05)               # leftover points of the cup held over the drawer
    surface=drawer_support_points(np.concatenate([bottom,counter,cup]),DRAWER)
    assert len(surface)>100 and .55<surface[:,0].max()<.62


def test_the_drawers_own_front_panel_is_not_taken_for_a_cover():
    # fix81: front panel face and its top edge (10 cm above the handle) fill the width just behind the handle
    g=np.mgrid[-.27:.27:.008,.68:.89:.008].reshape(2,-1).T
    panel=np.column_stack([np.full(len(g),.44),g[:,0],g[:,1]])        # vertical face 4 cm behind the handle
    rim=grid(.44,.48,-.27,.27,.88)                                    # its top edge
    bottom=grid(.49,.74,-.18,.18,.72)
    cabinet=grid(.72,1.2,-.5,.5,.95)                                  # counter edge, 28 cm behind the panel
    surface=drawer_support_points(np.concatenate([panel,rim,bottom,cabinet]),dict(DRAWER,opening=.375))
    assert len(surface)>300 and surface[:,0].max()<.72


def test_opened_drawer_box_in_the_map_is_limited_to_its_height_band():
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap
    from cross_episode_sim.evaluation.sensor_household import leaf_mask, DRAWER_HALF_WIDTH
    grid=ObservedNavigationMap()
    leaf=dict(hinge=[1.0,0.0],open_handle=[1.0,-.375],top=.69,bottom=.45,half_thickness=DRAWER_HALF_WIDTH,extension=.05)
    mask=leaf_mask(grid,leaf)
    heights=grid.config.min_obstacle_height+(np.arange(mask.shape[2])+.5)*grid.config.height_resolution
    used=heights[mask.any(axis=(0,1))]
    assert mask.any() and used.min()>=.45 and used.max()<=.69
    columns=mask.any(axis=2);rows,cols=np.where(columns)
    xy=grid.cell_to_world(np.stack([cols,rows],axis=-1))
    cell=grid.config.resolution   # cell centres: allow one cell of rounding
    assert np.ptp(xy[:,0])>=2*DRAWER_HALF_WIDTH-cell and xy[:,1].min()<=-.375-.05+cell and xy[:,1].max()<=cell
    door=leaf_mask(grid,dict(hinge=[0.,0.],open_handle=[.45,0.],top=1.0))   # door leaf: unchanged, down to the floor
    assert heights[door.any(axis=(0,1))].min()<.2
