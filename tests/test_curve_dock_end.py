"""The curve follower must finish a dock that is inside the docked tolerance."""
import math
import numpy as np
from cross_episode_sim.navigation.shaped_docking import ShapedDocking,DOCKED_TOLERANCE


def follower(goal):
    dock=ShapedDocking.__new__(ShapedDocking)
    points=np.linspace([goal[0]-.3,goal[1],goal[2]],goal,31)
    dock.curve=points;dock.curve_length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))];dock.curve_index=30
    dock.goal=np.asarray(goal,float);dock.path=[dock.goal];dock.target=None;dock.approaching=False;dock.surface_outward=None
    return dock


def test_sideways_residual_inside_the_tolerance_ends_the_dock():
    goal=np.array([0.,0.,0.]);dock=follower(goal)
    pose=np.array([0.,-.027,0.])           # 2.7 cm to the side, heading already right (fix125: 38 s here)
    assert np.linalg.norm(pose[:2]-goal[:2])<DOCKED_TOLERANCE
    command=dock.curve_command(pose)
    assert np.allclose(command,0.) and dock.path==[]


def test_sideways_residual_with_a_heading_error_turns_first():
    dock=follower(np.array([0.,0.,0.]))
    command=dock.curve_command(np.array([0.,-.027,.2]))
    assert command[0]==0. and command[1]<0. and dock.path


def test_residual_ahead_is_still_driven():
    dock=follower(np.array([0.,0.,0.]))
    command=dock.curve_command(np.array([-.027,0.,0.]))
    assert command[0]>0. and dock.path


def test_outside_the_tolerance_is_not_accepted():
    dock=follower(np.array([0.,0.,0.]))
    dock.curve_command(np.array([0.,-.035,0.]))
    assert dock.path
