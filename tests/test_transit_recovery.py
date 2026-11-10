import numpy as np
from types import SimpleNamespace as NS
from cross_episode_sim.navigation.transit_recovery import outside_transit_region,resume_transit


def test_corridor_departure_recovers_without_replacing_destination_or_map():
    free=np.array([[False,True],[False,True]])
    target=np.array([5.,3.]);pose=np.eye(4)
    follower=NS(layers=lambda xy:None,cell=lambda xy:(0,int(xy[0])),free=free,
                goal=target.copy(),terminal='room_route_unobserved',path=[target],
                velocity=np.ones(2),events=[],turn_recovery_active=True)
    tracking=NS(accepted=True,T_map_base=pose)
    assert outside_transit_region(follower,tracking)
    # The physical recovery moves into the existing free region; it does not
    # mark the formerly excluded cell as free to force a successful A* plan.
    before=free.copy();pose[0,3]=1.
    assert not outside_transit_region(follower,tracking)
    resume_transit(follower,tracking,12.)
    np.testing.assert_array_equal(free,before)
    np.testing.assert_array_equal(follower._recovery_destination,target)
    assert follower.terminal is None and not follower.path
    assert not follower.velocity.any() and follower.last_time==12.


def test_missing_localization_is_not_used_as_a_recovery_pose():
    assert not outside_transit_region(None,NS(accepted=False,T_map_base=None))
