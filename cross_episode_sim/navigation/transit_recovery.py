"""Recover small tracking departures without changing a frozen navigation map."""
import numpy as np


def outside_transit_region(follower, tracking):
    if not tracking.accepted or tracking.T_map_base is None:
        return False
    xy=tracking.T_map_base[:2,3]
    follower.layers(xy)
    row,col=follower.cell(xy)
    return not (0<=row<follower.free.shape[0] and 0<=col<follower.free.shape[1]
                and follower.free[row,col])


def resume_transit(follower, tracking, now):
    """Replan from the recovered pose, retaining the semantic destination."""
    if follower.goal is not None:
        follower._recovery_destination=follower.goal.copy()
    follower.terminal=None;follower.path=[];follower.velocity[:]=0.
    follower.last_time=now;follower.last_progress=tracking.T_map_base[:2,3].copy()
    follower.progress_time=now;follower.turn_recovery_active=False
    follower.events.append(dict(event='transit_region_recovered',time=now,
                                position=follower.last_progress.tolist()))
