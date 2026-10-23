"""Contact-only fridge opening and retained-handle closing.

The caller must clear obstructing room doors through explicit skills first.
No fixture state or collision mask is changed here.
"""
import numpy as np

from .door_motion import dock_if_needed, staged_door_motion, stroke
from .fixture_limits import opening_tolerance
from cross_episode_sim.fixtures.door_push import panel_push_geometry


def fridge_door_motion(c, target):
    names = ('door_push_contact', 'fixed_arm_panel_push', 'base_assisted_articulation',
             'base_assisted_close', 'contact_base_speed', 'panel_push_height_offset',
             'panel_push_edge_margin', 'panel_push_tilts', 'base_assisted_handle_close')
    missing = object()
    saved = {name: getattr(c, name, missing) for name in names}
    try:
        c.contact_base_speed = .08
        c.panel_push_height_offset = 0.
        c.panel_push_edge_margin = .14
        c.panel_push_tilts = (0., -30., 30., -45., 45., -60., 60.)
        if abs(c.angle()-target) < opening_tolerance():
            return
        opening = abs(target) > abs(c.angle())
        if not opening:
            # Keep one physical grasp and follow the handle with the base.
            # The old panel dock can lie in the wall in compact kitchens.
            c.base_assisted_handle_close = True
            stroke(c, target)
            if abs(c.angle()-target) > opening_tolerance():
                raise RuntimeError('Fridge did not remain closed after handle release')
            return
        c.door_push_contact = False
        initial = np.sign(target)*min(abs(target), np.radians(60.))
        if abs(c.angle()) < abs(initial)-opening_tolerance():
            staged_door_motion(c, initial)
        c.fixed_arm_panel_push = True
        _, surface, direction = panel_push_geometry(c, target)
        xy = surface[:2]-.60*direction[:2]
        yaw = float(np.arctan2(direction[1], direction[0]))
        dock_if_needed(c, xy, yaw)
        c.push_open_door(target)
    finally:
        for name, value in saved.items():
            if value is missing:
                c.__dict__.pop(name, None)
            else:
                setattr(c, name, value)
