"""Two-knob stove for evaluation scenes (user decision, November 19, 2026).

The front-control stove (RoboCasa Stove002) authors seven knobs: oven
temperature, oven timer, front-left, rear-left, centre, rear-right and
front-right. Every warm-two-pots episode turns only the front-left and
front-right knobs (174 of 174 v3/v4 episodes), and the other five are fixed
in all saved scenes. Evaluation scenes delete those five knob bodies and their
geometry outright, so a policy sees one knob per used burner.

Applied at compile time on the episode's own scene spec; nothing in the saved
episode folders changes. This is a scene change relative to the saved
episodes and is disclosed with evaluation results.
"""

KEPT_KNOBS = ('front_left', 'front_right')
REMOVED_KNOBS = ('temp', 'time', 'rear_left', 'center', 'rear_right')
PREFIX = 'stove_main_group_knob_'


def removed_knob_bodies():
    return tuple(PREFIX + name for name in REMOVED_KNOBS)


def remove_extra_stove_knobs(spec):
    """Delete the five unused knob bodies (with their geoms) from an MjSpec; return the deleted names."""
    deleted = []
    for name in removed_knob_bodies():
        body = spec.body(name)
        if body is None:
            continue
        spec.delete(body)
        deleted.append(name)
    return deleted


def has_extra_stove_knobs(model):
    """True when a compiled model still contains any of the five removed knob bodies."""
    import mujoco
    return any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0 for name in removed_knob_bodies())


# Pot scenes (user decision, November 19, 2026: "change the color to ... red
# for all pot scene"): the two kept knobs are painted plain red so a pointing
# model finds them on the black panel instead of the printed dial marks.
KNOB_RGBA = (.80, .05, .05, 1.)
PAINTED_TASKS = ('warm_two_pots',)


def paint_kept_knobs(model, rgba=KNOB_RGBA):
    """Set the kept knobs' visual geoms (no collision) to a plain colour, on a compiled model; return geom names."""
    import mujoco
    import numpy as np
    painted = []
    for name in KEPT_KNOBS:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + name)
        if body < 0:
            continue
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                model.geom_matid[g] = -1
                model.geom_rgba[g] = np.asarray(rgba, float)
                painted.append(model.geom(g).name or f'geom{g}')
    return painted
