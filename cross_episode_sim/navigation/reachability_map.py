"""Physics-verified reachability maps (OneRing-style navigation grids).

The precomputed ``<scene>_map.png`` occupancy (and the radius-inflated render)
approximate the robot as a circle, so a doorway narrower than 2*agent_radius is
sealed even when the real (non-circular) base fits, and A* on a blanket-carved
doorway routes through cells the physical robot does NOT fit -> the wide base
clips the frame.

This module is the MuJoCo analogue of AI2-THOR's ``get_reachable_positions``:
sweep the robot base over a lattice of candidate cells (default 0.1 m) and keep
only the cells where the FULL robot collision model fits without contacting
anything but the floor. The result is painted back into a ``ProcTHORMap`` so
every existing consumer (A* planner, NavGoalSampler) works unchanged.

Each cell is tested at multiple headings, not just yaw=0, and is only marked
reachable if the robot fits at all of them: LinearBot's footprint (tucked arms
plus the head-camera boom) is not rotationally symmetric, so a doorway can be
wide enough face-on but clip a corner at the heading the base actually arrives
with mid-turn.

Only ``compute_physics_reachable_map`` (+ its two helpers) is ported here; the
OneRing goal-pose/visibility selection from the source branch is intentionally
omitted -- the cleaning pipeline keeps its own same-room approach.
"""

from __future__ import annotations

import logging
import time

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

from molmo_spaces.utils.scene_maps import ProcTHORMap

log = logging.getLogger(__name__)


def _floor_root_ids(model) -> np.ndarray:
    """Root body ids whose name marks them as floor (contacts with these are OK)."""
    ids = []
    for i in range(model.nbody):
        if model.body_rootid[i] == i and "floor" in (model.body(i).name or "").lower():
            ids.append(i)
    return np.array(ids, dtype=np.int32)


_DOOR_NAME_PREFIXES = ("door_", "doorway_", "doorframe_")


def _door_root_ids(model) -> np.ndarray:
    """Root body ids for door/doorway/doorframe objects (contacts with these are OK).

    PORTED from shailes-h's `origin/linearbot_datagen--nav` commit 05c4a30 ("[BUGFIX]: Treat doors
    as walkable in the physics reachability sweep"), which we could not cherry-pick directly: their
    reachability_map.py has since grown to 714 lines against our 201, and the commit also touches
    molmo_spaces/onering/, which does not exist on this branch.

    Same naming convention and rationale scene_maps.py's static map builder already uses to treat
    every discovered door as open/walkable. The physics sweep does its own from-scratch collision
    check, so without this it re-blocks doorways the static map correctly opened: the door
    frame/leaf's own collision geometry (not the wall) counts as an obstacle, closing off
    single-width doorways the robot's true footprint can actually pass through.
    """
    ids = []
    for i in range(model.nbody):
        name = model.body(i).name or ""
        if model.body_rootid[i] == i and name.startswith(_DOOR_NAME_PREFIXES):
            ids.append(i)
    return np.array(ids, dtype=np.int32)


def _robot_in_collision(
    model, data, robot_root: int, ignorable_roots: np.ndarray, margin: float = 0.0
) -> bool:
    """Vectorized version of env.check_robot_collision_in_current_pose: any active contact between
    a robot body and an environment body that isn't in ``ignorable_roots`` (floor and
    door/doorway/doorframe roots -- see ``_floor_root_ids`` / ``_door_root_ids``).

    ``margin`` (meters, >= 0) tolerates up to that much interpenetration before counting a contact
    as blocking -- for a sweep that should open up doorways only marginally narrower than the
    robot's true footprint, at the cost of accepting light grazing contact the real robot would
    feel. 0.0 (default) is the exact/strict check used for real episode physics.
    """
    n = data.ncon
    if n == 0:
        return False
    con = data.contact
    active = con.dist[:n] <= -margin
    b1 = model.geom_bodyid[con.geom1[:n]]
    b2 = model.geom_bodyid[con.geom2[:n]]
    r1 = model.body_rootid[b1]
    r2 = model.body_rootid[b2]
    is_robot1 = r1 == robot_root
    is_robot2 = r2 == robot_root
    involved = is_robot1 ^ is_robot2  # XOR also drops robot self-contacts
    other = np.where(is_robot1, r2, r1)
    bad = active & involved & ~np.isin(other, ignorable_roots)
    # ProcTHOR floors and walls can share a room root. Ignore only low wheel/base
    # support contacts; do not ignore the room root wholesale or walls disappear.
    other_body = np.where(is_robot1, b2, b1)
    robot_body = np.where(is_robot1, b1, b2)
    for index in np.flatnonzero(bad):
        other_name = (model.body(int(other_body[index])).name or '').lower()
        robot_name = (model.body(int(robot_body[index])).name or '').lower()
        floor_support = (con.pos[index, 2] < .08
                         and other_name.startswith(('room_', 'floor'))
                         and any(token in robot_name for token in ('base', 'wheel')))
        if floor_support:
            bad[index] = False
    return bool(bad.any())


def compute_physics_reachable_map(
    env,
    robot_view,
    base_map: ProcTHORMap,
    grid_size: float = 0.1,
    base_z: float = 0.13,
    robot_base_body: str = "robot_0/base",
    keep_largest_component: bool = True,
    require_all_headings: bool = True,
    yaws_deg: tuple[float, ...] = tuple(range(0, 360, 45)),
    collision_margin_m: float = 0.0,
) -> ProcTHORMap:
    """Build an occupancy map from actual robot-vs-scene collision checks.

    For every ``grid_size`` cell whose center is free in ``base_map`` (a purely
    static-geometry prefilter), the robot base is placed at the cell center at
    height ``base_z`` and the scene collision model is evaluated at each heading
    in ``yaws_deg``; optionally only the largest connected component is kept.

    ``require_all_headings`` controls the per-cell rule:
      * True  -- navigable only if the base fits at EVERY heading. Stricter than
        AI2-THOR's (heading-agnostic) get_reachable_positions; guards against A*
        routing through a cell that only fits face-on and clipping mid-turn.
      * False -- navigable if the base fits at ANY heading (the get_reachable_
        positions semantics). REQUIRED for a wide, non-square base + narrow
        doors: a ~0.9 m doorway admits the 0.745x0.875 m base only at the
        through-heading (its 45-deg diagonal is ~1.15 m), so all-headings seals
        every doorway and disconnects the rooms. Any-heading keeps the door
        corridor connected; the holonomic base + NavigateSkill's doorway-centre
        steering align it to the opening on the way through.

    The robot must already be in its travel configuration (arms tucked, lift
    down) -- joint state is taken as-is; only the base pose is swept.

    Returns a new ProcTHORMap with the same transforms/rooms as ``base_map``.
    """
    model = env.current_model
    data = env.current_data

    robot_root = int(
        model.body_rootid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_base_body)]
    )
    # Floor AND doors are walkable: a door's own frame/leaf geometry must not re-block a doorway the
    # static map already opened (see _door_root_ids).
    ignorable_roots = np.concatenate([_floor_root_ids(model), _door_root_ids(model)])

    occ = base_map.occupancy
    stride = max(1, int(round(grid_size * base_map.px_per_m)))
    n_rows = occ.shape[0] // stride
    n_cols = occ.shape[1] // stride

    # Cell-center pixel lattice and static-map prefilter
    center_r = np.arange(n_rows) * stride + stride // 2
    center_c = np.arange(n_cols) * stride + stride // 2
    candidate = occ[np.ix_(center_r, center_c)]

    cand_rows, cand_cols = np.nonzero(candidate)
    centers_px = np.stack([center_r[cand_rows], center_c[cand_cols]], axis=1)
    centers_m = base_map.pos_px_to_m(centers_px)

    original_pose = robot_view.base.pose.copy()
    reachable = np.zeros((n_rows, n_cols), dtype=bool)
    pose = np.eye(4)
    rotations = [R.from_euler("z", np.deg2rad(yaw)).as_matrix() for yaw in yaws_deg]

    # mj_collision's cost is dominated by the O(n^2) broad phase over ALL geom
    # pairs, but we only care about robot-vs-everything-else. Filter the rest
    # out via contype/conaffinity for the sweep; restored below regardless.
    contype0 = model.geom_contype.copy()
    conaffinity0 = model.geom_conaffinity.copy()
    is_robot_geom = model.body_rootid[model.geom_bodyid] == robot_root
    model.geom_contype[is_robot_geom] = 1
    model.geom_conaffinity[is_robot_geom] = 1
    model.geom_contype[~is_robot_geom] = 0
    model.geom_conaffinity[~is_robot_geom] = 1

    n_yaw_checks = 0
    t0 = time.perf_counter()
    try:
        for (row, col), pos in zip(zip(cand_rows, cand_cols), centers_m, strict=True):
            pose[:3, 3] = (pos[0], pos[1], base_z)
            # all-headings: start True, one collision rejects. any-heading: start False, one fit accepts.
            ok = require_all_headings
            for rot in rotations:
                pose[:3, :3] = rot
                robot_view.base.pose = pose
                mujoco.mj_kinematics(model, data)
                mujoco.mj_collision(model, data)
                n_yaw_checks += 1
                collides = _robot_in_collision(
                    model, data, robot_root, ignorable_roots, collision_margin_m
                )
                if require_all_headings and collides:
                    ok = False
                    break  # one bad heading rejects the cell
                if not require_all_headings and not collides:
                    ok = True
                    break  # fits at some heading -> navigable (keeps door corridors connected)
            reachable[row, col] = ok
    finally:
        model.geom_contype[:] = contype0
        model.geom_conaffinity[:] = conaffinity0
        robot_view.base.pose = original_pose
        mujoco.mj_forward(model, data)

    n_checked = len(cand_rows)
    n_free = int(reachable.sum())

    if keep_largest_component and n_free > 0:
        labels, n_labels = ndimage.label(reachable, structure=np.ones((3, 3)))
        if n_labels > 1:
            sizes = np.bincount(labels.ravel())[1:]  # component sizes, label 0 = background
            reachable = labels == (int(np.argmax(sizes)) + 1)

    # Paint reachable cells back onto the base map's pixel canvas. A cell is free
    # only if the whole robot footprint fits at its center, so filling the full
    # cell is already conservative at robot scale.
    new_occ = np.zeros_like(occ, dtype=bool)
    blocks = np.kron(reachable, np.ones((stride, stride), dtype=bool))
    new_occ[: blocks.shape[0], : blocks.shape[1]] = blocks

    log.info(
        f"[REACHABILITY] swept {n_checked} cells ({grid_size}m, up to {len(yaws_deg)} "
        f"headings each, {n_yaw_checks} pose checks total) in "
        f"{time.perf_counter() - t0:.1f}s: {n_free} robot-fits, "
        f"{int(reachable.sum())} in main component"
    )

    return ProcTHORMap(
        occupancy=new_occ,
        world_to_map=base_map.world_to_map,
        map_to_world=base_map.map_to_world,
        px_per_m=base_map.px_per_m,
        room_map=base_map.room_map,
        room_ids_to_name=base_map.room_ids_to_name,
    )
