"""Shared pick/place task execution used by household composites and fixtures.

Task controllers select roles and geometry. These functions own pickup/delivery
skill boundaries; a completed pickup is not rolled back by a delivery failure.
"""

import inspect

import numpy as np

from cross_episode_sim.skills.atomic import CallbackSkill


class TransferOperation:
    """Role/fixture bookkeeping around the shared pickup and delivery skills.

    No outer checkpoint: successful pickup remains committed if delivery fails.
    The controller adapter resolves phase-dependent source/target geometry and
    storage access; execute_transfer owns the shared skill dispatch.
    """

    def __init__(self, controller, execute=None, verify=None):
        self.controller = controller
        self.execute = execute or controller.transfer_role
        self.verify = verify or controller.verify_role

    def validate_arguments(self, **arguments):
        inspect.signature(self.execute).bind(**arguments)
        inspect.signature(self.verify).bind(**arguments)

    def run(self, **arguments):
        self.validate_arguments(**arguments)
        self.execute(**arguments)
        if not self.verify(**arguments):
            raise RuntimeError("Measured postcondition failed: pick/place sequence")


def pickup(controller, execute_once):
    c = controller
    if not getattr(c, "speculative_dock_trials", False):
        return CallbackSkill(
            c,
            "pickup",
            (None,),
            lambda _: execute_once(),
            verify=lambda: bool(c.holding_loaf and len(c.contacts()) >= 2),
        ).run()
    info = c.object_info[c.object_name]
    # Freeze the candidates at the original scene, not at a failed endpoint.
    docks = list(c.dock_candidates(info["source"], c.bread_pose()[:3, 3], pickup=True))
    for dock in c.grasp_recovery_stances():
        if c.room_id(dock[:2]) == c.room_id(c.bread_pose()[:2, 3]) and not any(
            np.linalg.norm(dock - old) < 0.01 for old in docks
        ):
            docks.append(dock)
    candidates = [(dock, angled) for angled in (False, True) for dock in docks]
    physical_failures = 0
    limit = c.max_physical_grasp_attempts

    def execute(candidate):
        nonlocal physical_failures
        dock, angled = candidate
        c._trial_pick_dock = dock
        c._trial_pick_angled = angled
        c._trial_pick_active = True
        c.report["physical_grasp_attempted"] = False
        try:
            return execute_once()
        except RuntimeError:
            if c.report.get("physical_grasp_attempted"):
                physical_failures += 1
            raise

    def bounded():
        for candidate in candidates:
            if physical_failures >= limit:
                break
            yield candidate

    c.max_physical_grasp_attempts = 1
    try:
        return CallbackSkill(c, "pickup", bounded, execute, verify=lambda: bool(c.holding_loaf)).run()
    finally:
        c.max_physical_grasp_attempts = limit
        c._trial_pick_active = False


def delivery(controller, execute_once):
    c = controller
    if not getattr(c, "speculative_dock_trials", False):
        return CallbackSkill(
            c, "delivery", (None,), lambda _: execute_once(), verify=lambda: bool(c.report.get("success"))
        ).run()
    c._trial_rejected_docks = set()
    c._trial_delivery = True

    def execute(index):
        c._redock_index = index
        return execute_once()

    try:
        return CallbackSkill(
            c,
            "delivery",
            range(5),
            execute,
            verify=lambda: bool(c.report.get("success")),
            rebuild=c.rebuild_loaded_planner,
        ).run()
    finally:
        c._trial_delivery = False
        c._trial_rejected_docks = set()


def prepare_transfer_role(c, role):
    """Bind geometry without moving the robot or executing a transfer."""
    info = next(i for i in c.manifest["bindings"] if i["role"] == role)
    if c.assignment()[info["body"]] != c.task_supports[info["source"]]:
        raise RuntimeError(f"Unexpected starting support for {role}")
    c.select_object(info["body"])
    c.source, c.destination = c.task_supports[info["source"]], c.task_supports[info["destination"]]
    c.support_bids = c.table_bids[c.source]
    c.table_gids = {
        g
        for g in range(c.model.ngeom)
        if c.model.geom_bodyid[g] in c.support_bids and (c.model.geom_contype[g] or c.model.geom_conaffinity[g])
    }
    c.initial_object_pose = c.bread_pose().copy()
    c.transfer_start = c.data.joint(c.object_joint).qpos.copy()
    c.report["success"] = False


def transfer_role(c, role):
    prepare_transfer_role(c, role)
    c.execute_transfer()
    c.report.setdefault("executed_placement_targets", {})[role] = c.destination_pose[:3, 3].tolist()
