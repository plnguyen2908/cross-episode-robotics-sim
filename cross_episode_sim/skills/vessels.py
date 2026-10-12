"""Shared physical leaves for qualified breakfast vessel manipulation."""

from cross_episode_sim.skills.transfer import prepare_transfer_role


def navigate_to_vessel(controller, role):
    prepare_transfer_role(controller, role)
    controller.prepare_pickup()


def pick_vessel(controller):
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    # Use the same inherited grasp and hold qualification as the scripted task;
    # the public navigate action has already prepared the pickup dock.
    super(BreakfastEpisode, controller).pick_payload()
    controller.attach_payload_to_planner()


def navigate_with_vessel(controller, role, destination):
    dock = getattr(controller, '_skill_dock', None)
    if dock is not None and dock[:2] == (role, destination):
        # Reuse the existing alternate-dock/filling-surface policy, explicitly
        # inside navigation. Preserve the original pickup yaw for placement.
        controller.redock_loaded_for_placement()
    else:
        controller.transport_payload()


def place_vessel(controller, role):
    controller.place_payload()
    controller.report.setdefault('executed_placement_targets', {})[role] = controller.destination_pose[:3, 3].tolist()
