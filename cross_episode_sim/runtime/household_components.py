"""Reuse qualified live household components from a continuous task recipe.

The recipe supplies semantic targets, never saved component geometry or poses.
Fixtures retain this episode's RGB-D aperture estimates. Handoffs reference the
actual executed pickup so frozen-map carry gets its observed payload and paths.
"""
from . import artifacts


def confined_pick_options(target):
    if target == 'food container inside microwave':
        return dict(target=target, redock=True, redock_distance=.8,
                    source_receptacle='microwave', approach_mode='front_entry',
                    front_wrist='tilted_rim', wrist_tilt_degrees=55.,
                    rim_height_offset=.005, entry_height=.04)
    if target == 'cold drink inside fridge':
        return dict(target=target, redock=True, source_receptacle='fridge',
                    approach_mode='front_entry')
    return None


def microwave_place_options():
    return dict(receptacle='microwave', region='interior',
                approach_mode='front_entry', front_wrist='object_level',
                placement_preference='center', object_tilt_degrees=40., entry_height=0.)


def live_session(runtime, root, previous, fixtures, target, held=None):
    from ..code_policy.household_session import HouseholdSession
    from ..navigation.household_map_handoff import bind_observed_map
    root.mkdir(parents=True, exist_ok=False)
    bind_observed_map(runtime, previous)
    session = HouseholdSession(runtime, runtime.sensors, root, target,
        tool_budget=24, held=held, max_place_candidates=32, interactive_doors=False)
    session.mapper = runtime.mapper.view()
    # Seed this view from the existing tracker, without initializing another
    # tracker or making an empty pose array at the first local checkpoint.
    packet = runtime.sensors.capture(include_robot_mask=False)
    session.mapper.update(packet.cameras['navigation'], packet.timestamp)
    session.fixtures = fixtures
    if held is not None:
        session.payload = artifacts.load_array(previous/'observed_payload_tool.npy')
    return session


def pick_confined(runtime, root, previous, fixtures, target, *, redock=True):
    from ..mapping.continuity import load_memory, save_memory
    options = confined_pick_options(target)
    if options is None:
        raise ValueError('No qualified confined pickup profile for '+target)
    options['redock']=redock
    session = live_session(runtime, root, previous, fixtures, target)
    # Candidate ranking is enabled only for this fridge operation, just as in
    # its qualified component. Do not change other navigation/docking calls.
    original = getattr(runtime, 'qwen_docking', False)
    try:
        if options['source_receptacle'] == 'fridge':runtime.qwen_docking = True
        outcome = session.execute_skill('pick', options)
    finally:
        runtime.qwen_docking = original
    physics = getattr(session, 'last_pick_output', None)
    if physics is not None:
        save_memory(physics, load_memory(previous))
    return outcome, physics


def place_microwave(runtime, root, previous, fixtures, held):
    from ..mapping.continuity import load_memory, save_memory
    session = live_session(runtime, root, previous, fixtures, held, held=held)
    # Reuse the physically qualified failed-state recovery: expose the held
    # object/support, center on the measured opening at 0.60 m standoff, then
    # insert. Generic loaded approach can choose an unreachable lateral/far
    # dock merely because its base path is shorter.
    session.redock(target='microwave',purpose='place',distance=.6,lateral_offset=0.)
    outcome = session.execute_skill('place', microwave_place_options())
    placed = getattr(session, 'last_placement_output', None)
    if placed is not None:
        save_memory(placed, load_memory(previous))
    return outcome, placed
