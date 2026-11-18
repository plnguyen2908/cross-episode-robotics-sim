"""Reuse qualified live household components from a continuous task recipe.

The recipe supplies semantic targets, never saved component geometry or poses.
Fixtures retain this episode's RGB-D aperture estimates. Handoffs reference the
actual executed pickup so frozen-map carry gets its observed payload and paths.
"""
import json
from . import artifacts


FRIDGE_PICK_DISTANCE=.75
# Looks-and-plans spent on one pick inside the fridge or microwave.
CONFINED_PICK_ATTEMPTS=3
# Between failed looks at the food inside the microwave the robot stands this much
# closer (m), never nearer than the minimum.
MICROWAVE_PICK_CLOSER_STEP=.05
MICROWAVE_PICK_MIN_DISTANCE=.68
FRIDGE_PICK_CLOSER_STEP=.03
FRIDGE_PICK_MIN_DISTANCE=.69
# Receptacle -> (step closer per failed look (m), nearest stand distance (m)).
CLOSER_RETRY_STEPS={'microwave':(MICROWAVE_PICK_CLOSER_STEP,MICROWAVE_PICK_MIN_DISTANCE),
                    'fridge':(FRIDGE_PICK_CLOSER_STEP,FRIDGE_PICK_MIN_DISTANCE)}
CLOSER_RETRY_RECEPTACLES=tuple(CLOSER_RETRY_STEPS)


def confined_pick_options(target):
    if target == 'food container inside microwave':
        return dict(target=target, redock=True, redock_distance=.8,
                    source_receptacle='microwave', approach_mode='front_entry',
                    front_wrist='tilted_rim', wrist_tilt_degrees=55.,
                    rim_height_offset=.005, entry_height=.04)
    if target == 'cold drink inside fridge':
        # 0.75 m from the bottle, as in the qualified component
        # (docs/qwen_household_results.md, fridge_horizontal_pick15_full). At
        # the default 0.55-0.62 m the level insertion ran out of arm travel at
        # 97% of its path (fix105 saved state).
        return dict(target=target, redock=True, redock_distance=FRIDGE_PICK_DISTANCE,
                    source_receptacle='fridge', approach_mode='front_entry')
    return None


ORACLE_MICROWAVE_FLOOR_STANDOFF=.50


def floor_front_distance(surface):
    """Distance ahead of the base (m) to the front edge of an observed cavity floor (base frame, facing it)."""
    import numpy as np
    surface=np.asarray(surface,float).reshape(-1,3)
    return None if len(surface)<30 else float(np.percentile(surface[:,0],5))


def oracle_redock_distance(front,current=.6,standoff=ORACLE_MICROWAVE_FLOOR_STANDOFF,minimum=.35):
    """New door-plane distance that puts the floor's front edge `standoff` ahead; None when already there."""
    if front is None or front-standoff<.03:return None
    return float(max(minimum,current-(front-standoff)))


def oracle_microwave_distance(session):
    placed=getattr(session,'last_placement_output',None)
    if placed is None or not artifacts.exists(placed/'input.npz'):return None
    return oracle_redock_distance(floor_front_distance(artifacts.load_array(placed/'input.npz')['table_surface']))


# Distance ahead of the base (m) at which the measured cavity-floor centre is
# comfortably inside arm reach for the tilted centred placement.
MICROWAVE_CENTRE_REACH=.77
MICROWAVE_MIN_DOCK_DISTANCE=.45


def centre_redock_distance(centre_ahead,current=.6,reach=MICROWAVE_CENTRE_REACH,minimum=MICROWAVE_MIN_DOCK_DISTANCE):
    """Door-plane distance that brings a floor centre measured `centre_ahead` m ahead to `reach`; None when close enough."""
    if centre_ahead is None or centre_ahead-reach<.02:return None
    return float(max(minimum,current-(centre_ahead-reach)))


def centred_redock_distance(session):
    placed=getattr(session,'last_placement_output',None)
    if placed is None or not artifacts.exists(placed/'center_preference.json'):return None
    centre=json.loads(artifacts.read_text(placed/'center_preference.json')).get('target_base')
    return None if centre is None else centre_redock_distance(float(centre[0]))


# Insertion poses tried per microwave placement. With 32 the one workable pose for
# the grasp of fix111 (number 56) was never reached.
MICROWAVE_PLACE_CANDIDATES=96


# Sideways shifts (m, away from the opened door leaf) offered for the closer microwave dock.
MICROWAVE_REDOCK_SHIFTS=(0.,.06,.10)
# Stand spots (door-plane distance m, sideways shift m away from the leaf) tried in turn
# when the cavity floor was not seen from the first one.
MICROWAVE_SUPPORT_VIEW_REDOCKS=((.65,0.),(.60,.06),(.55,0.))
# Sideways shifts (m, away from the opened leaf) for the first placement dock when the
# centred spot has no route.
MICROWAVE_FIRST_DOCK_SHIFTS=(.06,.10,.14)


def away_from_open_leaf(aperture):
    """+1 or -1: the sign of a lateral dock offset that moves away from the opened leaf (0-safe default +1)."""
    import numpy as np
    if not aperture or 'open_handle' not in aperture:return 1.
    normal=np.asarray(aperture['closed_normal'],float)[:2];normal=normal/np.linalg.norm(normal)
    tangent=np.array([-normal[1],normal[0]])
    centre=(np.asarray(aperture['closed_handle'],float)[:2]+np.asarray(aperture['hinge'],float)[:2])/2
    leaf=np.asarray(aperture['open_handle'],float)[:2]-centre
    return -1. if float(leaf@tangent)>0 else 1.


def outcome_field(outcome,name,default=None):
    """A skill outcome is a dict from the session dispatcher or an object from a handler."""
    return outcome.get(name,default) if isinstance(outcome,dict) else getattr(outcome,name,default)


def microwave_place_options():
    return dict(receptacle='microwave', region='interior',
                approach_mode='front_entry', front_wrist='object_level',
                placement_preference='center', object_tilt_degrees=40., entry_height=0.)


def live_session(runtime, root, previous, fixtures, target, held=None, max_place_candidates=32):
    from ..code_policy.household_session import HouseholdSession
    from ..navigation.household_map_handoff import bind_observed_map
    root.mkdir(parents=True, exist_ok=False)
    bind_observed_map(runtime, previous)
    session = HouseholdSession(runtime, runtime.sensors, root, target,
        tool_budget=24, held=held, max_place_candidates=max_place_candidates, interactive_doors=False)
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
        for attempt in range(CONFINED_PICK_ATTEMPTS):
            outcome = session.execute_skill('pick', options)
            if outcome_field(outcome,'success',False):break
            # The same pick from the same dock failed live and passed when
            # replayed (fix113: every level reach refused in planning, nothing
            # moved): proposals and reach planning vary from look to look. Look
            # and plan again from where the arm now is before giving up.
            print('CONFINED_PICK_RETRY',json.dumps(dict(target=target,attempt=attempt+1,
                reason=str(outcome_field(outcome,'reason',''))[:80])),flush=True)
            if options['source_receptacle'] in CLOSER_RETRY_RECEPTACLES and attempt+1<CONFINED_PICK_ATTEMPTS:
                # Reach binds this pick: from 0.81 m the food 0.87 m ahead was
                # picked (fix125) and the same food 0.89 m ahead was not, every
                # grasp refused as unreachable part-way into the insertion
                # (fix132, three looks from the same spot). Stand closer for
                # the next look instead of repeating it from there.
                # The fridge drink likewise: picked from 0.76-0.77 m ahead of the
                # base on a later look (fix122, fix144), every look refused at
                # 0.78-0.79 m (fix145 call 26); from 0.55-0.62 m the level entry
                # ran out the other way (fix105), hence the smaller steps there.
                step,floor=CLOSER_RETRY_STEPS[options['source_receptacle']]
                closer=max(floor,options['redock_distance']-step*(attempt+1))
                try:
                    moved=session.redock(target,'pick',closer)
                    print('CONFINED_PICK_REDOCK_CLOSER',json.dumps(dict(distance_m=round(closer,3),
                        success=bool(outcome_field(moved,'success',False)),reason=str(outcome_field(moved,'reason',''))[:80],
                        final_distance_m=outcome_field(moved,'final_distance_m'))),flush=True)
                except RuntimeError as exc:
                    print('CONFINED_PICK_REDOCK_CLOSER',json.dumps(dict(distance_m=round(closer,3),success=False,reason=str(exc)[:120])),flush=True)
    finally:
        runtime.qwen_docking = original
    physics = getattr(session, 'last_pick_output', None)
    if physics is not None:
        memory = load_memory(previous)
        save_memory(physics, memory)
        # The carry that follows reads the room/table memory from the pick's
        # own observation folder (its "navigation source"), not from the
        # executed grasp. Without it the carry to the dining table stopped
        # with "Fixed-map navigation requires an observed destination" (fix107).
        continuation = physics/'continuation.json'
        if artifacts.exists(continuation):
            from pathlib import Path
            source = json.loads(artifacts.read_text(continuation)).get('navigation_source')
            if source and not artifacts.exists(Path(source)/'semantic_memory.json'):
                save_memory(Path(source), memory)
    return outcome, physics


def place_microwave(runtime, root, previous, fixtures, held):
    from ..mapping.continuity import load_memory, save_memory
    session = live_session(runtime, root, previous, fixtures, held, held=held, max_place_candidates=MICROWAVE_PLACE_CANDIDATES)
    # Reuse the physically qualified failed-state recovery: expose the held
    # object/support, center on the measured opening at 0.60 m standoff, then
    # insert. Generic loaded approach can choose an unreachable lateral/far
    # dock merely because its base path is shorter.
    try:session.redock(target='microwave',purpose='place',distance=.6,lateral_offset=0.)
    except RuntimeError as exc:
        if 'No loaded aperture-facing dock' not in str(exc):raise
        # No route to the centred stand spot: with the food raised, the upper
        # arm is 20 cm beside the base, where the opened door leaf stands
        # (fix128, fix134, fix136: this first dock refused, placement never
        # tried). Stand a little to the side away from the leaf instead; the
        # placement re-docks by the measured centre afterwards when needed.
        side=away_from_open_leaf(session.aperture('microwave',map_frame=True));stood=False
        for shift in MICROWAVE_FIRST_DOCK_SHIFTS:
            print('MICROWAVE_FIRST_DOCK_SHIFTED',json.dumps(dict(lateral_offset_m=side*shift,after=str(exc)[:80])),flush=True)
            try:
                session.redock(target='microwave',purpose='place',distance=.6,lateral_offset=side*shift);stood=True;break
            except RuntimeError as again:
                if 'No loaded aperture-facing dock' not in str(again):raise
        if not stood:raise
    outcome = session.execute_skill('place', microwave_place_options())
    if not outcome_field(outcome,'success',False) and 'support_not_visible' in str(outcome_field(outcome,'reason','')):
        # The cavity floor was not seen from this stand spot (fix139: placement
        # ended there with the food still held). The skill's own advice is to
        # re-dock to expose the support: look again from a little further
        # back, then from a little to the side away from the door leaf.
        side=away_from_open_leaf(session.aperture('microwave',map_frame=True))
        for distance,shift in MICROWAVE_SUPPORT_VIEW_REDOCKS:
            print('MICROWAVE_SUPPORT_VIEW_REDOCK',json.dumps(dict(redock_distance_m=distance,lateral_offset_m=side*shift)),flush=True)
            try:session.redock(target='microwave',purpose='place',distance=distance,lateral_offset=side*shift)
            except RuntimeError as exc:
                print('MICROWAVE_SUPPORT_VIEW_REDOCK_REFUSED',str(exc)[:160],flush=True);continue
            outcome = session.execute_skill('place', microwave_place_options())
            if outcome_field(outcome,'success',False) or 'support_not_visible' not in str(outcome_field(outcome,'reason','')):break
    if not outcome_field(outcome,'success',False) and 'unreachable' in str(outcome_field(outcome,'reason','')):
        # The qualified placements (docs/qwen_household_results.md: 0.60 m centred
        # dock, 40 degree tilt) had the measured cavity-floor centre 0.80-0.85 m
        # ahead and passed on their last few candidates; at 0.86 m every pose
        # was out of reach (fix104). The remembered door plane the 0.60 m is
        # measured from is several cm off, so stand by the floor centre this
        # attempt measured instead and place once more.
        distance=centred_redock_distance(session)
        if distance is None:distance=oracle_microwave_distance(session)
        if distance is not None:
            # Centred first; when that spot is refused (the open door beside
            # the arm), the same distance a little away from the door leaf.
            side=away_from_open_leaf(session.aperture('microwave',map_frame=True))
            for shift in MICROWAVE_REDOCK_SHIFTS:
                print('MICROWAVE_REACH_REDOCK',json.dumps(dict(redock_distance_m=distance,lateral_offset_m=side*shift)),flush=True)
                try:
                    session.redock(target='microwave',purpose='place',distance=distance,lateral_offset=side*shift)
                except RuntimeError as exc:
                    print('MICROWAVE_REACH_REDOCK_REFUSED',str(exc)[:160],flush=True);continue
                outcome = session.execute_skill('place', microwave_place_options())
                break
    placed = getattr(session, 'last_placement_output', None)
    if placed is not None:
        save_memory(placed, load_memory(previous))
    return outcome, placed
