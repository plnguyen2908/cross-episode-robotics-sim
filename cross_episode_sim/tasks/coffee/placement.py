"""Coffee apparatus placement along the kitchen's back counters.

The coffee workflow was validated with the machine mounted at VALIDATED_MOUNT.
A placement shifts the whole apparatus (machine, portafilter, grounds box, lid,
mug and the robot's starting dock) by one offset in the counter plane. Every
back counter faces -y, so the shifted episode is a rigid copy of the validated
one: arm motions relative to the machine are unchanged and only the
surroundings differ.
"""
import numpy as np

# Machine mount the coffee workflow was validated at.
VALIDATED_MOUNT = np.array([2.1373, -.245, .922])

# Collision extent of the apparatus in x relative to the mount: the machine
# (-0.122 to +0.117) through the mug spot at its right (+0.35).
APPARATUS_X = (-.122, .35)

# Offsets of the mount in x that keep the apparatus on a free counter stretch:
# the main counter between the sink and the stove (with the paper towel holder
# removed), and the right counter between the stove and the fridge, keeping the
# mug at least 15 cm from the fridge so the arm can still plan to it.
SLOTS = {
    'main_counter': (-.25, .22),
    'right_counter': (1.53, 1.60),
}

# Fixtures that may be removed to clear a slot, with their x half-widths; none is
# used by the coffee or breakfast tasks.
REMOVABLE_FIXTURES = {
    'paper_towel_main_group_main': .094,
    'toaster_main_group_main': .08,
    'knife_block_main_group_main': .06,
}

# The unused blender lid is parked on the free left stretch of the main counter.
LID_PARKING = (.50, -.50, .96)

# World-frame entries of a coffee spec, as (key, number of coordinates shifted).
SPEC_POINTS = (('lid_anchor', 2), ('dispenser_target', 2), ('hopper_xy', 2),
               ('lid_parking_xy', 2), ('dosing_parking_xy', 2))


def counter_for(off):
    """The counter body under the shifted apparatus."""
    x = VALIDATED_MOUNT[0] + float(off[0])
    return 'counter_right_main_group_main' if x > 3.2 else 'counter_main_main_group_main'


def sample_offset(rng, slot=None):
    """A machine offset [dx, dy] in a named or random slot."""
    names = sorted(SLOTS)
    name = slot if slot is not None else names[int(rng.integers(len(names)))]
    low, high = SLOTS[name]
    return name, np.array([float(rng.uniform(low, high)), 0.])


def offset(spec):
    """The spec's machine offset as a 3-vector (zero when unplaced)."""
    return np.array([*spec.get('machine_offset', (0., 0.)), 0.])


def shift(point, off):
    """Shift a world point (2D or 3D) by the machine offset."""
    point = np.array(point, dtype=float)
    point[:2] += np.asarray(off)[:2]
    return point


def shift_spec(spec, off):
    """Shift the world-frame entries a base episode's coffee spec carries."""
    for key, count in SPEC_POINTS:
        if key in spec:
            value = np.array(spec[key], dtype=float)
            value[:count] += np.asarray(off)[:count]
            spec[key] = value.tolist()
    if 'machine_x' in spec:
        spec['machine_x'] = float(spec['machine_x']) + float(off[0])
    spec['machine_offset'] = [float(off[0]), float(off[1])]


def blocking_fixtures(root, off, margin=.03):
    """Removable fixtures overlapping the shifted apparatus in x."""
    low = VALIDATED_MOUNT[0] + float(off[0]) + APPARATUS_X[0] - margin
    high = VALIDATED_MOUNT[0] + float(off[0]) + APPARATUS_X[1] + margin
    blocked = []
    for name, half in REMOVABLE_FIXTURES.items():
        node = root.find(f".//body[@name='{name}']")
        if node is None:
            continue
        x = float(np.fromstring(node.get('pos', '0 0 0'), sep=' ')[0])
        if x + half > low and x - half < high:
            blocked.append(name)
    return blocked
