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


# Validated robot dock in front of the machine: (x, y, yaw).
VALIDATED_DOCK = (2.12, -.94, np.pi/2)

# Front edge of the counter in the validated layout. On other surfaces the
# machine is placed this far behind the chosen edge, facing out over it.
VALIDATED_FRONT_EDGE_Y = -.65


def machine_pose(spec):
    """The spec's placement relative to the validated layout: dx, dy, dz, yaw (rad).

    The yaw turns the apparatus about the validated mount; dx, dy, dz then move
    the mount. Specs from the kitchen-slot placement carry only machine_offset.
    """
    pose = spec.get('machine_pose')
    if pose is not None:
        return (float(pose['dx']), float(pose['dy']), float(pose['dz']),
                np.radians(float(pose['yaw_deg'])))
    dx, dy = spec.get('machine_offset', (0., 0.))
    return float(dx), float(dy), 0., 0.


def transform(spec):
    """4x4 transform from validated-layout world coordinates to this placement."""
    dx, dy, dz, yaw = machine_pose(spec)
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    result[:3, 3] = VALIDATED_MOUNT + [dx, dy, dz] - result[:3, :3] @ VALIDATED_MOUNT
    return result


def to_world(spec, value):
    """Map a validated-layout point (2D or 3D) or 4x4 pose into this placement."""
    value = np.array(value, dtype=float)
    t = transform(spec)
    if value.shape == (4, 4):
        return t @ value
    if value.shape == (2,):
        return (t[:2, :2] @ value) + t[:2, 3]
    return t[:3, :3] @ value + t[:3, 3]


def to_validated(spec, value):
    """Inverse of to_world: express a world point or pose in the validated layout."""
    value = np.array(value, dtype=float)
    t = np.linalg.inv(transform(spec))
    if value.shape == (4, 4):
        return t @ value
    if value.shape == (2,):
        return (t[:2, :2] @ value) + t[:2, 3]
    return t[:3, :3] @ value + t[:3, 3]


def rotate(spec, rotation):
    """Rotate a validated-layout orientation into this placement."""
    return transform(spec)[:3, :3] @ np.asarray(rotation)


def yaw_degrees(spec):
    return float(np.degrees(machine_pose(spec)[3]))


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


def place_spec(spec, pose):
    """Record a full placement and move the base episode's world-frame entries."""
    spec['machine_pose'] = dict(dx=float(pose[0]), dy=float(pose[1]), dz=float(pose[2]),
                                yaw_deg=float(np.degrees(pose[3])))
    spec['machine_offset'] = [float(pose[0]), float(pose[1])]
    for key, _ in SPEC_POINTS:
        if key in spec:
            value = np.array(spec[key], dtype=float)
            if value.shape == (3,):
                spec[key] = to_world(spec, value).tolist()
            else:
                spec[key] = to_world(spec, value[:2]).tolist()
    if 'machine_x' in spec:
        spec['machine_x'] = float(to_world(spec, [spec['machine_x'], VALIDATED_MOUNT[1]])[0])
    if 'tray_z' in spec:
        spec['tray_z'] = float(spec['tray_z']) + float(pose[2])


def surface_placement(edge_point, outward, surface_z):
    """Machine pose that sets the apparatus at a surface edge, facing outward.

    edge_point is the point on the chosen edge in front of the machine and
    outward the horizontal unit vector leaving the surface there. The validated
    machine faces -y with the counter edge VALIDATED_FRONT_EDGE_Y in front of it.
    """
    outward = np.asarray(outward, dtype=float)[:2]
    outward = outward / np.linalg.norm(outward)
    yaw = float(np.arctan2(outward[1], outward[0]) - np.arctan2(-1., 0.))
    depth = VALIDATED_MOUNT[1] - VALIDATED_FRONT_EDGE_Y
    mount = np.asarray(edge_point, dtype=float)[:2] - depth*outward
    return (float(mount[0] - VALIDATED_MOUNT[0]), float(mount[1] - VALIDATED_MOUNT[1]),
            float(surface_z - .92), yaw)


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


# Area the apparatus occupies in the validated layout: the machine through the
# mug spot in x, and from the back wall to the counter's front edge in y.
FOOTPRINT_X = (VALIDATED_MOUNT[0] + APPARATUS_X[0] - .05, VALIDATED_MOUNT[0] + APPARATUS_X[1] + .05)
FOOTPRINT_Y = (VALIDATED_FRONT_EDGE_Y, -.02)

# Coffee bodies that the workflow positions itself.
COFFEE_BODIES = ('cup_one_test_object_main', 'skill_blender_lid_main')


def loose_objects_in_footprint(root, spec, keep=()):
    """Free bodies standing where the placed apparatus goes (not coffee's own)."""
    own = set(COFFEE_BODIES) | {spec.get('dosing_body')} | set(spec.get('grains', ())) | set(keep)
    found = []
    for body in root.find('worldbody').findall('body'):
        name = body.get('name', '')
        free = body.find('freejoint') is not None or any(
            j.get('type') == 'free' for j in body.findall('joint'))
        if not free or name in own or name.startswith('robot'):
            continue
        x, y = to_validated(spec, np.fromstring(body.get('pos', '0 0 0'), sep=' ')[:2])
        if FOOTPRINT_X[0] <= x <= FOOTPRINT_X[1] and FOOTPRINT_Y[0] <= y <= FOOTPRINT_Y[1]:
            found.append(name)
    return found


# Working area the robot needs in the validated layout, relative to the mount
# in x and absolute in y: the dock and arm sweep (the arm swings further toward
# the grounds box and mug on the +x side), from just behind the dock to the
# back wall line. Walls behind the machine are fine, as at the kitchen counter.
WORKSPACE_X = (-.70, .90)
WORKSPACE_Y = (VALIDATED_DOCK[1] - .51, -.02)


def is_room_wall(name):
    """Room wall bodies, as RoboCasa names them (floors and furniture excluded)."""
    if not name.endswith('_room_main'):
        return False
    return 'wall' in name or name.split('_')[0] in ('north', 'south', 'east', 'west')


def workspace_wall_conflicts(model, data, spec):
    """Room walls entering the robot's working area at this placement."""
    import mujoco
    conflicts = set()
    low = np.array([VALIDATED_MOUNT[0] + WORKSPACE_X[0], WORKSPACE_Y[0]])
    high = np.array([VALIDATED_MOUNT[0] + WORKSPACE_X[1], WORKSPACE_Y[1]])
    for gid in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[gid])).name
        if not is_room_wall(name) or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        center = data.geom_xpos[gid]; rotation = data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid]
        # Sample the wall's footprint so long thin walls are not missed.
        for u in np.linspace(-1., 1., 41):
            for v in (-1., 0., 1.):
                for axes in ((u, v), (v, u)):
                    point = center + rotation @ (half * np.array([*axes, 0.]))
                    local = to_validated(spec, point[:2])
                    if np.all(local >= low) and np.all(local <= high):
                        conflicts.add(name)
    return sorted(conflicts)
