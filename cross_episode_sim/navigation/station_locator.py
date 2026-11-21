"""Which station room the robot is in, from its own observed map.

A transit used to label the start of every route by a guess ("going to the
kitchen means coming from the dining table") unless the caller passed
--source-station. Set table never passes it, so a second kitchen transit was
recorded as a route from the dining table, putting a "dining table" station in
the kitchen (argtest 2026-11-21). User decision: decide the current room from
the map instead.

The robot is "at" a station when it is within AT_STATION_RADIUS along free
space of the frozen observed map of the free floor around the station object
exploration located (kitchen counter / dining table bounds + STATION_ZONE), and
clearly nearer to it than to the other. Otherwise the room is unknown (a third room, a hallway, the far side
of a large room). Wall-based room footprints are off (USE_ROOM_FOOTPRINTS).
Sensor estimates only: SLAM pose, observed map, semantic memory. No simulator
room truth.
"""
import numpy as np

STATIONS = ('kitchen', 'dining table')
# "At" a station: within this path distance (m) on the observed map of where
# exploration located it. Kitchen and dining table are at least 6 m apart in
# these houses, so the two radii never overlap. Anywhere else (a third room, a
# hallway, the far side of a large kitchen) is "unknown": the transit then drives
# but records no station label. A short extra drive is the worst case.
AT_STATION_RADIUS = 2.5
# Station zone: free floor within this distance (m) of the observed object's bounds.
STATION_ZONE = 1.2
# When the robot's own cell is inside the planning margin, start from the nearest free cell within this distance (m).
START_SEARCH = .6
# The nearer station must also be clearly nearer.
DECISIVE_RATIO = .6
USE_ROOM_FOOTPRINTS = False


def station_cells(policy, memory, label):
    """Free floor cells within STATION_ZONE of the observed object (its bounds or position)."""
    import numpy as np
    zone = []
    for item in memory.get('furniture', []):
        if item.get('label') != label or item.get('observations', 0) < 2:continue
        bounds = np.asarray(item.get('observed_bounds') or [item['position'], item['position']], float)[:, :2]
        lo, hi = bounds.min(axis=0) - STATION_ZONE, bounds.max(axis=0) + STATION_ZONE
        (r0, c0), (r1, c1) = policy.cell(lo), policy.cell(hi)
        rr, cc = np.mgrid[max(0, min(r0, r1)):min(policy.free.shape[0], max(r0, r1) + 1),
                          max(0, min(c0, c1)):min(policy.free.shape[1], max(c0, c1) + 1)]
        keep = policy.free[rr, cc]
        zone.extend(zip(rr[keep].tolist(), cc[keep].tolist()))
    return zone


def free_start(policy, xy):
    """The robot's cell, or the nearest free cell within START_SEARCH when it stands in the safety margin."""
    row, col = policy.cell(np.asarray(xy, float));r = int(np.ceil(START_SEARCH / policy.grid.config.resolution))
    best = None
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            rr, cc = row + dr, col + dc
            if 0 <= rr < policy.free.shape[0] and 0 <= cc < policy.free.shape[1] and policy.free[rr, cc]:
                if best is None or dr * dr + dc * dc < best[0]:best = (dr * dr + dc * dc, (rr, cc))
    return None if best is None else best[1]


def station_path_costs(policy, xy, memory):
    """Path length (m) on the observed map from the robot to the free floor around each observed station object."""
    from .house_exploration import reachable_paths
    distance, _ = policy.layers(np.asarray(xy, float))
    start = free_start(policy, xy)
    if start is None:return {label: np.inf for label in STATIONS}
    costs, _ = reachable_paths(policy.free, start, distance)
    result = {}
    for label in STATIONS:
        cells = station_cells(policy, memory, label)
        result[label] = min((float(costs[r, c]) for r, c in cells), default=np.inf) * policy.grid.config.resolution
    return result


def straight_costs(xy, memory):
    """Straight-line distance (m) to each observed station object's position; only when no map path exists."""
    return {label: min((float(np.linalg.norm(np.asarray(item['position'], float)[:2] - np.asarray(xy, float)[:2]))
                        for item in memory.get('furniture', []) if item.get('label') == label and item.get('observations', 0) >= 2),
                       default=np.inf) for label in STATIONS}


def station_rooms(grid, memory):
    """Room footprint around each located station, from observed tall walls and
    doorway necks (observed_room_topology.py). Conservative: a spot near a door
    can fall outside every footprint."""
    from ..mapping.furniture import furniture_viewpoints
    from .observed_room_topology import ObservedRoomTopology
    rooms = {}
    for label in STATIONS:
        points = furniture_viewpoints(memory, label)
        if not points:continue
        topology = ObservedRoomTopology();topology.update(grid, np.r_[np.asarray(points[0], float)[:2], 0.])
        footprint = topology.rooms.get(topology.active, {}).get('footprint') or frozenset()
        if footprint:rooms[label] = footprint
    # Two stations in one enclosure cannot tell rooms apart: do not use it.
    if len(rooms) == 2 and rooms['kitchen'] & rooms['dining table']:return {}
    return rooms


def room_station(grid, xy, memory):
    resolution = grid.config.resolution
    cell = tuple(np.floor(np.asarray(xy, float)[:2] / resolution + 1e-7).astype(int))
    rooms = station_rooms(grid, memory)
    inside = [label for label, footprint in rooms.items() if cell in footprint]
    return (inside[0] if len(inside) == 1 else None), {label: round(len(f) * resolution ** 2, 1) for label, f in rooms.items()}


def current_station(policy, xy, memory):
    """(station or None, evidence). The station's room footprint from observed
    walls first; otherwise within AT_STATION_RADIUS along the map; else None."""
    # Room footprints from observed walls (station_rooms) are not used:
    # on the argtest maps they missed the kitchen room in one snapshot and
    # merged kitchen and dining into one 48 m2 room in another (2026-11-21).
    # Kept for the saved-map / memory work; USE_ROOM_FOOTPRINTS enables them.
    areas = None
    if USE_ROOM_FOOTPRINTS:
        try:
            station, areas = room_station(policy.grid, xy, memory)
            if station is not None:
                return station, dict(basis='observed_room_footprint', room_areas_m2=areas, station=station,
                                     position=[round(float(x), 3) for x in np.asarray(xy)[:2]])
        except Exception as exc:areas = f'rooms_unavailable:{type(exc).__name__}'
    try:costs = station_path_costs(policy, xy, memory);basis = 'observed_map_path'
    except Exception as exc:costs = {};basis = f'path_unavailable:{type(exc).__name__}'
    if not costs or all(not np.isfinite(v) for v in costs.values()):
        costs = straight_costs(xy, memory);basis = 'straight_line_no_map_path'
    ranked = sorted(costs.items(), key=lambda kv: kv[1])
    (near, d0), (_, d1) = ranked[0], ranked[1]
    station = near if (np.isfinite(d0) and d0 <= AT_STATION_RADIUS
                       and (not np.isfinite(d1) or d0 <= DECISIVE_RATIO * d1)) else None
    return station, dict(basis=basis, costs_m={k: (round(v, 2) if np.isfinite(v) else None) for k, v in costs.items()},
                         room_areas_m2=areas, station=station, position=[round(float(x), 3) for x in np.asarray(xy)[:2]])
