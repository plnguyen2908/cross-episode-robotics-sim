"""Discover kitchen work surfaces in the loaded scene, not a fixed fixture list."""
import re

import mujoco
import numpy as np

from cross_episode_sim.tasks.breakfast.scene import bounds, support_bounds
from cross_episode_sim.manipulation.edge_access import object_bodies


def box_gap(low, high, other_low, other_high):
    """Euclidean separation of two axis-aligned boxes (zero for overlap)."""
    return float(np.linalg.norm(np.maximum(np.maximum(other_low-high, low-other_high), 0.)))


def filling_candidates(model, data, obj, surfaces, reserved=()):
    """Rank supported edge sites across the kitchen by usable surrounding space.

    A geometric prefilter for empty-vessel gathering, not a replacement for
    loaded A*/cuRobo. Keep room around the hand and an open approach in front.
    """
    own = object_bodies(model, obj)
    low, high = bounds(model, data, obj)
    origin = data.body(obj).xpos
    lower, upper = low-origin, high-origin
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    obstacles = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid]); name = model.body(bid).name or ''
        if bid in own or name.startswith('robot_') or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        vertices = data.geom_xpos[gid] + (model.geom_aabb[gid, :3] + corners*model.geom_aabb[gid, 3:]) @ data.geom_xmat[gid].reshape(3, 3).T
        obstacles.append((bid, vertices.min(0), vertices.max(0)))
    result = []
    groups = model.geom_group.copy()
    try:
        for site, surface in surfaces.items():
            if surface['kind'] in ('stove', 'cooktop') and not stove_is_off(model, data, surface['body']):
                continue
            support = object_bodies(model, surface['body'])
            slo, shi = np.asarray(surface['low']), np.asarray(surface['high'])
            top = shi[2]
            # Include countertop-attached appliances above the slab as obstacles.
            nearby = [(lo, hi) for bid, lo, hi in obstacles
                      if hi[2] > top+.012 and lo[2] < top+(upper[2]-lower[2])+.22]
            nearby += [(np.asarray(lo), np.asarray(hi)) for lo, hi in reserved]
            base_obstacles = [(lo, hi) for _, lo, hi in obstacles if hi[2] > .08 and lo[2] < .55]
            model.geom_group[:] = 5
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] in support and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                    model.geom_group[gid] = 4
            # Consider all sides of islands/tables. Fixed counters use their front.
            directions = [surface['outward']]
            if surface['kind'] in ('island', 'table'):
                directions += [d for d in ([1.,0.],[-1.,0.],[0.,1.],[0.,-1.]) if d != surface['outward']]
            for outward in directions:
                outward = np.asarray(outward); axis = int(np.argmax(np.abs(outward))); tangent = 1-axis
                for fraction in np.linspace(.08, .92, 29):
                    for inset in (.035, .065, .095):
                        point = (slo+shi)/2 - (lower+upper)/2
                        point[tangent] = slo[tangent]+fraction*(shi[tangent]-slo[tangent])-(lower[tangent]+upper[tangent])/2
                        point[axis] = shi[axis]-inset-upper[axis] if outward[axis]>0 else slo[axis]+inset-lower[axis]
                        point[2] = top+.002-lower[2]
                        lo, hi = lower+point, upper+point
                        if np.any(lo[:2] < slo[:2]+.008) or np.any(hi[:2] > shi[:2]-.008):continue
                        hand_clearance = min((box_gap(lo[:2], hi[:2], a[:2], b[:2]) for a,b in nearby), default=1.)
                        if hand_clearance < .10:continue
                        dock = point[:2].copy()
                        edge = shi[axis] if outward[axis]>0 else slo[axis]
                        dock[axis] = edge+outward[axis]*.45
                        # Broad preference for an open base/withdrawal lane.
                        # Actual robot footprint and swept path are checked by A*.
                        lane_clearance = min((box_gap(dock, dock, a[:2], b[:2]) for a,b in base_obstacles), default=1.)
                        if lane_clearance < .40:continue
                        supported = True
                        for x in np.linspace(lo[0]+.002, hi[0]-.002, 3):
                            for y in np.linspace(lo[1]+.002, hi[1]-.002, 3):
                                distance = mujoco.mj_ray(model, data,
                                    np.array([x,y,top+.10]), np.array([0.,0.,-1.]),
                                    np.array([0,0,0,0,1,0], dtype=np.uint8), True, -1, None)
                                if distance < 0 or abs(distance-.10) > .006:
                                    supported = False; break
                            if not supported:break
                        if not supported:continue
                        score = min(hand_clearance,.60) + .3*min(lane_clearance,.60) - .2*inset
                        # Prefer a clear counter/table when space is comparable;
                        # an off stovetop remains a valid less-preferred surface.
                        if surface['kind'] in ('stove','cooktop'):score -= .08
                        result.append(dict(site=site, position=point.tolist(), score=score,
                            hand_clearance_m=hand_clearance, approach_clearance_m=lane_clearance,
                            outward=outward.tolist(), footprint=[lo.tolist(),hi.tolist()]))
    finally:
        model.geom_group[:] = groups
    return sorted(result, key=lambda c:(-c['score'],c['site'],c['position']))


def allocate_filling_targets(model, data, manifest, surfaces):
    """Reserve open gathering spots without moving initial objects or serving goals."""
    reserved = []
    choices = {}
    for info in manifest['bindings']:
        if info.get('initial_source') == 'kitchen':
            # Already gathered vessels stay put and reserve their actual footprint.
            reserved.append(bounds(model,data,info['body']))
            info['filling_site'] = 'kitchen'
            info['filling_position'] = data.body(info['body']).xpos.tolist()
    for info in manifest['bindings']:
        if info.get('initial_source') == 'kitchen':continue
        candidates = filling_candidates(model, data, info['body'], surfaces, reserved)
        if not candidates:
            raise ValueError(f'No open kitchen filling site with hand/approach clearance for {info["role"]}')
        chosen = candidates[0]
        info['filling_site'] = chosen['site']; info['filling_position'] = chosen['position']
        reserved.append(tuple(np.asarray(v) for v in chosen['footprint']))
        choices[info['role']] = dict(chosen, candidates=len(candidates))
    manifest['filling_target_policy'] = 'rank all kitchen surfaces by hand and approach clearance before navigation'
    manifest['filling_target_selection'] = choices
    return choices


def surface_kind(name):
    tokens = set(re.split(r'[_\W]+', name.lower()))
    return next((kind for kind in ('stove', 'cooktop', 'counter', 'island', 'table')
                 if kind in tokens), None)


def stove_is_off(model, data, fixture='stove_main_group_main'):
    """Read knob state in live or frozen scene exports; no thermal simulation."""
    from cross_episode_sim.fixtures.stove_knob import burner_on
    root = model.body(fixture).id
    knobs = []
    for bid in range(1, model.nbody):
        ancestor = bid
        while ancestor and ancestor != root:
            ancestor = model.body_parentid[ancestor]
        name = model.body(bid).name or ''
        if ancestor == root and 'knob' in name and any(
                part in name for part in ('front_left', 'front_right', 'rear_left', 'rear_right')):
            knobs.append(bid)
    if not knobs:
        return False
    for bid in knobs:
        if model.body_jntnum[bid]:
            first = model.body_jntadr[bid]
            angles = [float(data.qpos[model.jnt_qposadr[j]])
                      for j in range(first, first + model.body_jntnum[bid])]
        else:
            angles = [2 * np.arccos(np.clip(abs(model.body_quat[bid, 0]), 0., 1.))]
        if any(burner_on(angle) for angle in angles):
            return False
    return True


def discover_filling_surfaces(model, data, kitchen_support, kitchen_floor='floor_room_main'):
    """Find static work fixtures on the kitchen floor, including rotated islands.

    Fixture semantics identify work surfaces; collision geometry determines their
    extent. Candidate footprints, support rays and motion planning run later.
    Other-room furniture and object/appliance tops are not kitchen work surfaces.
    """
    floor_low, floor_high = bounds(model, data, kitchen_floor)
    found = {}
    for bid in range(1, model.nbody):
        body = model.body(bid).name or ''
        kind = surface_kind(body)
        if not kind or model.body_parentid[bid] != 0 or model.body_jntnum[bid]:
            continue
        centre = data.xpos[bid, :2]
        if np.any(centre < floor_low[:2]) or np.any(centre > floor_high[:2]):
            continue
        try:
            low, high = support_bounds(model, data, body)
        except ValueError:
            continue
        if not .55 <= high[2] <= 1.2 or np.any(high[:2] - low[:2] < .12):
            continue
        # Preserve readable aliases for existing scene logs, not eligibility.
        aliases = {'stove_main_group_main': 'kitchen_stove',
                   'counter_right_main_group_main': 'kitchen_right_counter'}
        site = 'kitchen' if body == kitchen_support else aliases.get(body, 'kitchen_surface_' + body)
        outward = data.xmat[bid].reshape(3, 3)[:2, :2] @ np.array([0., -1.])
        # Placement support bounds are world AABBs; four cardinal approach
        # sides are considered below and physical rays reject empty corners.
        axis = int(np.argmax(np.abs(outward)))
        direction = np.zeros(2); direction[axis] = np.sign(outward[axis])
        found[site] = dict(body=body, kind=kind, outward=direction.tolist(),
                           low=low.tolist(), high=high.tolist())
    return found


def register_filling_surfaces(controller):
    """Register distinct physical supports without merging contact identities."""
    c = controller
    floor = c.manifest.get('kitchen_floor', 'floor_room_main')
    c.filling_surface_info = discover_filling_surfaces(c.model, c.data, c.task_supports['kitchen'], floor)
    c.filling_sites = ['kitchen']
    for site, surface in c.filling_surface_info.items():
        body = surface['body']
        c.task_supports[site] = body
        c.task_outward[site] = surface['outward']
        c.table_bids[body] = c.descendants(body)
        c.placement_history.setdefault(body, [])
        if site != 'kitchen':
            c.filling_sites.append(site)
    c.manifest['filling_surfaces'] = dict(c.filling_surface_info)
    c.report['filling_surfaces'] = dict(c.filling_surface_info)
