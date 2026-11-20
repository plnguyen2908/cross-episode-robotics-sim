"""Transit to observed station endpoints before local household manipulation.

Furniture centroids are not parking poses. Reuse the episode route graph, or
connect to its observed station endpoints with the existing frozen-map A*.
The qualified set-table navigator remains unchanged.

Exploration records a station where its furniture was recognised from, which
can be across the room. A loaded carry that arrives there continues on the
same frozen map to a stand-off at the observed furniture and turns to view it.
"""
import math

import numpy as np

from .fixed_map import FixedMapNavigator, FixedRoomWaypointFollower
from .house_exploration import reachable_paths
from .object_view import visible_object_heading
from .route_memory import SIGHTING

# Arrival this close to a sighting entry is arrival at that entry.
SIGHTING_RADIUS = .5
FACING_SECONDS = 12.


class HouseholdStationNavigator(FixedMapNavigator):
    def step(self, packet, tracking, now):
        if self.terminal:
            return self.follower._stop(self.terminal)
        if getattr(self, 'facing', None) is not None:
            return self._face(packet, tracking, now)
        if not self.cache_attempted and tracking.accepted:
            from .route_memory import episode_routes
            from ..runtime.episode import current_runtime
            memory = episode_routes()
            runtime = current_runtime()
            stations = [] if memory is None else memory.stations.get(self.destination, [])
            if stations and runtime is not None and runtime.mapper is not None:
                anchor = runtime.mapper.last_result.T_map_base @ np.linalg.inv(tracking.T_map_base)
                inverse = np.linalg.inv(anchor)
                follower = FixedRoomWaypointFollower(self.grid, self.route_config)
                follower.set_destinations([(inverse @ row['pose'])[:2, 3] for row in stations])
                follower.route_memory = memory
                follower.route_destination = self.destination
                follower.route_anchor = anchor
                self.cached_transit = follower
                self.cache_attempted = True
                follower.events.append(dict(event='observed_station_transit',
                    destination=self.destination, endpoints=len(stations)))
        command = super().step(packet, tracking, now)
        if self.terminal != 'known_destination_reached' or getattr(self, 'cached_transit', None) is None:
            return command
        if getattr(self, 'standoff_target', None) is not None:
            self.terminal = None
            self.facing = dict(started=now, heading=None)
            return self.follower._stop('station_standoff_reached')
        return self._continue_past_sighting(packet, tracking, now, command)

    def _continue_past_sighting(self, packet, tracking, now, command):
        """Replace arrival at a sighting with travel to the furniture seen from it.

        Every refusal keeps the arrival as it was; no status here asks the
        caller for a fresh observation.
        """
        first = self.cached_transit
        memory = getattr(first, 'route_memory', None)
        if memory is None or not tracking.accepted:
            return command
        xy = tracking.T_map_base[:2, 3].copy()
        inverse = np.linalg.inv(first.route_anchor)
        if not any(row.get('kind') == SIGHTING and
                   np.linalg.norm((inverse @ row['pose'])[:2, 3]-xy) <= SIGHTING_RADIUS
                   for row in memory.stations.get(self.destination, [])):
            return command
        targets = sorted((t for t in self.targets.targets if t.label == self.target_label),
                         key=lambda t: np.linalg.norm(t.position[:2]-xy))
        # Already within the navigation stand-off band of its furniture.
        if not targets or np.linalg.norm(targets[0].position[:2]-xy) <= self.follower.max_distance+.08:
            return command
        follower = FixedRoomWaypointFollower(self.grid, self.route_config)
        clearance, _ = follower.layers(xy)
        costs, _ = reachable_paths(follower._search_free(xy), follower.cell(xy), clearance)
        nodes = np.argwhere(np.isfinite(costs))
        chosen = None
        if len(nodes):
            points = follower.world(nodes)
            travel = costs[nodes[:, 0], nodes[:, 1]]*self.grid.config.resolution
            for target in targets:
                separation = np.linalg.norm(points-target.position[:2], axis=1)
                eligible = np.flatnonzero((separation >= .48) & (separation <= self.follower.max_distance))
                # Same ranking as the empty-hand target stand-off.
                order = np.argsort(travel[eligible]+6*np.abs(separation[eligible]-self.follower.preferred_distance))
                for index in eligible[order][:40]:
                    if visible_object_heading(tracking.T_map_base, target.position,
                                              packet.cameras['navigation'], points[index]) is None:
                        continue
                    chosen = target, points[index].copy(), float(separation[index]), float(travel[index])
                    break
                if chosen:
                    break
        error = 'no_observed_standoff'
        if chosen:
            follower.set_destinations([chosen[1]])
            error = follower._plan(xy, now)
        if error is not None:
            first.events.append(dict(time=now, event='station_sighting_accepted', destination=self.destination,
                                     reason=error, position=xy.tolist()))
            self.route_events = list(first.events)
            return command
        target, point, separation, travel = chosen
        follower.last_time = now
        follower.events[:0] = list(first.events)+[dict(time=now, event='continue_past_station_sighting',
            destination=self.destination, sighting=xy.tolist(), furniture=target.position[:2].tolist(),
            standoff=point.tolist(), separation_m=separation, travel_m=travel)]
        self.cached_transit = follower
        self.standoff_target = target
        self.route_events = list(follower.events)
        self.terminal = None
        return self.follower._stop('station_sighting_reached')

    def _face(self, packet, tracking, now):
        """Turn in place at the stand-off until the mast camera views the furniture."""
        follower = self.cached_transit
        if not tracking.accepted:
            return follower._stop('tracking_lost')
        pose = tracking.T_map_base
        xy = pose[:2, 3].copy()
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        state = self.facing
        if state['heading'] is None:
            target = self.standoff_target.position
            # Offset from the bearing when the carried hand would hide the target.
            heading = visible_object_heading(pose, target, packet.cameras['navigation'])
            state['heading'] = math.atan2(target[1]-xy[1], target[0]-xy[0]) if heading is None else heading
        error = math.atan2(math.sin(state['heading']-yaw), math.cos(state['heading']-yaw))
        result = None
        if abs(error) < .08:
            result = 'viewing'
        elif now-state['started'] > FACING_SECONDS:
            result = 'timeout'
        else:
            desired = float(np.clip(1.8*error, -.4, .4))
            previous = follower.velocity[1]
            turn = np.array([0., previous+np.clip(desired-previous, -.016, .016)])
            follower.layers(xy)
            follower._depth(packet.cameras['navigation'])
            if follower._command_clear(xy, yaw, turn):
                follower.velocity = turn
                return dict(status='face_station_furniture', forward=0., yaw=float(turn[1]))
            result = 'clearance_stop'
        follower.events.append(dict(time=now, event='station_furniture_view', result=result,
                                    heading_error=float(error), seconds=float(now-state['started'])))
        self.route_events = list(follower.events)
        self.terminal = 'known_destination_reached'
        return follower._stop(self.terminal)
