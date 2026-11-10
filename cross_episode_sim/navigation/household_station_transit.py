"""Transit to observed station endpoints before local household manipulation.

Furniture centroids are not parking poses. Reuse the episode route graph, or
connect to its observed station endpoints with the existing frozen-map A*.
The qualified set-table navigator remains unchanged.
"""
import numpy as np

from .fixed_map import FixedMapNavigator, FixedRoomWaypointFollower


class HouseholdStationNavigator(FixedMapNavigator):
    def step(self, packet, tracking, now):
        if self.terminal:
            return self.follower._stop(self.terminal)
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
        return super().step(packet, tracking, now)
