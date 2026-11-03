"""Experimental TARE goal selection with sensor-checked ground execution.

TARE selects the exploration destination. The local follower only routes to
that destination through observed floor; it never substitutes its own frontier.
"""
import math
import numpy as np
from .house_exploration import HouseExplorer, HouseExplorationConfig, reachable_paths


class TareExplorer(HouseExplorer):
    def __init__(self, grid, planner, config=HouseExplorationConfig()):
        super().__init__(grid, config)
        self.planner = planner
        self.response = {}
        self.planner_timestamp = None
        self.sensor_offset = np.zeros(2)
        self.no_goal_since = None
        self.wait_observation = None

    def step(self, packet, tracking, now):
        self._sync_support()
        if self.started is not None and now-self.started >= self.config.max_seconds:
            self.terminal = 'budget_exhausted'
        if self.terminal:
            return self._stop(self.terminal)
        if tracking.accepted and packet.timestamp != self.planner_timestamp:
            self.response = self.planner.update(packet.cameras['navigation'], tracking)
            self.planner_timestamp = packet.timestamp
            pose = tracking.T_map_base
            optical = pose @ packet.cameras['navigation'].T_base_optical
            self.sensor_offset = optical[:2, 3]-pose[:2, 3]
        # A missing TARE path is never replaced by our custom frontier selector.
        if self.phase == 'travel' and not self.path and not self.terminal:
            if packet.timestamp == self.wait_observation:
                if self.no_goal_since is not None and now-self.no_goal_since > 20:
                    self.terminal = 'tare_no_path_timeout'
                self.last_time = now
                return self._stop(self.terminal or 'awaiting_tare_path')
            waypoint = self.response.get('waypoint')
            # Include grid-centre rounding when deciding whether this is an
            # already-reached lookahead. TARE publishes its next waypoint at 1Hz.
            arrival = .3 + self.grid.config.resolution/math.sqrt(2)
            at_current = (tracking.accepted and waypoint is not None and
                np.linalg.norm(np.asarray(waypoint[:2])-self.sensor_offset-tracking.T_map_base[:2, 3]) < arrival)
            if not self.response.get('sequence') or waypoint is None or at_current:
                if self.no_goal_since is None:
                    self.no_goal_since = now
                if now-self.no_goal_since > 20:
                    self.terminal = 'tare_no_path_timeout'
                self.last_time = now
                return self._stop(self.terminal or 'awaiting_tare_path')
        result = super().step(packet, tracking, now)
        if result['status'] == 'tare_goal_at_current_position':
            # A reachable grid centre may also snap toward us near obstacles.
            # This is a transient handoff, not a completed/failed exploration.
            self.terminal = None
            self.path = []
            self.wait_observation = packet.timestamp
            if self.no_goal_since is None:
                self.no_goal_since = now
            if now-self.no_goal_since > 20:
                self.terminal = 'tare_no_path_timeout'
            return self._stop(self.terminal or 'awaiting_tare_path')
        if result['status'] == 'stalled_replan':
            # The shared follower has already attempted local replanning.
            # Do not reset its progress clock on every identical TARE goal.
            self.terminal = 'tare_local_stalled'
            return self._stop(self.terminal)
        if result['status'] == 'scan_complete':
            self.progress_time = now
            self.last_progress = tracking.T_map_base[:2, 3].copy()
        if result['status'] in ('travel', 'scanning'):
            self.no_goal_since = None
        if result['status'] == 'arrived_frontier' and self.scan_locations:
            # TARE's waypoint is a short lookahead along its path, not necessarily
            # a new frontier/viewpoint. Do not spin after every lookahead point.
            xy = tracking.T_map_base[:2, 3]
            nearby = np.linalg.norm(xy-self.scan_locations[-1]) < 2.
            clearance_ok, clearance, required = self._scan_clearance(xy)
            if nearby or not clearance_ok:
                self.phase = 'travel'
                result['status'] = 'tare_waypoint_reached'
                if not nearby:
                    self.events.append(dict(time=now, event='scan_deferred_for_clearance',
                        clearance=clearance, required=required, position=xy.tolist()))
        return result

    def _lookahead_reached(self, xy):
        # TARE lookaheads are routing guides, not precise parking poses. Once
        # inside the existing waypoint tolerance, request the next guide instead
        # of orbiting a redundant intermediate corner. Full scans still require
        # their separate clearance check, and every next route starts at the
        # measured SLAM pose; no point or room is marked visited by this hook.
        return self.goal is not None and np.linalg.norm(self.goal-xy)<self.config.waypoint_tolerance

    def _scan_clearance(self, xy):
        # A periodic full turn needs more margin than following a route.
        # Unknown floor is not treated as free, and live dense depth remains
        # authoritative during the turn. This is not a full 3D certificate.
        distance, _ = self.layers(xy)
        c = self.grid.config
        required = (self.scanner.config.radius + self.scanner.config.margin
                    + c.localization_margin + c.resolution/math.sqrt(2))
        clearance = float(distance[self.cell(xy)])
        return clearance >= required, clearance, required

    def _plan(self, xy, now):
        self.last_plan = now
        self.plans += 1
        distance, _ = self.layers(xy)
        costs, parents = reachable_paths(self.free, self.cell(xy), distance)
        nodes = np.argwhere(np.isfinite(costs))
        if not len(nodes):
            return 'departure_blocked'
        sensor_goal = np.asarray(self.response['waypoint'][:2])
        requested = sensor_goal-self.sensor_offset
        points = self.world(nodes)
        gaps = np.linalg.norm(points-requested, axis=1)
        best = int(np.argmin(gaps))
        # Limited snapping to a reachable grid centre, not an alternate goal.
        if gaps[best] > .6:
            self.events.append(dict(time=now, event='tare_goal_unreachable',
                                    requested=requested.tolist(), nearest_gap=float(gaps[best])))
            return 'tare_goal_outside_observed_reachable_space'
        node = tuple(nodes[best])
        self.goal = points[best]
        if np.linalg.norm(self.goal-xy) < .3:
            return 'tare_goal_at_current_position'
        reverse = [node]
        while node in parents:
            node = parents[node]; reverse.append(node)
        remaining = list(self.world(reverse[::-1]))
        simplified = [xy.copy()]
        while remaining:
            j = len(remaining)-1
            while j >= 0 and not self._line_free(simplified[-1], remaining[j]):
                j -= 1
            if j < 0:
                return 'start_connector_blocked'
            simplified.append(remaining[j]); remaining = remaining[j+1:]
        self.path = list(simplified[1:])
        if self.last_progress is None:
            self.last_progress = xy.copy()
            self.progress_time = now
        self.events.append(dict(time=now, event='tare_goal', sensor_goal=sensor_goal.tolist(),
            requested_base_goal=requested.tolist(), goal=self.goal.tolist(),
            sequence=self.response['sequence'], route=[p.tolist() for p in simplified]))
        return None
