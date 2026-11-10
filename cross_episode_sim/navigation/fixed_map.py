"""Frozen-map transit: plan against observed geometry, without a live-depth veto."""
import hashlib
import numpy as np
from .semantic_navigation import SemanticNavigator, TargetFollower


def map_digest(grid):
    h=hashlib.sha256()
    for value in (grid.origin,grid.floor_views,grid.evidence):
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def freeze_map(grid):
    ij=np.argwhere(grid.evidence>=2)
    grid.fixed_obstacle_points=np.column_stack((
        grid.origin[0]+(ij[:,1]+.5)*grid.config.resolution,
        grid.origin[1]+(ij[:,0]+.5)*grid.config.resolution,
        grid.config.min_obstacle_height+(ij[:,2]+.5)*grid.config.height_resolution))
    for value in (grid.origin,grid.floor_views,grid.evidence,grid.fixed_obstacle_points):
        value.setflags(write=False)
    return map_digest(grid)


def live_obstacle_points(packet,tracking,stride=4):
    """Current depth endpoints only. No fusion, ray carving or persistent map."""
    if not tracking.accepted:raise ValueError('Live collision guard requires localization')
    points=[]
    for camera in packet.cameras.values():
        v,u=np.mgrid[0:camera.depth_m.shape[0]:stride,0:camera.depth_m.shape[1]:stride]
        z=camera.depth_m[::stride,::stride]
        valid=camera.valid_depth[::stride,::stride]&np.isfinite(z)&(z>0)&(z<3.)
        if camera.robot_mask is not None:valid &= ~camera.robot_mask[::stride,::stride]
        z=z[valid];u=u[valid];v=v[valid];K=camera.K
        p=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
        p=p@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
        from cross_episode_sim.sensors.onboard import without_robot_points
        p=without_robot_points(p,camera)
        p=p[(p[:,2]>.05)&(p[:,2]<2.)&(np.linalg.norm(p[:,:2],axis=1)<2.)]
        T=tracking.T_map_base;points.append(p@T[:3,:3].T+T[:3,3])
    return np.concatenate(points) if points else np.empty((0,3))


class FrozenMapDepthPolicy:
    """Depth remains available to localization, not a second transit veto.

    A* and the route follower retain the frozen map's inflated obstacles and
    braking corridor. Fresh target geometry belongs to the local docking skill.
    """
    def _depth(self, camera):
        self.near_points = np.empty((0, 3))


class FixedTargetFollower(FrozenMapDepthPolicy, TargetFollower):
    pass


class FixedMapNavigator(SemanticNavigator):
    """Travel to an already confirmed destination; no search/confirmation dwell.

    Arrival is to a remembered stand-off, not fresh perception or manipulation
    success. The following local observation/docking stage verifies the target.
    """
    def __init__(self,grid,targets,config,preferred_distance=1.1,destination='dining table'):
        if not targets:raise ValueError('Fixed-map navigation requires an observed destination')
        super().__init__(grid,None,destination,config=config,preferred_distance=preferred_distance)
        self.destination=destination
        self.follower=FixedTargetFollower(grid,config,preferred_distance)
        self.targets.targets=targets
        self.allow_exploration=False
        self.cached_transit=None;self.cache_attempted=False;self.route_events=[]
        self.route_config=config

    def observe(self,packet,tracking):
        # Do not rerun a detector or VLM while following the known route.
        return

    def step(self,packet,tracking,now):
        # Arrival remains terminal while the base physically settles. The
        # underlying frontier follower switches to scan on arrival; calling
        # it again would overwrite successful fixed-map transit with a scan error.
        if self.terminal:
            return self.follower._stop(self.terminal)
        if not getattr(self,'cache_attempted',True) and tracking.accepted:
            self.cache_attempted=True
            from .route_memory import episode_routes
            from cross_episode_sim.runtime.episode import current_runtime
            memory=episode_routes();runtime=current_runtime()
            if memory is not None and runtime.mapper is not None:
                follower=FixedRoomWaypointFollower(self.grid,self.route_config)
                follower.set_destinations([])
                follower.route_memory=memory;follower.route_destination=self.destination
                # Tracking result is stage-local; session result is continuous.
                follower.route_anchor=runtime.mapper.last_result.T_map_base @ np.linalg.inv(tracking.T_map_base)
                xy=tracking.T_map_base[:2,3];follower.layers(xy)
                if follower._cached_route(xy,now):
                    self.cached_transit=follower;self.route_events=list(follower.events)
        if getattr(self,'cached_transit',None) is not None:
            command=self.cached_transit.step(packet,tracking,now)
            self.route_events=list(self.cached_transit.events)
            if command['status']=='arrived_frontier':
                self.terminal='known_destination_reached'
                return self.follower._stop(self.terminal)
            if self.cached_transit.terminal:
                self.terminal=self.cached_transit.terminal
            return command
        command=super().step(packet,tracking,now)
        if command['status']=='at_target_standoff':
            self.terminal='known_destination_reached'
            return self.follower._stop(self.terminal)
        if command['status']=='redock_needs_observation':
            self.terminal='known_route_unavailable'
            return self.follower._stop(self.terminal)
        return command

from .hybrid_navigation import RoomWaypointFollower
from .semantic_navigation import astar_observed


class FixedRoomWaypointFollower(FrozenMapDepthPolicy, RoomWaypointFollower):
    """Choose a reachable stand-off near a visually labeled room observation.

    A room label denotes a region, not a required exact parking pose. This is
    waypoint arrival; a later task-specific observation verifies local targets.
    """
    def step(self,packet,tracking,now):
        result=super().step(packet,tracking,now)
        if (getattr(self,'following_cached_route',False) and
                result['status'] in {'local_clearance_stop','waypoint_route_changed','stalled_replan'}):
            # A sampled successful route can graze a raster corner. Reusing
            # precisely the same tiny segments at every tick cannot repair
            # that tracking failure. Replan on the unchanged map instead.
            self.cached_route_needs_repair=True;self.following_cached_route=False
            self.path=[];self.detour_turn_in_place=True
            self.events.append(dict(event='cached_route_local_repair',reason=result['status']))
        return result

    def layers(self, xy):
        # Previously occupied base footprints remain measured support, including
        # floor hidden directly under the robot. Never override obstacle cells.
        if not getattr(self, '_inherited_support_loaded', False):
            self._sync_support()
            resolution=self.grid.config.resolution
            radius=self.config.ego_support_radius
            for pose in getattr(self.grid,'inherited_support_poses',[]):
                row,col=self.cell(pose[:2]);r=int(np.ceil(radius/resolution))+1
                r0,r1=max(0,row-r),min(self.support.shape[0],row+r+1)
                c0,c1=max(0,col-r),min(self.support.shape[1],col+r+1)
                yy,xx=np.mgrid[r0:r1,c0:c1]
                points=self.grid.origin+(np.stack((xx,yy),-1)+.5)*resolution
                self.support[r0:r1,c0:c1] |= np.linalg.norm(points-pose[:2],axis=-1)<radius
            self._inherited_support_loaded=True
        return super().layers(xy)

    def _search_free(self,xy):
        # Allow only the robot's current raster cell as an A* entry. Every
        # returned segment is still checked against actual geometry below.
        free=self.free;row,col=self.cell(xy)
        if (0<=row<free.shape[0] and 0<=col<free.shape[1]
                and not free[row,col] and self._line_free(xy,xy)):
            free=free.copy();free[row,col]=True
        return free

    def departure_region(self, packet, tracking):
        """Choose departure clearance from the frozen observed costmap."""
        self.layers(tracking.T_map_base[:2,3])
        return self.destination_component().copy()

    def destination_component(self):
        """Free cells connected to an accepted station/remembered route endpoint."""
        from scipy.ndimage import label
        components, _ = label(self.free)  # conservative four-connected regions
        goals = list(self.destinations)
        memory = getattr(self, 'route_memory', None)
        if memory is not None:
            goals += [points[-1] for points, _ in memory.candidates(self.route_destination, self.route_anchor)]
        nodes = np.argwhere(self.free)
        valid = set()
        if len(nodes):
            points = self.world(nodes)
            for goal in goals:
                near = np.flatnonzero(np.linalg.norm(points-goal, axis=1) <= 1.)
                for i in near:
                    valid.add(int(components[tuple(nodes[i])]))
        return self.free & np.isin(components, list(valid))

    def _cached_route(self, xy, now):
        if getattr(self,'cached_route_needs_repair',False):return False
        memory = getattr(self, 'route_memory', None)
        if memory is None:
            return False
        def connector(start, goal):
            cells=[self.cell(start),self.cell(goal)]
            if any(not (0<=r<self.free.shape[0] and 0<=c<self.free.shape[1]) for r,c in cells):return None
            path=astar_observed(self._search_free(start),*cells)
            if not path:return None
            points=np.vstack((start,self.world(path),goal))
            if any(not self._line_free(a,b) for a,b in zip(points,points[1:])):return None
            return points
        result=memory.plan(xy,self.route_destination,self.route_anchor,self._line_free,connector)
        if result is None:return False
        route=result.pop('path')
        self._recovery_destination=np.asarray(route[-1]).copy()
        # Recorded SLAM samples can lie millimetres from an inflated-cell
        # boundary. Such a route passes a center-line check but is unstable to
        # normal localization/turn drift. Prefer cell-centered A* before driving
        # into that corner, rather than discovering it at the braking guard.
        margin=min(.015,self.grid.config.resolution*.3)
        offsets=[np.array([margin,0.]),np.array([-margin,0.]),
                 np.array([0.,margin]),np.array([0.,-margin])]
        if any(not self._line_free(a+offset,b+offset)
               for a,b in zip(route,route[1:]) for offset in offsets):
            self.cached_route_needs_repair=True;self.detour_turn_in_place=True
            self.events.append(dict(event='cached_route_local_repair',reason='insufficient_tracking_margin',margin_m=margin))
            return False
        self.path=list(route)[1:]
        if not self.path:self.path=[xy.copy()]
        self.goal=self.path[-1].copy()
        # Keep the observed destination when the cached graph edge later needs
        # a local A* repair. Cached-only followers have no explicit destinations.
        self._recovery_destination=self.goal.copy()
        self.last_progress=xy.copy();self.progress_time=now
        self.following_cached_route=True
        self.events.append(dict(event='reuse_successful_route',strategy='shortest_successful_route_graph',
                                destination=self.route_destination,**result))
        return True

    def _plan(self,xy,now):
        self.layers(xy);self.last_plan=now;self.plans+=1
        if self._cached_route(xy,now):return None
        nodes=np.argwhere(self.free)
        if not len(nodes):return 'room_route_unobserved'
        from scipy.ndimage import label
        search_free=self._search_free(xy)
        components, _ = label(search_free)
        start = self.cell(xy)
        if not (0 <= start[0] < self.free.shape[0] and 0 <= start[1] < self.free.shape[1]) or not search_free[start]:
            return 'room_route_unobserved'
        nodes = nodes[components[nodes[:,0],nodes[:,1]] == components[start]]
        points=self.world(nodes)
        destinations=list(self.destinations)
        if hasattr(self,'_recovery_destination'):destinations.insert(0,self._recovery_destination)
        for destination in destinations:
            distances=np.linalg.norm(points-destination,axis=1)
            choices=np.argsort(distances)
            for index in choices[:32]:
                if distances[index]>1.:break
                goal=tuple(nodes[index]);path=astar_observed(search_free,self.cell(xy),goal)
                if not path:continue
                # Keep the current cell's center when needed to connect the
                # continuous robot pose to the grid route around a corner.
                route=np.vstack((xy,self.world(path)))
                if any(not self._line_free(a,b) for a,b in zip(route,route[1:])):continue
                self.path=list(route[1:]) or [points[index].copy()]
                self.goal=points[index].copy();self.last_progress=xy.copy();self.progress_time=now
                self.events.append(dict(event='observed_room_standoff',label_position=destination.tolist(),waypoint=self.goal.tolist(),offset_m=float(distances[index])))
                return None
        return 'room_route_unobserved'
