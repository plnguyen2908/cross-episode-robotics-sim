"""Short observed doorway crossing using the calibrated robot shape once."""
from types import SimpleNamespace
import numpy as np
from .observed_undock import ObservedUndock
from .house_exploration import HouseExplorationConfig
from .shaped_docking import ShapedDocking
from .placement_docks import plan_to_any_dock


def fresh_portal_map(source,support_poses=()):
    """Refine local observations without inventing finer floor measurements."""
    from dataclasses import replace
    from .observed_map import ObservedNavigationMap
    grid=ObservedNavigationMap(replace(source.config,resolution=.05,height_resolution=.05,extent_m=16))
    samples=list(getattr(source,'floor_samples',{}).values())
    if samples:
        points=np.array([p for p,n in samples]);counts=np.array([n for p,n in samples])
        grid.ensure_bounds(points[:,:2]);grid.record_floor_samples(points,counts)
        cells=grid.world_to_cell(points[:,:2])
        np.maximum.at(grid.floor_views,(cells[:,1],cells[:,0]),counts.astype(np.uint16))
    grid.inherited_support_poses=[];seen=set()
    for T in support_poses:
        pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        key=tuple(np.floor(pose/np.array([.025,.025,.05])).astype(int))
        if key not in seen:grid.inherited_support_poses.append(pose);seen.add(key)
    return grid


def plan_portal_route(dock,pose,q,candidates,center,normal):
    """Align in open space, then cross the observed doorway in one straight leg.

    A start-relative 45-degree lattice can miss a narrow valid doorway even
    when its exact observed normal is known. All connectors and the crossing
    retain the same full robot-shape swept check.
    """
    dock.checker=dock.make_checker(pose,q)
    heading=np.arctan2(normal[1],normal[0]);stages=[];exits=[];rejections=[]
    leaf_reach=getattr(dock.grid,'opened_leaf_reach',None)
    if leaf_reach is not None:
        from .success_graph_transit import drive_connection
        # First retreat beyond the observed open leaf, then move parallel to
        # the doorway, then cross straight through its actual open segment.
        # This prevents a diagonal shortcut from nudging the leaf closed.
        for extra in (0.,.2,.4):
            setback=leaf_reach+HouseExplorationConfig().footprint_radius+.15+extra
            along=(pose[:2]-center)@normal
            outward=np.r_[pose[:2]-(along+setback)*normal,pose[2]]
            for goal in candidates:
                goal=np.r_[goal[:2],heading]
                stage=np.r_[goal[:2]-((goal[:2]-center)@normal+setback)*normal,heading]
                path=[];previous=pose
                for waypoint in (outward,stage,goal):
                    connection=drive_connection(previous,waypoint,dock.checker)
                    if connection is None:break
                    path.extend(connection);previous=waypoint
                else:
                    dock.path=path;dock.goal=goal
                    dock.events.append(dict(event='observed_leaf_outward_first_crossing',leaf_reach_m=leaf_reach,setback_m=setback,route=[p.tolist() for p in path]))
                    return True
    for goal in candidates:
        goal=np.r_[goal[:2],heading]
        if not dock.checker.clear(goal):continue
        along=(goal[:2]-center)@normal
        for setback in (.8,1.,1.2,1.4):
            stage=np.r_[goal[:2]-(along+setback)*normal,heading]
            if dock.checker.swept(stage,goal):
                stages.append(stage);exits.append(goal)
                break
        else:rejections.append(dict(goal=goal.tolist(),detail=dock.checker.detail))
    if stages and plan_to_any_dock(dock,pose,q,stages):
        index=int(np.argmin([np.linalg.norm(dock.goal-stage) for stage in stages]))
        dock.path.append(exits[index]);dock.goal=exits[index]
        dock.events.append(dict(event='observed_doorway_aligned_crossing',stage=stages[index].tolist(),goal=dock.goal.tolist()))
        return True
    dock.events.append(dict(event='observed_doorway_alignment_unverified',staging_candidates=len(stages),rejections=rejections))
    return False


class PortalCrossing:
    def __init__(self,grid,geometry,center,normal,start,*,transit_grid=None):
        self.grid=grid;self.center=np.asarray(center)[:2]
        self.transit_grid=transit_grid
        self.normal=np.asarray(normal,dtype=float)[:2].copy();self.normal/=np.linalg.norm(self.normal)
        if (np.asarray(start)[:2]-self.center)@self.normal>0:self.normal*=-1
        self.dock=ShapedDocking(grid,geometry)
        self.dock.require_observed_support=False
        self.motion=ObservedUndock(self.dock,erode_goals=False,runtime_collision_stop=False,motion_config=HouseExplorationConfig(),max_distance=4. if transit_grid is not None else 2.5,goal_heading=np.arctan2(self.normal[1],self.normal[0]),
            planner=lambda dock,pose,q,goals:plan_portal_route(dock,pose,q,goals,self.center,self.normal))
        self.follower=SimpleNamespace(terminal=None,path=[],stop_detail=None,events=self.dock.events)
        self._goals=None

    def step(self,packet,tracking,now):
        if self._goals is None:self._goals=self.goal_mask()
        result=self.motion.step(packet,tracking,self._goals,now)
        self.follower.path=self.dock.path or []
        self.follower.stop_detail=getattr(getattr(self.dock,'checker',None),'detail',None) if self.motion.error else None
        if self.motion.error:self.follower.terminal=result['status']
        if result['status']=='undock_complete':return dict(status='portal_crossed',forward=0.,yaw=0.)
        if result['status']=='observed_undock_to_transit':result['status']='cross_observed_portal'
        return result

    def goal_mask(self):
        yy,xx=np.indices(self.grid.floor_views.shape)
        xy=self.grid.cell_to_world(np.stack((xx,yy),axis=-1))
        delta=xy-self.center
        along=delta@self.normal
        lateral=abs(delta@np.array([-self.normal[1],self.normal[0]]))
        # Exploration may enter unobserved floor. Propose a short centered exit
        # geometrically; do not demand a pre-mapped destination or transit disk.
        # Existing occupied geometry and fresh obstacle updates remain active.
        goals=~(self.grid.evidence>=2).any(axis=2)
        goals &= (along>.55)&(along<.85)&(lateral<=max(.04,self.grid.config.resolution*.75))
        self.dock.events.append(dict(event='portal_geometric_exit_candidates',
                                     count=int(goals.sum()),requires_observed_floor=False))
        return goals


class OpenedPortalHandoff:
    """Continue from a door operation without restarting an angular scan.

    Keep the existing observed map. Two fresh accepted sensor packets permit
    its changed-panel evidence to update, then use the same shape-checked A*
    crossing. Unknown floor remains unknown in the map but is not a veto on
    exploration movement. Detected obstacles remain collision constraints.
    """
    def __init__(self,grid,geometry,center,normal,*,after_timestamp,refresh_view=False,resolve_opening=True):
        self.grid=grid;self.geometry=geometry;self.center=center;self.normal=normal
        self.resolve_opening=resolve_opening;self.aperture=None
        self.after_timestamp=after_timestamp;self.observations=set();self.crossing=None
        self.retry_after=None;self.retries=0;self.history=[]
        from .doorway_view_refresh import DoorwayViewRefresh
        self.view=DoorwayViewRefresh(grid,geometry,center) if refresh_view else None
        self.follower=SimpleNamespace(terminal=None,path=[],stop_detail=None,events=[])

    def step(self,packet,tracking,now):
        if self.view is not None and not self.view.complete:
            result=self.view.step(packet,tracking,now)
            if self.view.error:
                self.follower.terminal=self.view.error
                self.follower.stop_detail=getattr(getattr(self.view.dock,'checker',None),'detail',None)
            if not self.view.complete:return result
            self.after_timestamp=packet.timestamp;self.observations.clear()
            self.history.append(dict(event='directed_doorway_refresh',room_scan=False))
        if (tracking.accepted and packet.timestamp>self.after_timestamp
                and abs(packet.timestamp-tracking.timestamp)<1e-6):
            self.observations.add(packet.timestamp)
        if self.retry_after is not None:
            if packet.timestamp<=self.retry_after or not tracking.accepted:
                return dict(status='refresh_open_portal',forward=0.,yaw=0.)
            self.retry_after=None
        if self.crossing is None:
            if len(self.observations)<2:
                if now-self.after_timestamp>2.:
                    self.follower.terminal='post_open_observation_unavailable'
                return dict(status=self.follower.terminal or 'refresh_open_portal',forward=0.,yaw=0.)
            if self.resolve_opening:
                from .open_aperture import observed_open_aperture
                self.aperture=observed_open_aperture(self.grid,self.center,self.normal)
                if self.aperture is None:
                    self.follower.terminal='open_aperture_unobserved'
                    self.follower.events.append(dict(event='open_aperture_unobserved',seed=list(self.center)))
                    return dict(status='open_aperture_unobserved',forward=0.,yaw=0.)
                self.center=self.aperture['center']
                self.history.append(dict(event='observed_open_aperture',**self.aperture))
            self.crossing=PortalCrossing(self.grid,self.geometry,self.center,self.normal,
                                         tracking.T_map_base[:2,3],transit_grid=self.grid)
            self.follower=self.crossing.follower
            self.follower.events.extend(self.history)
            self.follower.events.append(dict(event='post_open_map_handoff',forced_scan=False,
                                              fresh_observations=len(self.observations)))
        result=self.crossing.step(packet,tracking,now)
        if self.follower.terminal=='undock_route_unverified' and self.retries<2:
            self.retries+=1
            self.history=list(self.follower.events)+[dict(event='doorway_evidence_retry',attempt=self.retries,forced_scan=False)]
            self.crossing=None;self.retry_after=packet.timestamp+.5
            self.follower=SimpleNamespace(terminal=None,path=[],stop_detail=None,events=self.history)
            return dict(status='refresh_open_portal',forward=0.,yaw=0.)
        return result
