"""Leave a close manipulation dock using only observed shape clearance."""
import numpy as np
from .dock_follower import DockFollower
from .placement_docks import plan_to_any_dock

class ObservedUndock:
    def __init__(self,docking,*,max_distance=1.5,goal_heading=None,planner=None,erode_goals=True,runtime_collision_stop=True,motion_config=None,goal_pose_required=False):
        self.goal_pose_required=goal_pose_required
        self.motion_config=motion_config
        self.runtime_collision_stop=runtime_collision_stop
        self.erode_goals=erode_goals
        self.max_distance=max_distance;self.goal_heading=goal_heading
        self.planner=planner or plan_to_any_dock
        self.docking=docking;self.follower=None;self.last_frame=None
        self.velocity=np.zeros(2);self.started=None;self.error=None;self.budget=30.;self.replans=0

    def step(self,packet,tracking,free,now):
        stop=lambda status:dict(status=status,forward=0.,yaw=0.)
        if not tracking.accepted:return stop('tracking_lost')
        T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        if self.started is None:self.started=now
        if now-self.started>self.budget:self.error='undock_timeout';return stop(self.error)
        dock=self.docking
        if self.follower is None:
            # A pose within the follower's stopping tolerance must still be in
            # transit-free space, including grid discretization uncertainty.
            from scipy.ndimage import distance_transform_edt
            resolution=dock.grid.config.resolution
            interior=distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*resolution
            goal_free=free & (interior > .015+np.sqrt(2)*resolution) if self.erode_goals else free.copy()
            self.goal_region=goal_free
            self.departure_start=pose[:2].copy()
            cells=np.argwhere(goal_free);xy=dock.grid.cell_to_world(cells[:,::-1])
            delta=xy-pose[:2];distance=np.linalg.norm(delta,axis=1)
            ids=np.flatnonzero((distance>.15)&(distance<self.max_distance))
            ids=ids[np.argsort(distance[ids])]
            candidates=[]
            dock.checker=dock.make_checker(pose,packet.arm.position)
            for i in ids:
                # Preserve the current orientation where possible; the A* path
                # may use checked forward/reverse segments to reach open floor.
                p=np.r_[xy[i],pose[2] if self.goal_heading is None else self.goal_heading]
                if any(np.linalg.norm(p[:2]-q[:2])<.1 for q in candidates):continue
                if not dock.checker.clear(p):continue
                candidates.append(p)
                if len(candidates)>=30:break
            direct=None
            # Grid centers can miss the exact current heading by half a cell.
            # Test continuous straight departure poses before adding turns.
            forward=np.array([np.cos(pose[2]),np.sin(pose[2])])
            aligned=[]
            for distance in np.arange(.15,self.max_distance+.01,.05):
                for sign in (-1.,1.):
                    candidate=np.r_[pose[:2]+sign*distance*forward,pose[2]]
                    col,row=dock.grid.world_to_cell(candidate[None,:2])[0]
                    if 0<=row<free.shape[0] and 0<=col<free.shape[1] and goal_free[row,col]:
                        aligned.append(candidate)
            for candidate in aligned+candidates:
                delta=candidate[:2]-pose[:2]
                lateral=abs(delta@np.array([-np.sin(pose[2]),np.cos(pose[2])]))
                heading_ok=not self.goal_pose_required or self.goal_heading is None or abs((candidate[2]-self.goal_heading+np.pi)%(2*np.pi)-np.pi)<.015
                if heading_ok and lateral<.025 and dock.checker.swept(pose,candidate):
                    direct=candidate;break
            if direct is not None:
                dock.path=[direct];dock.goal=direct
            elif not candidates or not self.planner(dock,pose,packet.arm.position,candidates):
                dock.events.append(dict(event='undock_route_rejected',observed_goal_cells=int(np.count_nonzero(free)),interior_goal_cells=int(np.count_nonzero(goal_free)),nearby_candidates=int(len(ids)),shape_clear_candidates=len(candidates),detail=dock.checker.detail))
                self.error='undock_route_unverified';return stop(self.error)
            original=list(dock.path)
            dock.path=departure_route(pose,original,dock.checker,keep_final_rotation=self.goal_pose_required)
            dock.goal=dock.path[-1]
            route=[pose]+dock.path
            distance=sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip(route,route[1:]))
            turns=sum(abs((b[2]-a[2]+np.pi)%(2*np.pi)-np.pi) for a,b in zip(route,route[1:]))
            # Include acceleration and exponential endpoint slowdown; length/speed
            # alone underestimates the actual DockFollower arrival time.
            speed=.20 if self.motion_config is None else self.motion_config.speed
            yaw_speed=.3 if self.motion_config is None else self.motion_config.yaw_speed
            gain=.8 if self.motion_config is None else self.motion_config.approach_gain
            tolerance=.015 if self.runtime_collision_stop else .04
            settling=max(0.,np.log((speed/gain)/tolerance)/gain)
            self.budget=max(15.,distance/speed+turns/yaw_speed+(settling+speed/.4+2.)*len(dock.path))
            dock.events.append(dict(event='departure_motion_budget',seconds=self.budget,distance_m=distance,turn_radians=turns,removed_waypoints=len(original)-len(dock.path),reason='Observed route length, turns, and measured-settling allowance'))
            self.follower=DockFollower(dock.path,max_speed=speed,max_yaw_speed=yaw_speed,approach_gain=gain,recover_turn_drift=not self.runtime_collision_stop,
                                      position_tolerance=tolerance,turn_drift_tolerance=.04 if self.runtime_collision_stop else .15)
            dock.events.append(dict(event='observed_undock_to_transit',goal=dock.goal.tolist(),reason='Close manipulation dock is outside conservative room-transit clearance'))
        if self.runtime_collision_stop and packet.timestamp!=self.last_frame:
            dock.checker=dock.make_checker(pose,packet.arm.position);self.last_frame=packet.timestamp
        col,row=dock.grid.world_to_cell(pose[None,:2])[0]
        region_reached=(not self.goal_pose_required and 0<=row<free.shape[0] and 0<=col<free.shape[1]
                        and self.goal_region[row,col]
                        and np.linalg.norm(pose[:2]-self.departure_start)>.15)
        # Departure has a free-region goal, not a manipulation pose. Brake
        # inside its validated interior rather than chasing a jittery endpoint.
        if region_reached:
            desired=np.zeros(2)
            if np.max(np.abs(self.velocity))<1e-8:
                return stop('undock_complete')
        else:
            try:desired=np.array(self.follower.command(pose))
            except RuntimeError as exc:
                if str(exc)!='Dock turn drift requires replanning' or self.replans>=2:
                    self.error=str(exc);return stop('undock_tracking_error')
                # A localization correction moved the estimate off the checked
                # segment. Stop and check a new departure from where it is now.
                self.replans+=1;self.follower=None;self.started=now;self.velocity[:]=0.
                dock.events.append(dict(event='undock_replanned',position=pose[:2].tolist(),reason=str(exc)))
                return stop('observed_undock_to_transit')
        if self.follower.complete:
            if not self.erode_goals:
                # Portal goals are geometric exit poses, not a room-transit
                # free-region certificate. The shape-checked follower may stop
                # within its tolerance in the neighboring raster cell.
                self.velocity[:]=0.
                return stop('undock_complete')
            col,row=dock.grid.world_to_cell(pose[None,:2])[0]
            if not (0<=row<free.shape[0] and 0<=col<free.shape[1] and free[row,col]):
                self.error='undock_endpoint_outside_transit';return stop(self.error)
            return stop('undock_complete')
        limit=np.array([.4,.8])*.02
        command=self.velocity+np.clip(desired-self.velocity,-limit,limit)
        if self.runtime_collision_stop:
            p=pose.copy()
            for t in np.arange(0,.25+max(abs(command[0])/.4,abs(command[1])/.8)+.02,.02):
                v=np.sign(command[0])*max(0,abs(command[0])-.4*max(0,t-.25))
                w=np.sign(command[1])*max(0,abs(command[1])-.8*max(0,t-.25))
                p[:2]+=v*.02*np.array([np.cos(p[2]),np.sin(p[2])]);p[2]+=w*.02
                if not dock.checker.clear(p):
                    dock.events.append(dict(event='undock_clearance_stop',detail=dock.checker.detail))
                    self.error='undock_clearance_stop';return stop(self.error)
        self.velocity=command
        return dict(status='observed_undock_to_transit',forward=float(command[0]),yaw=float(command[1]))


def departure_route(start, route, checker, *, keep_final_rotation=False):
    """No destination-facing requirement for leaving a dock into open floor."""
    path=[np.asarray(p).copy() for p in route]
    if not keep_final_rotation and len(path)>1 and np.linalg.norm(path[-1][:2]-path[-2][:2])<1e-6:
        # The preceding arrival pose was already swept-checked. Restoring the
        # departure heading is redundant; room navigation chooses its heading.
        path.pop()
    # Consecutive lattice rotations at one position need not settle at every
    # 45-degree heading. Only collapse if the direct rotation sweeps clear.
    compact=[];i=0
    while i<len(path):
        j=i
        while j+1<len(path) and np.linalg.norm(path[j+1][:2]-path[i][:2])<1e-6:j+=1
        previous=compact[-1] if compact else np.asarray(start)
        if j>i and np.linalg.norm(previous[:2]-path[j][:2])<1e-6 and checker.swept(previous,path[j]):
            compact.append(path[j])
        elif j>i and checker.swept(path[i],path[j]):
            compact.extend([path[i],path[j]])
        else:compact.extend(path[i:j+1])
        i=j+1
    result=[np.asarray(start)]
    for point in compact:
        if len(result)>1:
            a,b=result[-2:]
            wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
            ab=b[:2]-a[:2];bc=point[:2]-b[:2]
            collinear=abs(ab[0]*bc[1]-ab[1]*bc[0])<1e-7 and np.dot(ab,bc)>=0
            if collinear and abs(wrap(a[2]-b[2]))<1e-6 and abs(wrap(b[2]-point[2]))<1e-6 and checker.swept(a,point):
                result.pop()
        result.append(point)
    return result[1:]
