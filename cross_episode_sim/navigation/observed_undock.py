"""Leave a close manipulation dock using only observed shape clearance."""
import numpy as np
from .dock_follower import DockFollower
from .placement_docks import plan_to_any_dock

# Extra clearance (metres) tried in order when planning a departure route.
PLANNING_CLEARANCES=(.04,.02,0.)
# Used only when the pose the robot is standing in is itself inside the nominal
# safety margin (voxel rounding of a counter front 44 cm beside the base did
# this in fix95/fix97). The robot physically occupies that pose, so the gap
# there is smaller than the margin, not zero. The route is then planned and
# followed with a smaller margin; the shape check itself always stays on.
# Without this the departure fell back to retracing the recorded approach,
# which is not collision-checked and ran the base into the just-opened door.
STANDING_MARGIN_REDUCTIONS=(-.015,-.03)


# Leaving a standing pose that is inside an observed obstacle's safety margin:
# straight back (or forward) in these steps, at most this far (m).
ESCAPE_STEP=.02
ESCAPE_LIMIT=.35


def straight_escape(checker,pose,step=ESCAPE_STEP,limit=ESCAPE_LIMIT):
    """Pose straight behind (or ahead of) a blocked standing pose that is clear, or None.

    After a push-close the base can stand with a cabinet front inside its
    padded footprint (fix124: 75 map points 4 cm from the front corner, every
    departure plan rejected, 'undock_route_unverified', whole fridge half
    lost), and after opening the microwave with the door handle inside the
    padded gripper volume. The back-off is accepted only while the number of
    observed points inside the padded robot volumes never grows, and only if
    it ends clear. Returns None when the pose is clear or no such back-off exists.
    """
    pose=np.asarray(pose,float)
    if checker.clear(pose):return None
    heading=np.array([np.cos(pose[2]),np.sin(pose[2])])
    for direction in (-1.,1.):
        count=checker.contact_count(pose)
        for k in range(1,int(round(limit/step))+1):
            p=pose.copy();p[:2]+=direction*k*step*heading
            now=checker.contact_count(p)
            if now>count:break
            count=now
            if checker.clear(p):return p
    return None


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

    def _plan_departure(self,dock,pose,packet,free,extra):
        """Plan the route out of the dock with `extra` clearance beyond the per-step check."""
        own=dock.__dict__.get('make_checker');make=dock.make_checker
        touched=[]
        def padded(p,q):
            checker=make(p,q)
            # Checkers without a clearance margin plan as before. A dock may
            # hand out one shared checker every time (the transit stage does):
            # set its margin from the remembered nominal value, never add to
            # what an earlier attempt left there. Adding made the three
            # attempts in fix95/fix97 run at +4, +6 and +6 cm instead of
            # +4, +2 and 0, so the nominal margin was never tried.
            if hasattr(checker,'margin'):
                if not hasattr(checker,'nominal_margin'):checker.nominal_margin=checker.margin
                checker.margin=checker.nominal_margin+extra;touched.append(checker)
            return checker
        dock.make_checker=padded
        try:
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
                    dock.events.append(dict(event='undock_route_rejected',planning_clearance_m=extra,observed_goal_cells=int(np.count_nonzero(free)),interior_goal_cells=int(np.count_nonzero(goal_free)),nearby_candidates=int(len(ids)),shape_clear_candidates=len(candidates),detail=dock.checker.detail))
                    return False
                return True
        finally:
            # Planning clearance is for planning only: the per-step check
            # while driving uses the nominal margin again.
            for checker in touched:checker.margin=checker.nominal_margin
            if own is None:dock.__dict__.pop('make_checker',None)
            else:dock.make_checker=own

    def step(self,packet,tracking,free,now):
        stop=lambda status:dict(status=status,forward=0.,yaw=0.)
        if not tracking.accepted:return stop('tracking_lost')
        T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        if self.started is None:self.started=now
        if now-self.started>self.budget:self.error='undock_timeout';return stop(self.error)
        dock=self.docking
        if self.follower is None:
            # Plan with more clearance than the per-step check requires, so
            # small tracking errors do not trip it beside an opened door
            # (fix82: the rear corner entered the margin of the open cabinet
            # leaf while turning and the departure stopped). Fall back to
            # less clearance, then none, before giving up.
            self.margin_reduction=0.
            for extra in PLANNING_CLEARANCES:
                if self._plan_departure(dock,pose,packet,free,extra):
                    dock.events.append(dict(event='undock_route_planned',planning_clearance_m=extra));break
            else:
                nominal=dock.make_checker(pose,packet.arm.position)
                if hasattr(nominal,'margin'):nominal.margin=getattr(nominal,'nominal_margin',nominal.margin)
                standing_blocked=hasattr(nominal,'margin') and not nominal.clear(pose)
                counter=getattr(nominal,'contact_count',None)
                dock.events.append(dict(event='undock_standing_pose_check',inside_nominal_margin=bool(standing_blocked),
                    contact_points=int(counter(pose)) if standing_blocked and counter is not None else 0))
                for extra in STANDING_MARGIN_REDUCTIONS if standing_blocked else ():
                    if getattr(nominal,'nominal_margin',nominal.margin)+extra<0.:continue
                    if self._plan_departure(dock,pose,packet,free,extra):
                        self.margin_reduction=extra
                        dock.events.append(dict(event='undock_route_planned',planning_clearance_m=extra,
                            reason='standing pose is inside the nominal margin; planned and followed with a reduced margin'));break
                else:
                    # Last resort from a pose inside the margin: straight out of
                    # it, never deeper into any observed point, then plan the
                    # departure again from the clear pose.
                    escape=straight_escape(nominal,pose) if standing_blocked and counter is not None else None
                    if escape is None:
                        self.error='undock_route_unverified';return stop(self.error)
                    dock.checker=nominal;dock.path=[escape];dock.goal=escape;self.escaping=True
                    self.goal_region=np.zeros(free.shape,bool);self.departure_start=pose[:2].copy()
                    dock.events.append(dict(event='undock_straight_escape',contact_points=int(counter(pose)),
                        along_heading_m=float((escape[:2]-pose[:2])@np.array([np.cos(pose[2]),np.sin(pose[2])])),
                        reason='standing pose is inside the margin and no departure route was found'))
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
            # Follow a reduced-margin plan with the margin it was checked with.
            reduction=getattr(self,'margin_reduction',0.)
            if hasattr(dock.checker,'margin'):
                if not hasattr(dock.checker,'nominal_margin'):dock.checker.nominal_margin=dock.checker.margin
                dock.checker.margin=dock.checker.nominal_margin+reduction
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
        if self.follower.complete and getattr(self,'escaping',False):
            self.escaping=False;self.follower=None;self.started=now;self.velocity[:]=0.
            dock.events.append(dict(event='undock_escape_complete',position=pose[:2].tolist()))
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
            # Escape: a dock beside a just-opened door can start with map
            # points inside the base's safety margin (fix73: the open leaf's
            # slab at the rear corner), so every step was stopped, including
            # steps away from it. While still in contact and near the dock, a
            # step is allowed if it never increases the contact; once clear
            # the full check applies again.
            counter=getattr(dock.checker,'contact_count',None)
            start_contact=counter(pose) if counter is not None and np.linalg.norm(pose[:2]-self.departure_start)<.4 and not dock.checker.clear(pose) else 0
            for t in np.arange(0,.25+max(abs(command[0])/.4,abs(command[1])/.8)+.02,.02):
                v=np.sign(command[0])*max(0,abs(command[0])-.4*max(0,t-.25))
                w=np.sign(command[1])*max(0,abs(command[1])-.8*max(0,t-.25))
                p[:2]+=v*.02*np.array([np.cos(p[2]),np.sin(p[2])]);p[2]+=w*.02
                if not dock.checker.clear(p):
                    if start_contact and counter(p)<=start_contact:
                        if not getattr(self,'escape_logged',False):
                            self.escape_logged=True
                            dock.events.append(dict(event='undock_escape_from_contact',contact_points=start_contact,detail=dock.checker.detail))
                        continue
                    dock.events.append(dict(event='undock_clearance_stop',detail=dock.checker.detail,start_contact_points=start_contact))
                    if self.replans<2:
                        # Stopped short of the obstacle, still on checked
                        # ground: plan a new way out from here instead of
                        # abandoning the departure.
                        self.replans+=1;self.follower=None;self.started=now;self.velocity[:]=0.
                        dock.events.append(dict(event='undock_replanned',position=pose[:2].tolist(),reason='clearance stop'))
                        return stop('observed_undock_to_transit')
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
