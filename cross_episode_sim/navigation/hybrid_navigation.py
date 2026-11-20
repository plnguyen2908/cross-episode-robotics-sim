"""Room-scale SLAM routing, local RING, then measured readiness / A* redocking."""
import numpy as np
from .house_exploration import HouseExplorer
from .semantic_navigation import astar_observed
from .edge_first import edge_first

EDGE_LOOK_RANGE=1.6    # the docking planner's own close-range band
EDGE_LOOK_ROUNDS=3     # pointing rounds: two form a track, the third settles its category
EDGE_LOOK_SECONDS=4.


def scan_stop_count_resets(status):
    """Whether a local-scan step status clears the count of collision-stopped scan actions.

    Only a completed action (or a status from outside the scan) does. A step
    that merely starts an action must not: beside the counter the first 50 Hz
    tick of each turn was clear and the second was stopped, so the count never
    reached its limit and the robot twitched in place for over 60 simulated
    seconds (fix144, last drive to the fridge: 686 scan predictions, no motion).
    """
    if status in {'await_fresh_ring_observation','ring_action_timeout'}:return False
    return status=='ring_action_complete' or not str(status).startswith('ring_')


class RoomWaypointFollower(HouseExplorer):
    def _depth(self,camera):
        if hasattr(self.grid,'fixed_obstacle_points'):
            self.near_points=np.empty((0,3))
        else:super()._depth(camera)

    def set_destinations(self,points):
        self.destinations=[np.asarray(p,float) for p in points]
        self.phase='travel';self.path=[];self.goal=None;self.terminal=None
    def _plan(self,xy,now):
        self.layers(xy);self.last_plan=now;self.plans+=1
        for point in self.destinations:
            goal=self.cell(point)
            if not(0<=goal[0]<self.free.shape[0] and 0<=goal[1]<self.free.shape[1]):continue
            path=astar_observed(self.free,self.cell(xy),goal)
            if not path:continue
            self.path=list(self.world(path))[1:] or [point.copy()];self.goal=point.copy()
            self.last_progress=xy.copy();self.progress_time=now
            return None
        return 'room_route_unobserved'

# A confirmed handle this close is left to the fixture skill's own redock.
FIXTURE_HANDOFF_DISTANCE=2.
# Handle heights (map z, metres) the arm reaches from a frontal dock.
HANDLE_HEIGHT=(.25,1.35)


class HybridNavigator:
    """Only sensor packets, SLAM estimates and inferred semantic maps enter.

    The oracle/VLM task planner supplies desired room/object CATEGORY strings.
    No simulator room IDs, object positions or reference trajectories are used.
    """
    def __init__(self,semantic,local,rooms,readiness,goal_room='kitchen',docking=None):
        self.docking=docking;self.docking_success=False
        if docking is not None and hasattr(docking,"make_checker"):local.shape_checker_factory=docking.make_checker
        self.semantic=semantic;self.local=local;self.rooms=rooms;self.readiness=readiness
        self.goal_room=goal_room;self.mode='room_search';self.terminal=None;self.started=None
        self.transit=RoomWaypointFollower(semantic.grid,semantic.explorer.config)
        self.last_readiness=None;self.events=[];self.last_room=None;self.redock_attempts=0;self.max_redocks=3
        self.blocked_predictions=0
        self.observation_recoveries=0;self.handoff_blocked_at=None
        self.undock=None;self.visited_room_waypoints=[]
        self.search_window=[];self.search_recovery=None;self.search_recovery_started=None
        self.searched_viewpoints=[];self.room_approach=False
        self.tentative_attempted=set();self.tentative_until=None;self.proposal_inspection_started=None
        self.confirmation_since=None;self.done_at=None;self.found_target=None;self.object_search_success=False
        self.edge_look=None;self.edge_looked_for=None
    def _switch(self,mode,now):
        if mode==self.mode:return
        self.events.append(dict(time=now,event='mode',previous=self.mode,next=mode))
        self.local.reset_motion()
        if mode=='local':
            self.local.policy.reset();self.search_window=[]
        if mode=='room_search':
            self.semantic.explorer.path=[];self.semantic.explorer.terminal=None
            self.semantic.explorer.last_time=now;self.semantic.explorer.velocity[:]=0
        self.mode=mode
    def _request_redock(self,now):
        self.redock_attempts+=1;self.blocked_predictions=0
        if self.redock_attempts>self.max_redocks:
            self.terminal='redocking_exhausted'
            return self.semantic.follower._stop(self.terminal)
        self.semantic.terminal=None;self.semantic.mode='explore';self.semantic.started=self.started;self.semantic.allow_exploration=False
        self.semantic.required_track_id=getattr(self.found_target,'track_id',None)
        self.semantic.follower.terminal=None;self.semantic.follower.path=[]
        self.semantic.follower.last_time=now
        self.semantic.explorer.terminal=None;self.semantic.explorer.path=[]
        self.semantic.explorer.last_time=now;self.semantic.explorer.velocity[:]=0
        if self.docking is not None:
            chosen=self.found_target or self.semantic.follower.target
            if chosen is None:
                self.terminal='redock_target_unverified';return self.semantic.follower._stop(self.terminal)
            self.docking.reset(chosen)
        self._switch('redock',now)
        return self.semantic.follower._stop('redock_requested')

    def _room_destinations(self,xy):
        # A viewing waypoint is not proof of room entry. Once reached without
        # label confirmation, search elsewhere instead of replanning to itself.
        return [p for p in self.rooms.destinations(self.goal_room,xy)
                if np.linalg.norm(np.asarray(p)-xy)>.3 and
                all(np.linalg.norm(np.asarray(p)-v)>.5 for v in self.visited_room_waypoints)]

    def _start_room_approach(self,xy,now):
        # A room label at a doorway is not a useful object-search viewpoint.
        # Only sensor-derived room places and observed-free A* routes enter.
        room_points=self._room_destinations(xy)
        points=[p for p in room_points if np.linalg.norm(p-xy)>=.6]
        # Prefer a deeper previously observed room viewpoint over the nearest
        # doorway sighting. Route feasibility still decides which is reachable.
        points.sort(key=lambda p:np.linalg.norm(p-xy),reverse=True)
        follower=RoomWaypointFollower(self.semantic.grid,self.semantic.explorer.config)
        follower.set_destinations(points)
        strategy='room_viewpoint'
        if not points or follower._plan(xy,now) is not None:
            follower=HouseExplorer(self.semantic.grid,self.semantic.explorer.config)
            follower.phase='travel';follower.failed_goals=list(self.visited_room_waypoints)
            strategy='frontier'
        follower.last_time=now
        self.search_recovery=follower;self.search_recovery_started=now;self.room_approach=True
        self.events.append(dict(time=now,event='approach_room_before_local_search',strategy=strategy))
        self._switch('search_recovery',now)

    def _search_stalled(self,xy,now):
        self.search_window.append((now,np.asarray(xy).copy()))
        while len(self.search_window)>1 and self.search_window[1][0]<=now-15.:
            self.search_window.pop(0)
        return (now-self.search_window[0][0]>=15. and
                np.max(np.linalg.norm(np.array([p for _,p in self.search_window])-xy,axis=1))<.6)

    def _start_search_recovery(self,xy,now):
        # Old observations are search hints only, never fresh grasp targets.
        hints=getattr(self.semantic,'remembered_object_hypotheses',[])
        points=[];look_hints=[]
        for hint in hints:
            if hint.label!=self.semantic.target_label:continue
            look_hints.append(hint.position)
            for radius in (1.,1.4):
                for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
                    point=hint.position[:2]+radius*np.array([np.cos(angle),np.sin(angle)])
                    if np.linalg.norm(point-xy)<.6:continue
                    if any(np.linalg.norm(point-v)<.6 for v in self.searched_viewpoints):continue
                    points.append(point)
        # A missing object detection does not mean the floor map is missing.
        # Search from already mapped viewpoints around the requested station.
        # These landmarks were observed during exploration, not scene truth.
        station=getattr(self.semantic,'search_station',None)
        for landmark in getattr(self.semantic,'remembered_furniture',[]):
            if not station or landmark['label']!=station:continue
            center=np.asarray(landmark['position'],float)
            look_hints.append(center)
            station_points=[np.asarray(p,float)[:2] for p in landmark.get('viewpoints',[])]
            for radius in (.8,1.2,1.6,2.):
                station_points.extend(center[:2]+radius*np.array([np.cos(a),np.sin(a)])
                    for a in np.linspace(-np.pi,np.pi,16,endpoint=False))
            for point in station_points:
                if np.linalg.norm(point-xy)<.6:continue
                if any(np.linalg.norm(point-v)<.6 for v in self.searched_viewpoints):continue
                points.append(point)
        self.recovery_look_hint=(min(look_hints,key=lambda p:np.linalg.norm(p[:2]-xy)).copy()
                                 if look_hints else None)
        points.sort(key=lambda p:np.linalg.norm(p-xy))
        follower=RoomWaypointFollower(self.semantic.grid,self.semantic.explorer.config)
        follower.set_destinations(points)
        if not points or follower._plan(xy,now) is not None:
            if hasattr(self.semantic.grid,'fixed_obstacle_points'):
                # A valid close dock can lie outside the circular transit
                # footprint. Use the existing shape-checked departure first;
                # do not misclassify that as an unexplored room.
                follower.layers(xy)
                row,col=follower.cell(xy)
                inside=0<=row<follower.free.shape[0] and 0<=col<follower.free.shape[1]
                if points and inside and not follower.free[row,col] and self.docking is not None:
                    from .observed_undock import ObservedUndock
                    self.search_recovery=follower
                    self.undock=ObservedUndock(self.docking,runtime_collision_stop=False)
                    self.events.append(dict(time=now,event='search_departure_required'))
                    self._switch('search_departure',now)
                    return
                self.search_recovery=None
                self.events.append(dict(time=now,event='search_viewpoints_unreachable',candidates=len(points)))
                self._switch('search_recovery',now)
                return
            # The shared frontier planner selects only observed-free routes.
            follower=HouseExplorer(self.semantic.grid,self.semantic.explorer.config)
            follower.phase='travel'
            follower.failed_goals=list(self.searched_viewpoints)
        follower.last_time=now
        self.search_recovery=follower;self.search_recovery_started=now;self.room_approach=False
        self.events.append(dict(time=now,event='stalled_search_recovery',
                                strategy='remembered_viewpoint' if isinstance(follower,RoomWaypointFollower) else 'frontier'))
        self._switch('search_recovery',now)

    def _target(self,xy,now):
        preferred=getattr(self,'preferred_target',None)
        if preferred is not None:
            # Chosen from the whole-kitchen survey (CabinetSurvey): keep it
            # rather than switching to whichever handle is nearest.
            return preferred
        candidates=[t for t in (self.semantic.targets.targets+getattr(getattr(getattr(self,'pointing',None),'targets',None),'targets',[])) if t.label==self.semantic.target_label and t.observations>=2 and 0<=now-t.last_seen<=self.readiness.max_age]
        return min(candidates,key=lambda t:np.linalg.norm(t.position[:2]-xy)) if candidates else None

    def _reuse_docking_target(self):
        return (hasattr(self.semantic.grid,'fixed_obstacle_points')
                and self.mode=='redock' and self.docking is not None
                and getattr(self.docking,'target',None) is not None
                and self.edge_look is None)

    def _tracks(self):
        return self.semantic.targets.targets+getattr(getattr(getattr(self,'pointing',None),'targets',None),'targets',[])

    def _edge_look_due(self,xy):
        """True at the close-range stop where the pickup dock is about to be planned.

        The first sighting can be metres away, where neighbouring instances
        are a few pixels wide and only one of them forms a track.
        """
        docking=self.docking;current=getattr(docking,'target',None)
        return (self.edge_look is None and getattr(docking,'edge_first',False) and self.mode=='redock'
                and current is not None and current is not self.edge_looked_for
                and getattr(docking,'path',0) is None and not docking.terminal
                and getattr(self,'pointing',None) is not None
                and hasattr(self.semantic.grid,'fixed_obstacle_points')
                and np.linalg.norm(current.position[:2]-xy)<=EDGE_LOOK_RANGE)

    def _finish_edge_look(self,packet,tracking,now):
        look,self.edge_look=self.edge_look,None
        current=self.docking.target
        seen=[t for t in self._tracks() if t is not current and t.label==self.semantic.target_label and t.observations>=2
              and len({h['timestamp'] for h in getattr(t,'history',[]) if h['timestamp']>=look['stamp']})>=2]
        chosen,measured=edge_first(current,seen,packet,tracking.T_map_base)
        self.edge_looked_for=chosen
        self.events.append(dict(time=now,event='edge_look',seconds=now-look['started'],seen=len(seen),instances=measured,
                                current=np.asarray(current.position,float).tolist(),chosen=np.asarray(chosen.position,float).tolist()))
        # Standing still to look is not docking time.
        if getattr(self.docking,'started',None) is not None:self.docking.started+=now-look['started']
        if chosen is not current:
            self.found_target=chosen;self.semantic.required_track_id=getattr(chosen,'track_id',None)
            self.docking.reset(chosen)
            self.events.append(dict(time=now,event='prefer_edge_instance',track_id=getattr(chosen,'track_id',None)))

    def step(self,packet,tracking,now):
        stop=self.semantic.follower._stop
        if self.started is None:self.started=now
        if self.terminal:return stop(self.terminal)
        if now-self.started>=self.semantic.follower.config.max_seconds:self.terminal='budget_exhausted';return stop(self.terminal)
        if not tracking.accepted:return stop('tracking_lost')
        if not 0<=now-packet.timestamp<=self.semantic.follower.config.sensor_timeout:return stop('stale_observation')
        if self.docking is not None and hasattr(self.docking,'record_support'):
            T=tracking.T_map_base
            self.docking.record_support(np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])])
        reuse_target=self._reuse_docking_target()
        if (hasattr(self.semantic.grid,'fixed_obstacle_points') and self.docking is not None
                and not (reuse_target and self.docking.path is not None)
                and packet.timestamp!=getattr(self,'last_live_points_stamp',None)):
            from .fixed_map import live_obstacle_points
            self.docking.live_points=live_obstacle_points(packet,tracking)
            self.last_live_points_stamp=packet.timestamp
        # The target was visually confirmed before this handoff. Frozen-map
        # docking follows that estimate; manipulation obtains a fresh view.
        # Resume semantic perception when returning to search/recovery.
        if not reuse_target:
            if getattr(self,'pointing',None) is not None:self.pointing.observe(packet,tracking)
            self.semantic.observe(packet,tracking)
        if self.semantic.detection_rounds<3:
            return stop('confirm_initial_observations')
        mapped=self.semantic.last_mapping or {}
        if self.goal_room:
            self.rooms.update(packet,tracking,mapped.get('rooms',[]))
        xy=tracking.T_map_base[:2,3];target=self._target(xy,packet.timestamp)
        current=self.rooms.current_label
        if self.mode=='search_departure':
            result=self.undock.step(packet,tracking,self.search_recovery.free,now)
            if result['status']=='undock_complete':
                self.undock=None
                self._start_search_recovery(xy,now)
                return stop('search_departure_complete')
            if self.undock.error:
                self.terminal=self.undock.error
            return result
        if self.mode in {'room_search','room_transit'} and (not self.goal_room or current==self.goal_room):
            if not self.goal_room or target is not None:
                self._switch('local',now)
            else:
                self._start_room_approach(xy,now)
                return stop('room_viewpoint_approach_requested')
        known_viewpoint_route=(self.mode=='search_recovery' and
            isinstance(self.search_recovery,RoomWaypointFollower) and bool(self.search_recovery.path))
        if known_viewpoint_route:self.recovery_route_follower=self.search_recovery
        elif (self.mode=='search_recovery' and self.search_recovery is not None
                and getattr(self,'recovery_route_follower',None) is self.search_recovery):
            # This recovery had a route to a remembered viewpoint; a braking-arc
            # stop clears the path after 1 s (house_exploration), which is not a
            # missing route. fix178 call 9: the second-pot search ended
            # 'exploration_required' 0.1 s into recovery. Let the recovery block
            # below replan, time out, or turn toward remembered sightings.
            known_viewpoint_route=True
            if not getattr(self,'recovery_route_kept_logged',False):
                self.recovery_route_kept_logged=True
                self.events.append(dict(time=now,event='recovery_route_cleared_by_stop_kept',
                                        detail=getattr(self.search_recovery,'stop_detail',None)))
        if (self.mode in {'room_search','search_recovery'} and
                hasattr(self.semantic.grid,'fixed_obstacle_points') and not known_viewpoint_route):
            self.terminal='exploration_required'
            return dict(status=self.terminal,forward=0.,yaw=0.,exploration_request=dict(resume_skill='local_navigation',detail=dict(reason='search_requires_exploration')))
        if self.mode=='room_search':
            self.transit.layers(xy)
            row,col=self.transit.cell(xy)
            blocked=not self.transit.free[row,col]
            if self.docking is not None and (self.undock is not None or blocked):
                from .observed_undock import ObservedUndock
                if self.undock is None:self.undock=ObservedUndock(self.docking)
                result=self.undock.step(packet,tracking,self.transit.free,now)
                if result['status']=='undock_complete':self.undock=None
                elif self.undock.error:
                    self.terminal=self.undock.error
                    return stop(self.terminal)
                else:return result
            destinations=self._room_destinations(xy)
            if destinations:
                self.transit.set_destinations(destinations)
                self.transit.last_time=now
                if self.transit._plan(xy,now) is None:self._switch('room_transit',now)
            if self.mode=='room_search':
                result=self.semantic.explorer.step(packet,tracking,now)
                if self.semantic.explorer.terminal:self.terminal=self.semantic.explorer.terminal
                return result
        if self.mode=='room_transit':
            result=self.transit.step(packet,tracking,now)
            if result['status']=='arrived_frontier':
                self.visited_room_waypoints.append(xy.copy())
                self.events.append(dict(time=now,event='room_waypoint_observed',position=xy.tolist(),confirmed_room=current))
                # Arrival at a mapped room waypoint is not semantic room proof.
                self._switch('room_search',now)
            elif self.transit.terminal:
                self._switch('room_search',now)
            return result
        if self.mode=='search_recovery':
            if target is not None:
                self._switch('local',now)
            else:
                result=self.search_recovery.step(packet,tracking,now)
                if (self.search_recovery.terminal or result['status']=='arrived_frontier'
                        or now-self.search_recovery_started>=25.):
                    goal=self.search_recovery.goal
                    self.searched_viewpoints.append(xy.copy() if goal is None else goal.copy())
                    if self.room_approach and result['status']!='arrived_frontier':
                        # Failure or timeout is not proof of reaching a viewpoint.
                        if goal is not None:self.visited_room_waypoints.append(goal.copy())
                        if self.search_recovery.terminal:
                            self.terminal='room_approach_'+self.search_recovery.terminal
                            return stop(self.terminal)
                        self._switch('room_search',now)
                        return stop('room_approach_replan')
                    self.room_approach=False
                    hints=[h for h in getattr(self.semantic,'remembered_object_hypotheses',[])
                           if h.label==self.semantic.target_label and .2<np.linalg.norm(h.position[:2]-xy)<3.]
                    if hints:
                        self.view_hint=min(hints,key=lambda h:np.linalg.norm(h.position[:2]-xy)).position.copy()
                        self.view_align_started=now
                        self._switch('view_align',now)
                    elif getattr(self,'recovery_look_hint',None) is not None:
                        self.view_hint=self.recovery_look_hint.copy();self.view_align_started=now
                        self._switch('view_align',now)
                    else:self._switch('local',now)
                    return stop('search_viewpoint_observed')
                return result
        if self.mode=='view_align':
            if target is not None:
                self._switch('local',now)
            else:
                result=self.local.face_point(packet,tracking,now,self.view_hint)
                if result['status'] in {'viewpoint_facing_ready','viewpoint_turn_blocked'} or now-self.view_align_started>12.:
                    self._switch('local',now)
                    return stop(result['status'])
                return result
        if self.mode=='local':
            # Hold a newly seen raw target proposal long enough for BotSort and
            # the VLM to inspect distinct frames. This is not a confirmed target
            # and must never authorize docking or grasping. Bound the hold so a
            # persistent false proposal cannot freeze search indefinitely.
            pending=self.semantic.target_label in getattr(self.semantic,'pending_categories',set())
            if target is None and pending:
                if self.proposal_inspection_started is None:
                    self.proposal_inspection_started=now;self.local.reset_motion()
                if now-self.proposal_inspection_started<3.2:
                    return stop('confirm_raw_target_proposal')
            else:self.proposal_inspection_started=None
            if target is None:
                tentative=[t for t in self.semantic.targets.targets if t.label==self.semantic.target_label
                           and t.observations==1 and 0<=packet.timestamp-t.last_seen<=self.readiness.max_age]
                for sighting in tentative:
                    key=(getattr(sighting,'track_id',id(sighting)),sighting.last_seen)
                    if key not in self.tentative_attempted:
                        self.tentative_attempted.add(key);self.tentative_until=now+2.2
                        break
                if self.tentative_until is not None and now<self.tentative_until:
                    return stop('confirm_tentative_target')
            else:self.tentative_until=None
            if target is None and self._search_stalled(xy,now):
                self._start_search_recovery(xy,now)
                return stop('stalled_search_recovery')
            if target is not None:
                self.last_readiness=self.readiness.check(target,packet,tracking,self.semantic.grid)
                if self.last_readiness.ready:self.terminal='manipulation_ready';return stop(self.terminal)
            moved=self.handoff_blocked_at is None or np.linalg.norm(xy-self.handoff_blocked_at)>=.25
            if moved and self.docking is not None and target is not None and 0<=packet.timestamp-target.last_seen<=self.readiness.max_age:
                # A repeatedly seen target can start rolling approach without
                # waiting for One-Ring's navigation-distance stop criterion.
                self.done_at=now;self.confirmation_since=now-self.readiness.max_age;self._switch('confirm_found',now)
                return stop('confirm_visible_target')
            result=self.local.step(packet,tracking,now)
            if result['status']=='exploration_required':
                # One suggested local step may leave observed support while
                # another viewing position is reachable on the frozen map.
                # Try that route before escalating to new-space exploration.
                if hasattr(self.semantic.grid,'fixed_obstacle_points'):
                    self._start_search_recovery(xy,now)
                    return stop('local_step_requires_viewpoint_recovery')
                self.terminal='exploration_required';return result
            if result['status']=='ring_collision_stop':
                self.blocked_predictions+=1
                if self.blocked_predictions>=3 and target is not None:
                    return self._request_redock(now)
            elif scan_stop_count_resets(result['status']):
                self.blocked_predictions=0
            if result['status']=='local_done':
                self.done_at=now;self.confirmation_since=now
                self._switch('confirm_found',now)
                return stop('confirm_found_object')
            return result
        if self.mode=='confirm_found':
            fresh=[t for t in (self.semantic.targets.targets+getattr(getattr(getattr(self,'pointing',None),'targets',None),'targets',[]))
                   if t.label==self.semantic.target_label and t.observations>=2
                   and 0<=packet.timestamp-t.last_seen<=self.readiness.max_age
                   and len({h['timestamp'] for h in getattr(t,'history',[])
                            if h['timestamp']>=max(self.confirmation_since if self.confirmation_since is not None else self.done_at,packet.timestamp-self.readiness.max_age)})>=2]
            if not fresh:
                if now-self.done_at>=3.2:self.terminal='object_found_unconfirmed';return stop(self.terminal)
                return stop('confirm_found_object')
            self.found_target=min(fresh,key=lambda t:np.linalg.norm(t.position[:2]-xy))
            if getattr(self.docking,'edge_first',False):
                self.found_target,measured=edge_first(self.found_target,fresh,packet,tracking.T_map_base)
                if measured:self.events.append(dict(time=now,event='edge_ranked_instances',instances=measured))
            self.object_search_success=True
            self.events.append(dict(time=now,event='object_found_verified',track_id=getattr(self.found_target,'track_id',None)))
            if getattr(self,'fixture_handle',False):
                # Any eligible cabinet/drawer counts; prefer a handle the arm
                # can reach (a wall cabinet handle at 1.6 m was not).
                reachable=[t for t in fresh if HANDLE_HEIGHT[0]<=float(t.position[2])<=HANDLE_HEIGHT[1]]
                # Identity, not ==: targets hold numpy arrays and == raised
                # 'truth value of an array is ambiguous' (fix66 drawer navigation).
                if reachable and not any(t is self.found_target for t in reachable):
                    self.found_target=min(reachable,key=lambda t:np.linalg.norm(t.position[:2]-xy))
            reach=float(np.linalg.norm(self.found_target.position[:2]-xy))
            if getattr(self,'fixture_handle',False) and reach<=FIXTURE_HANDOFF_DISTANCE:
                # A door/drawer handle is confirmed nearby. The fixture skill
                # redocks in front of the handle it observes itself (Qwen-ranked
                # stand-offs), so do not run the object pickup dock here: with
                # several identical cabinets it looped between a blocked dock and
                # blocked search turns until redocking was exhausted.
                self.events.append(dict(time=now,event='fixture_handle_handoff',distance_m=reach))
                self.terminal='fixture_handle_found';return stop(self.terminal)
            if getattr(self,'confined_target',False) and reach<=FIXTURE_HANDOFF_DISTANCE:
                # An object inside an opened fridge or microwave is confirmed
                # nearby. Its pick docks in front of the opening by itself
                # (aligned front entry); the open-surface pickup dock here
                # looked for a counter edge that an appliance shelf does not
                # have and ran out its time (fix105: 452 s, shape_dock_timeout).
                self.events.append(dict(time=now,event='confined_target_handoff',distance_m=reach))
                self.terminal='confined_target_found';return stop(self.terminal)
            self.last_readiness=self.readiness.check(self.found_target,packet,tracking,self.semantic.grid)
            if self.last_readiness.ready:self.terminal='manipulation_ready';return stop(self.terminal)
            if self.last_readiness.reason=='reachability_unverified' and self.docking is None:
                self.terminal='reachability_unverified';return stop(self.terminal)
            return self._request_redock(now)
        if self.docking is not None:
            # Choose among neighbouring instances before the dock pose exists:
            # stand still at the close-range stop, look, then dock once.
            if self._edge_look_due(xy):
                self.edge_look=dict(started=now,stamp=packet.timestamp)
                self.events.append(dict(time=now,event='edge_look_started',track_id=getattr(self.docking.target,'track_id',None),
                                        distance_m=float(np.linalg.norm(self.docking.target.position[:2]-xy))))
                return stop('edge_look')
            if self.edge_look is not None:
                rounds=len({e['timestamp'] for e in self.pointing.events if e['timestamp']>=self.edge_look['stamp']})
                if rounds<EDGE_LOOK_ROUNDS and now-self.edge_look['started']<EDGE_LOOK_SECONDS:return stop('edge_look')
                self._finish_edge_look(packet,tracking,now)
                return stop('edge_look_complete')
            # Category search may discover a better instance in the wrist view.
            # Change only while stopped, with fresh repeated visual evidence.
            current=getattr(self.docking,"target",None)
            alternate=None if reuse_target else self._target(xy,packet.timestamp)
            if current is not None and alternate is not None and alternate is not current and np.linalg.norm(getattr(self.docking,'velocity',np.zeros(2)))<.01:
                recent={h['timestamp'] for h in getattr(alternate,'history',[]) if packet.timestamp-self.readiness.max_age<=h['timestamp']<=packet.timestamp}
                if len(recent)>=2 and np.linalg.norm(alternate.position[:2]-xy)+.12<np.linalg.norm(current.position[:2]-xy):
                    self.found_target=alternate
                    self.events.append(dict(time=now,event='prefer_closer_observed_instance',track_id=getattr(alternate,'track_id',None)))
                    return self._request_redock(now)
            result=self.docking.step(packet,tracking,now)
            if result['status']=='exploration_required':
                self.terminal='exploration_required';return result
            if self.docking.terminal=='shape_route_unverified':
                # Do not ask the image navigator to repeat blocked turns when
                # the geometric planner explicitly needs floor observations.
                # The pipeline can execute a checked wrist scan, then continue
                # from this exact physical state with the accumulated map.
                recent=getattr(self.docking,'events',[])[-6:]
                missing_floor=any(
                    e.get('detail',{}).get('reason')=='unobserved_base_support'
                    or any(r.get('detail',{}).get('reason')=='unobserved_base_support' for r in e.get('rejected',[]))
                    for e in recent)
                if missing_floor:
                    self.terminal='needs_floor_observation'
                    return stop(self.terminal)
            if self.docking.terminal=='shape_route_unverified' and self.observation_recoveries<3:
                self.observation_recoveries+=1;self.handoff_blocked_at=xy.copy()
                self.events.append(dict(time=now,event='resume_visual_search_for_clearance',attempt=self.observation_recoveries))
                self._switch('local',now)
                return stop('observe_approach_corridor')
            if self.docking.terminal:
                self.docking_success=self.docking.terminal=='shape_docked'
                if self.docking_success:
                    self.last_readiness=self.readiness.check(self.docking.target,packet,tracking,self.semantic.grid)
                    self.terminal='manipulation_ready' if self.last_readiness.ready else 'docked_'+self.last_readiness.reason
                else:self.terminal=self.docking.terminal
                return stop(self.terminal)
            return result
        result=self.semantic.step(packet,tracking,now)
        if result['status']=='redock_needs_observation':
            self.terminal='redock_route_unverified'
            return stop(self.terminal)
        if self.semantic.terminal:
            if self.semantic.terminal=='target_reached':
                self.last_readiness=self.readiness.check(self.semantic.follower.target,packet,tracking,self.semantic.grid)
                if self.last_readiness.ready:self.terminal='manipulation_ready'
                elif self.last_readiness.reason=='reachability_unverified':self.terminal='reachability_unverified'
                else:
                    self.terminal='redock_'+self.last_readiness.reason
            else:self.terminal='redock_'+self.semantic.terminal
        return stop(self.terminal) if self.terminal else result
