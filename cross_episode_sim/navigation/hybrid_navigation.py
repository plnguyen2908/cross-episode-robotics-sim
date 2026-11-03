"""Room-scale SLAM routing, local RING, then measured readiness / A* redocking."""
import numpy as np
from .house_exploration import HouseExplorer
from .semantic_navigation import astar_observed

class RoomWaypointFollower(HouseExplorer):
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
        self.tentative_attempted=set();self.tentative_until=None
        self.confirmation_since=None;self.done_at=None;self.found_target=None;self.object_search_success=False
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
        points=[]
        for hint in hints:
            if hint.label!=self.semantic.target_label:continue
            for radius in (1.,1.4):
                for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
                    point=hint.position[:2]+radius*np.array([np.cos(angle),np.sin(angle)])
                    if np.linalg.norm(point-xy)<.6:continue
                    if any(np.linalg.norm(point-v)<.6 for v in self.searched_viewpoints):continue
                    points.append(point)
        points.sort(key=lambda p:np.linalg.norm(p-xy))
        follower=RoomWaypointFollower(self.semantic.grid,self.semantic.explorer.config)
        follower.set_destinations(points)
        if not points or follower._plan(xy,now) is not None:
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
        candidates=[t for t in self.semantic.targets.targets if t.label==self.semantic.target_label and t.observations>=2 and 0<=now-t.last_seen<=self.readiness.max_age]
        return min(candidates,key=lambda t:np.linalg.norm(t.position[:2]-xy)) if candidates else None
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
        self.semantic.observe(packet,tracking)
        if self.semantic.detection_rounds<3:
            return stop('confirm_initial_observations')
        mapped=self.semantic.last_mapping or {}
        self.rooms.update(packet,tracking,mapped.get('rooms',[]))
        xy=tracking.T_map_base[:2,3];target=self._target(xy,packet.timestamp)
        current=self.rooms.current_label
        if self.mode in {'room_search','room_transit'} and (not self.goal_room or current==self.goal_room):
            if not self.goal_room or target is not None:
                self._switch('local',now)
            else:
                self._start_room_approach(xy,now)
                return stop('room_viewpoint_approach_requested')
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
            if result['status']=='ring_collision_stop':
                self.blocked_predictions+=1
                if self.blocked_predictions>=3 and target is not None:
                    return self._request_redock(now)
            elif result['status'] not in {'await_fresh_ring_observation','ring_action_timeout'}:
                self.blocked_predictions=0
            if result['status']=='local_done':
                self.done_at=now;self.confirmation_since=now
                self._switch('confirm_found',now)
                return stop('confirm_found_object')
            return result
        if self.mode=='confirm_found':
            fresh=[t for t in self.semantic.targets.targets
                   if t.label==self.semantic.target_label and t.observations>=2
                   and 0<=packet.timestamp-t.last_seen<=self.readiness.max_age
                   and len({h['timestamp'] for h in getattr(t,'history',[])
                            if h['timestamp']>=max(self.confirmation_since if self.confirmation_since is not None else self.done_at,packet.timestamp-self.readiness.max_age)})>=2]
            if not fresh:
                if now-self.done_at>=3.2:self.terminal='object_found_unconfirmed';return stop(self.terminal)
                return stop('confirm_found_object')
            self.found_target=min(fresh,key=lambda t:np.linalg.norm(t.position[:2]-xy))
            self.object_search_success=True
            self.events.append(dict(time=now,event='object_found_verified',track_id=getattr(self.found_target,'track_id',None)))
            self.last_readiness=self.readiness.check(self.found_target,packet,tracking,self.semantic.grid)
            if self.last_readiness.ready:self.terminal='manipulation_ready';return stop(self.terminal)
            if self.last_readiness.reason=='reachability_unverified' and self.docking is None:
                self.terminal='reachability_unverified';return stop(self.terminal)
            return self._request_redock(now)
        if self.docking is not None:
            # Category search may discover a better instance in the wrist view.
            # Change only while stopped, with fresh repeated visual evidence.
            current=getattr(self.docking,"target",None)
            alternate=self._target(xy,packet.timestamp)
            if current is not None and alternate is not None and alternate is not current and np.linalg.norm(getattr(self.docking,'velocity',np.zeros(2)))<.01:
                recent={h['timestamp'] for h in getattr(alternate,'history',[]) if packet.timestamp-self.readiness.max_age<=h['timestamp']<=packet.timestamp}
                if len(recent)>=2 and np.linalg.norm(alternate.position[:2]-xy)+.12<np.linalg.norm(current.position[:2]-xy):
                    self.found_target=alternate
                    self.events.append(dict(time=now,event='prefer_closer_observed_instance',track_id=getattr(alternate,'track_id',None)))
                    return self._request_redock(now)
            result=self.docking.step(packet,tracking,now)
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
