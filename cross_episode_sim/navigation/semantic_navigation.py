"""Partial-map object navigation, with exploration only while no route is known.

Public inputs: onboard SensorPacket, visual SLAM TrackingResult, RGB detector.
No simulator, object IDs, room graph, or environment collision queries.
"""
from dataclasses import replace
import heapq
import math
import time
import numpy as np
from .house_exploration import HouseExplorer, HouseExplorationConfig, reachable_paths
from .local_scan import ScanConfig
from ..perception.visual_targets import VisualTargetMap
from .object_view import visible_object_heading


def astar_observed(free, start, goal):
    """Eight-connected A* over observed free cells, without diagonal cuts."""
    if not free[start] or not free[goal]:return []
    queue=[(math.dist(start,goal),0.,start)];cost={start:0.};parent={}
    while queue:
        _,g,node=heapq.heappop(queue)
        if g>cost[node]:continue
        if node==goal:
            path=[node]
            while node in parent:node=parent[node];path.append(node)
            return path[::-1]
        for dr,dc in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt=node[0]+dr,node[1]+dc
            if not(0<=nxt[0]<free.shape[0] and 0<=nxt[1]<free.shape[1]) or not free[nxt]:continue
            if dr and dc and not(free[node[0]+dr,node[1]] and free[node[0],node[1]+dc]):continue
            ng=g+math.hypot(dr,dc)
            if ng>=cost.get(nxt,float('inf')):continue
            cost[nxt]=ng;parent[nxt]=node;heapq.heappush(queue,(ng+math.dist(nxt,goal),ng,nxt))
    return []


class TargetFollower(HouseExplorer):
    def __init__(self, grid, config, preferred_distance=.55):
        if not .48<=preferred_distance<=1.5:raise ValueError("Preferred distance must be between 0.48 and 1.5 m")
        self.preferred_distance=preferred_distance;self.max_distance=preferred_distance+.17
        super().__init__(grid,replace(config,waypoint_tolerance=.08));self.phase='travel';self.target=None;self.camera=None;self.base_pose=None;self.view_heading=None;self.rejected_views={}

    def _lookahead_reached(self,xy):
        return self.goal is not None and np.linalg.norm(self.goal-xy)<self.config.waypoint_tolerance

    def _plan(self,xy,now):
        self.last_plan=now;self.plans+=1
        distance,_=self.layers(xy)
        costs,_=reachable_paths(self.free,self.cell(xy),distance)
        nodes=np.argwhere(np.isfinite(costs));points=self.world(nodes)
        if not len(nodes):return 'departure_blocked'
        separation=np.linalg.norm(points-self.target.position[:2],axis=1)
        # Navigation stand-off only; manipulation reachability remains unqualified.
        eligible=(separation>=.48)&(separation<=self.max_distance)
        for rejected in self.rejected_views.get(self.target.track_id,[]):
            eligible &= np.linalg.norm(points-rejected,axis=1)>.35
        candidates=np.flatnonzero(eligible)
        if not len(candidates):return 'target_not_reachable_yet'
        ranked=sorted(candidates,key=lambda i:costs[tuple(nodes[i])]*self.grid.config.resolution+6*abs(separation[i]-self.preferred_distance))
        best=None
        for i in ranked:
            heading=visible_object_heading(self.base_pose,self.target.position,self.camera,points[i]) if self.camera is not None else 0.
            if heading is not None:best=i;self.view_heading=heading;break
        if best is None:return 'no_observed_viewpoint'
        node=tuple(nodes[best]);route=astar_observed(self.free,self.cell(xy),node)
        if not route:return 'target_not_reachable_yet'
        remaining=list(self.world(route));simplified=[xy.copy()]
        while remaining:
            j=len(remaining)-1
            while j>=0 and not self._line_free(simplified[-1],remaining[j]):j-=1
            if j<0:return 'start_connector_blocked'
            simplified.append(remaining[j]);remaining=remaining[j+1:]
        self.goal=points[best];self.path=simplified[1:]
        if self.last_progress is None:
            self.last_progress=xy.copy();self.progress_time=now
        self.events.append(dict(time=now,event='target_astar',track_id=self.target.track_id,route=[p.tolist() for p in simplified]))
        return None


class SemanticNavigator:
    def __init__(self,grid,detector,target='dining table',explorer=None,config=HouseExplorationConfig(),detection_period=1.,object_mapper=None,preferred_distance=.55,allow_exploration=True):
        self.required_track_id=None
        self.allow_exploration=allow_exploration
        self.grid=grid;self.detector=detector;self.target_label=target
        self.explorer=explorer or HouseExplorer(grid,config)
        self.follower=TargetFollower(grid,config,preferred_distance);self.object_mapper=object_mapper
        if object_mapper is not None:
            from .hydra_targets import HydraTargetMap
            if target not in {'cup','bowl'}:raise ValueError('Hydra object navigation supports cup/bowl')
            self.targets=HydraTargetMap()
        else:self.targets=VisualTargetMap()
        self.detection_rounds=0;self.last_mapping=None
        self.resense_until=None;self.resense_attempts=0
        self.last_detection=-np.inf;self.detection_period=detection_period
        self.detection_wall_seconds=0.;self.detection_calls=0
        self.last_target_probe=-np.inf;self.retry_after={};self.attempts={};self.attempt_observations={}
        self.view_offsets=(0.,.45,-.45,.7,-.7);self.view_index=0;self.view_arrived_at=None
        self.inspection_started=None;self.view_plan_stamp=None;self.planned_heading=None;self.needs_confirmation=False
        self.mode='explore';self.terminal=None;self.started=None;self.last_detections={};self.alignment_started=None
        self.remembered_object_hypotheses=[]

    def _candidate_allowed(self,target,now):
        if self.required_track_id is not None and target.track_id!=self.required_track_id:return False
        # A single sensor-verified sighting warrants inspection, not success.
        if target.label!=self.target_label or target.observations<1:return False
        if now<self.retry_after.get(target.track_id,-np.inf):return False
        if self.attempts.get(target.track_id,0)>=3:
            return target.observations>self.attempt_observations.get(target.track_id,0)
        return True

    def _recover_target(self,xy,now,reason):
        target=self.follower.target
        if target is not None:
            self.follower.rejected_views.setdefault(target.track_id,[]).append(xy.copy())
            self.retry_after[target.track_id]=now+3.
        self.follower.events.append(dict(time=now,event='target_recovery',reason=reason,
                                        track_id=None if target is None else target.track_id))
        self.mode='explore';self.follower.terminal=None;self.follower.path=[]
        self.follower.goal=None;self.follower.target=None
        self.alignment_started=None;self.view_plan_stamp=None
        # A paused exploration path may no longer start at the current position.
        self.explorer.path=[];self.explorer.goal=None;self.explorer.last_time=now
        self.explorer.velocity=np.zeros(2)
        return self.follower._stop('recover_'+reason)

    def observe(self,packet,tracking):
        """Update sensor-derived object memory without selecting or executing motion."""
        if not tracking.accepted:return
        if packet.timestamp-self.last_detection>=self.detection_period:
            self.last_detection=packet.timestamp
            self.needs_confirmation=False
            if hasattr(self.detector,'set_map_targets'):self.detector.set_map_targets(self.targets.targets+self.remembered_object_hypotheses,tracking)
            for name,camera in packet.cameras.items():
                started=time.perf_counter()
                detections=(self.detector.detect_camera(camera,packet.timestamp) if hasattr(self.detector,'detect_camera') else self.detector.detect(camera.rgb))
                self.detection_wall_seconds+=time.perf_counter()-started;self.detection_calls+=1
                self.last_detections[name]=detections
                self.needs_confirmation |= getattr(self.detector,'needs_confirmation',False)
                if self.object_mapper is None:self.targets.update(camera,detections,tracking,packet.timestamp)
            if self.object_mapper is not None:
                self.last_mapping=(self.object_mapper.update_many(packet.cameras,tracking,self.last_detections,packet.timestamp)
                                   if hasattr(self.object_mapper,'update_many') else
                                   self.object_mapper.update(packet.cameras['navigation'],tracking,self.last_detections.get('navigation',[]),packet.timestamp))
                if self.last_mapping is not None:self.targets.update(self.last_mapping['objects'],packet.cameras,self.last_detections,tracking,packet.timestamp)
            self.detection_rounds+=1

    def step(self,packet,tracking,now):
        if self.started is None:self.started=now
        if self.terminal:return self.follower._stop(self.terminal)
        if now-self.started>=self.follower.config.max_seconds:
            self.terminal='budget_exhausted';return self.follower._stop(self.terminal)
        if not tracking.accepted:return self.follower._stop('tracking_lost')
        if not 0<=now-packet.timestamp<=self.follower.config.sensor_timeout:return self.follower._stop('stale_observation')
        if not np.allclose(packet.arm.position,self.follower.config.arm_posture,atol=.05):return self.follower._stop('arm_not_stowed')
        self.observe(packet,tracking)
        # Upstream track confirmation needs distinct observations before motion
        # changes the view. Stop only until those observations have been taken.
        if self.object_mapper is not None and self.detection_rounds<3:
            return self.follower._stop('confirm_initial_observations')
        if self.needs_confirmation and self.mode=='explore' and not any(t.label==self.target_label and t.observations>=2 for t in self.targets.targets):
            if self.inspection_started is None:self.inspection_started=now
            if now-self.inspection_started<3.2:return self.follower._stop('confirm_candidate_observations')
        else:self.inspection_started=None
        if self.mode!='explore' and self.follower.target is not None and not any(t is self.follower.target for t in self.targets.targets):
            self.mode='explore';self.follower.terminal=None;self.follower.path=[]
        self.follower.camera=packet.cameras['navigation'];self.follower.base_pose=tracking.T_map_base
        xy=tracking.T_map_base[:2,3]
        if self.resense_until is not None:
            if now<self.resense_until:return self.follower._stop('resense_blocked_departure')
            self.resense_until=None;self.explorer.terminal=None
            self.explorer.path=[];self.explorer.goal=None;self.explorer.last_time=now
            self.explorer.velocity=np.zeros(2)
        if self.mode=='explore':
            # Inspect tentative sightings as well as confirmed targets, without changing exploration
            # if the partial map cannot yet connect to their stand-off position.
            candidates=self.targets.targets if now-self.last_target_probe>=.5 else []
            if candidates:self.last_target_probe=now
            for target in sorted(candidates,key=lambda t:np.linalg.norm(t.position[:2]-xy)):
                if not self._candidate_allowed(target,now):continue
                if self.object_mapper is None and now-target.last_seen>15. and not getattr(target,'navigation_memory',False):continue
                self.explorer._sync_support();self.follower._sync_support()
                self.follower.support|=self.explorer.support
                self.follower.target=target
                if self.follower._plan(xy,now) is None:
                    self.follower.phase='travel';self.follower.terminal=None
                    self.follower.last_time=now;self.follower.velocity=np.zeros(2)
                    self.follower.last_progress=xy.copy();self.follower.progress_time=now
                    self.attempts[target.track_id]=self.attempts.get(target.track_id,0)+1
                    self.attempt_observations[target.track_id]=target.observations
                    self.mode='navigate';break
            if self.mode=='explore':
                if not self.allow_exploration:
                    return self.follower._stop('redock_needs_observation')
                result=self.explorer.step(packet,tracking,now)
                if self.explorer.terminal:
                    if self.explorer.terminal in {'departure_blocked','tare_no_path_timeout'} and self.resense_attempts<2:
                        self.resense_attempts+=1;self.resense_until=now+3.
                        self.follower.events.append(dict(time=now,event='resense',reason=self.explorer.terminal))
                        return self.follower._stop('resense_blocked_departure')
                    self.terminal=self.explorer.terminal
                return result
        if self.mode=='align':
            pose=tracking.T_map_base;yaw=math.atan2(pose[1,0],pose[0,0]);delta=self.follower.target.position[:2]-xy
            # A partial surface estimate can shift as more of a table becomes
            # visible. Turning cannot repair a stand-off that is now too far.
            target=self.follower.target
            distance=float(np.linalg.norm(delta))
            if target.observations>=2 and now-target.last_seen<2. and not .45<=distance<=self.follower.max_distance+.08:
                error=self.follower._plan(xy,now)
                if error is not None:return self._recover_target(xy,now,'refined_target_'+error)
                self.follower.phase='travel';self.follower.terminal=None
                self.follower.last_time=now;self.follower.last_progress=xy.copy();self.follower.progress_time=now
                self.mode='navigate';self.alignment_started=None
                self.follower.events.append(dict(time=now,event='replan_refined_target',distance=distance))
                return self.follower._stop('target_geometry_updated')
            error=math.atan2(math.sin(math.atan2(delta[1],delta[0])-yaw),math.cos(math.atan2(delta[1],delta[0])-yaw))
            self.follower.layers(xy);self.follower._depth(packet.cameras['navigation'])
            confirmations=sum(h['timestamp']>=self.alignment_started for h in self.follower.target.history)
            if abs(error)<.8 and .45<=np.linalg.norm(delta)<=self.follower.max_distance+.08 and now-self.follower.target.last_seen<2. and confirmations>=2:
                self.terminal='target_reached';return self.follower._stop(self.terminal)
            if now-self.alignment_started>20.:
                return self._recover_target(xy,now,'confirmation_failed')
            # The stowed arm can occlude the centre of the mast camera. Take
            # checked alternative headings, using fresh mask/depth evidence.
            if self.view_plan_stamp!=packet.timestamp:
                self.planned_heading=visible_object_heading(pose,self.follower.target.position,packet.cameras['navigation']);self.view_plan_stamp=packet.timestamp
            planned_heading=self.planned_heading
            desired_heading=(planned_heading if planned_heading is not None else yaw+error)+self.view_offsets[self.view_index]
            view_error=math.atan2(math.sin(desired_heading-yaw),math.cos(desired_heading-yaw))
            if abs(view_error)<.08:
                if self.view_arrived_at is None:self.view_arrived_at=now
                elif now-self.view_arrived_at>=2.2 and self.view_index+1<len(self.view_offsets):
                    self.view_index+=1;self.view_arrived_at=None
                    self.follower.events.append(dict(time=now,event='confirmation_view_change',heading_offset=self.view_offsets[self.view_index]))
            else:self.view_arrived_at=None
            desired=float(np.clip(1.8*view_error,-.4,.4));previous=self.follower.velocity[1]
            command=np.array([0.,previous+np.clip(desired-previous,-.016,.016)])
            if not self.follower._command_clear(xy,yaw,command):return self.follower._stop('alignment_clearance_stop')
            self.follower.velocity=command
            return dict(status='align_target',forward=0.,yaw=float(command[1]),view_offset=self.view_offsets[self.view_index])
        result=self.follower.step(packet,tracking,now)
        if result['status']=='arrived_frontier':
            self.mode='align';self.alignment_started=now;self.view_index=0;self.view_arrived_at=None
            return self.follower._stop('at_target_standoff')
        if result['status']=='stalled_replan':self.follower.terminal='target_route_stalled'
        if self.follower.terminal:return self._recover_target(xy,now,self.follower.terminal)
        return result
