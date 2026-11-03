"""Whole-house frontier exploration from onboard observations only.

No room count, graph, object IDs, global robot pose or simulator access. The
planning layer is observed floor + a separately recorded ego support prior;
3D depth obstacles and online swept-motion checks remain authoritative.
"""
from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.spatial import cKDTree
from .local_scan import LocalScanSkill, ScanConfig


@dataclass(frozen=True)
class HouseExplorationConfig:
    # Expected retained posture only; physical base bridge still requires swept clearance.
    arm_posture: tuple = ScanConfig().stowed_arm
    max_seconds: float = 600.
    footprint_radius: float = .45
    center_clearance: float = .56
    ego_support_radius: float = .40
    speed: float = .4
    yaw_speed: float = .6
    acceleration: float = .4
    yaw_acceleration: float = .8
    waypoint_tolerance: float = .2
    approach_gain: float = 1.2
    sensor_timeout: float = .25
    scan_spacing: float = 1.


def reachable_paths(free,start,clearance):
    """Dijkstra with obstacle clearance penalty; no diagonal corner cutting."""
    costs=np.full(free.shape,np.inf);parents={}
    if not (0<=start[0]<free.shape[0] and 0<=start[1]<free.shape[1] and free[start]):return costs,parents
    costs[start]=0;queue=[(0.,start)]
    while queue:
        cost,node=heapq.heappop(queue)
        if cost>costs[node]:continue
        for dr,dc in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt=node[0]+dr,node[1]+dc
            if not (0<=nxt[0]<free.shape[0] and 0<=nxt[1]<free.shape[1] and free[nxt]):continue
            if dr and dc and not(free[node[0]+dr,node[1]] and free[node[0],node[1]+dc]):continue
            value=cost+math.hypot(dr,dc)*(1.+.15/max(.1,clearance[nxt]))
            if value<costs[nxt]:costs[nxt]=value;parents[nxt]=node;heapq.heappush(queue,(value,nxt))
    return costs,parents


class HouseExplorer:
    def __init__(self,grid,config=HouseExplorationConfig()):
        self.grid=grid;self.config=config;self.support=np.zeros(grid.floor_views.shape,bool);self.support_origin=grid.origin.copy()
        self.scanner=LocalScanSkill(ScanConfig(stowed_arm=self.config.arm_posture));self.phase='scan';self.scan_locations=[];self.failed_goals=[]
        self.path=[];self.goal=None;self.velocity=np.zeros(2);self.last_time=None;self.started=None
        self.last_sensor=None;self.last_plan=-np.inf;self.events=[];self.terminal=None
        self.last_progress=None;self.progress_time=0.;self.free=None;self.obstacles=None;self.near_points=np.empty((0,3))
        self.turn_recovery_active=False
        self.scans=0;self.plans=0;self.traveled=0.;self.previous_xy=None;self.stop_detail=None

    def cell(self,xy):
        col,row=self.grid.world_to_cell(xy);return int(row),int(col)

    def world(self,cells):return self.grid.cell_to_world(np.asarray(cells)[:,::-1])

    def _sync_support(self):
        if self.support.shape!=self.grid.floor_views.shape or not np.array_equal(self.support_origin,self.grid.origin):
            shift=np.rint((self.support_origin-self.grid.origin)/self.grid.config.resolution).astype(int)
            row,col=int(shift[1]),int(shift[0]);height,width=self.support.shape
            expanded=np.zeros(self.grid.floor_views.shape,bool)
            expanded[row:row+height,col:col+width]=self.support
            self.support=expanded;self.support_origin=self.grid.origin.copy()
    def layers(self,xy):
        self._sync_support()
        c=self.grid.config;row,col=self.cell(xy);r=int(np.ceil(self.config.ego_support_radius/c.resolution))+1
        r0,r1=max(0,row-r),min(self.support.shape[0],row+r+1);c0,c1=max(0,col-r),min(self.support.shape[1],col+r+1)
        yy,xx=np.mgrid[r0:r1,c0:c1];points=self.grid.origin+(np.stack((xx,yy),-1)+.5)*c.resolution
        # Kept separate from camera evidence. Never overrides observed obstacles.
        self.support[r0:r1,c0:c1]|=np.linalg.norm(points-xy,axis=-1)<self.config.ego_support_radius
        self.obstacles=(self.grid.evidence>=2).any(axis=2)
        floor=self.grid.floor_views>=c.minimum_floor_views
        supported=(floor|self.support)&~self.obstacles
        distance=distance_transform_edt(np.pad(supported,1))[1:-1,1:-1]*c.resolution
        self.free=supported&(distance>self.config.center_clearance)
        frontier=floor&~self.obstacles&binary_dilation(~(floor|self.support|self.obstacles))
        return distance,frontier

    def _depth(self,c):
        v,u=np.mgrid[0:c.depth_m.shape[0]:4,0:c.depth_m.shape[1]:4];z=c.depth_m[::4,::4]
        good=np.isfinite(z)&c.valid_depth[::4,::4]
        if c.robot_mask is not None:good &= ~c.robot_mask[::4,::4]
        v,u,z=v[good],u[good],z[good]
        p=np.column_stack(((u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z))
        p=p@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3]
        self.near_points=p[(p[:,2]>.05)&(p[:,2]<1.85)]

    def _line_free(self,a,b):
        # Check every crossed cell, including arbitrarily short corner clips.
        # Fixed-distance samples can jump over a blocked cell near a corner and
        # create a shortcut that the denser physical braking guard cannot follow.
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);delta=b-a
        crossings=[0.,1.];resolution=self.grid.config.resolution
        for axis in range(2):
            if abs(delta[axis])<1e-12:continue
            origin=self.grid.origin[axis]
            lo=int(np.floor((min(a[axis],b[axis])-origin)/resolution))+1
            hi=int(np.floor((max(a[axis],b[axis])-origin)/resolution))
            for index in range(lo,hi+1):
                t=(origin+index*resolution-a[axis])/delta[axis]
                if 0.<t<1.:crossings.append(t)
        crossings=np.unique(crossings)
        samples=np.r_[0.,1.,(crossings[:-1]+crossings[1:])/2]
        for t in samples:
            cell=self.cell(a+t*delta)
            if not(0<=cell[0]<self.free.shape[0] and 0<=cell[1]<self.free.shape[1] and self.free[cell]):return False
        return True

    def _plan(self,xy,now):
        old_goal=None if self.goal is None else self.goal.copy()
        distance,frontier=self.layers(xy);costs,parents=reachable_paths(self.free,self.cell(xy),distance)
        nodes=np.argwhere(np.isfinite(costs));self.last_plan=now;self.plans+=1
        if not len(nodes):return 'departure_blocked'
        groups,count=label(frontier,np.ones((3,3),bool));tree=cKDTree(nodes);choices=[]
        for i in range(1,count+1):
            cells=np.argwhere(groups==i)
            if len(cells)<4:continue
            # Prefer a clear stand-off viewpoint rather than hugging the frontier.
            gap,_=cKDTree(cells).query(nodes)
            points=self.world(nodes)
            eligible=(gap*self.grid.config.resolution<=1.5)&(np.linalg.norm(points-xy,axis=1)>=.65)
            for visited in self.scan_locations+self.failed_goals:
                eligible &= np.linalg.norm(points-visited,axis=1)>=self.config.scan_spacing
            indices=np.flatnonzero(eligible)
            if not len(indices):continue
            merit=np.minimum(distance[nodes[indices,0],nodes[indices,1]],1.2)-.05*gap[indices]*self.grid.config.resolution-.03*costs[nodes[indices,0],nodes[indices,1]]*self.grid.config.resolution
            j=indices[int(np.argmax(merit))];node=tuple(nodes[j]);point=points[j]
            travel=costs[node]*self.grid.config.resolution
            score=min(len(cells),100)**.5/(1.+.35*travel)
            choices.append((score,node,point))
        if not choices:return 'no_reachable_frontier'
        _,node,self.goal=max(choices,key=lambda x:x[0]);path=[node]
        while node in parents:node=parents[node];path.append(node)
        remaining=list(self.world(path[::-1]));simplified=[xy.copy()]
        while remaining:
            j=len(remaining)-1
            while j>=0 and not self._line_free(simplified[-1],remaining[j]):j-=1
            if j<0:return 'start_connector_blocked'
            simplified.append(remaining[j]);remaining=remaining[j+1:]
        self.path=list(simplified[1:]);self.events.append(dict(time=now,event='frontier_goal',goal=self.goal.tolist(),route=[p.tolist() for p in simplified]))
        if self.last_progress is None or old_goal is None or np.linalg.norm(self.goal-old_goal)>.4:
            self.last_progress=xy.copy();self.progress_time=now
        return None

    def _lookahead_reached(self,xy):
        return False

    def _stop(self,status):
        self.velocity[:]=0;return dict(status=status,forward=0.,yaw=0.,scans=self.scans,plans=self.plans)

    def step(self,packet,tracking,now):
        dt=.02 if self.last_time is None else now-self.last_time;self.last_time=now
        if self.started is None:self.started=now
        if self.terminal:return self._stop(self.terminal)
        if now-self.started>=self.config.max_seconds:self.terminal='budget_exhausted';return self._stop(self.terminal)
        if not tracking.accepted:return self._stop('tracking_lost')
        if not 0<=now-packet.timestamp<=self.config.sensor_timeout:return self._stop('stale_observation')
        if not np.allclose(packet.arm.position,self.config.arm_posture,atol=.05):return self._stop('arm_not_stowed')
        pose=tracking.T_map_base;xy=pose[:2,3];yaw=math.atan2(pose[1,0],pose[0,0])
        new=packet.timestamp!=self.last_sensor
        if new:
            self.last_sensor=packet.timestamp;self._depth(packet.cameras['navigation']);self.layers(xy)
            if self.previous_xy is not None:self.traveled+=float(np.linalg.norm(xy-self.previous_xy))
            self.previous_xy=xy.copy()
        if self.phase=='scan':
            decision=self.scanner.step(packet,tracking,now)
            if decision.complete:
                self.scans+=1;self.scan_locations.append(xy.copy());self.phase='travel';self.path=[]
                self.events.append(dict(time=now,event='scan_complete',position=xy.tolist()))
                return self._stop('scan_complete')
            if self.scanner.terminal:
                self.terminal=decision.status;return self._stop(self.terminal)
            return dict(status='scanning',forward=0.,yaw=decision.yaw_rps,scans=self.scans,plans=self.plans)
        if self.path and self._lookahead_reached(xy):
            self.path=[];self.phase='scan';self.scanner=LocalScanSkill(ScanConfig(stowed_arm=self.config.arm_posture))
            return self._stop('arrived_frontier')
        if not self.path:
            error=self._plan(xy,now)
            if error:self.terminal=error;return self._stop(error)
        while self.path and np.linalg.norm(self.path[0]-xy)<self.config.waypoint_tolerance:
            if len(self.path)>1 and not self._line_free(xy,self.path[1]):break
            if len(self.path)==1:
                clearance=distance_transform_edt(~self.obstacles)[self.cell(self.goal)]*self.grid.config.resolution
                tolerance=min(self.config.waypoint_tolerance,max(.03,clearance-self.config.center_clearance-.05))
                if np.linalg.norm(self.path[0]-xy)>=tolerance:break
            self.path.pop(0)
        if not self.path:
            self.phase='scan';self.scanner=LocalScanSkill(ScanConfig(stowed_arm=self.config.arm_posture));return self._stop('arrived_frontier')
        if np.linalg.norm(xy-self.last_progress)>.15:self.last_progress=xy.copy();self.progress_time=now
        if now-self.progress_time>15.:
            self.failed_goals.append(self.goal.copy());self.path=[];return self._stop('stalled_replan')
        target=self.path[0];angle=math.atan2(target[1]-xy[1],target[0]-xy[0])-yaw;angle=math.atan2(math.sin(angle),math.cos(angle))
        desired=np.array([self.config.speed*max(0.,math.cos(angle))**2 if abs(angle)<.5 else 0.,np.clip(1.8*angle,-self.config.yaw_speed,self.config.yaw_speed)])
        # Complete a checked turn before retrying translation. Otherwise each
        # control tick retries a tiny forward arc and accumulates lateral drift
        # until the current cell itself leaves the safe corridor.
        if self.turn_recovery_active:
            if abs(angle)<.03:self.turn_recovery_active=False
            else:desired[0]=0.
        desired[0]=min(desired[0],math.sqrt(2*self.config.acceleration*np.linalg.norm(target-xy)),self.config.approach_gain*np.linalg.norm(target-xy))
        limits=np.array([self.config.acceleration,self.config.yaw_acceleration])*dt
        command=self.velocity+np.clip(desired-self.velocity,-limits,limits)
        clear=self._command_clear(xy,yaw,command)
        if not clear and abs(self.velocity[0])<1e-6 and abs(angle)>.015:
            # At a tight turn, a curved translating motion can leave the safe
            # corridor even when rotating in place is clear. Check that separate
            # candidate with exactly the same map/depth guard; never clear cells.
            turn=np.array([0.,self.velocity[1]+np.clip(desired[1]-self.velocity[1],-limits[1],limits[1])])
            blocked=self.stop_detail
            if self._command_clear(xy,yaw,turn):
                self.velocity=turn
                self.turn_recovery_active=True
                if not self.events or self.events[-1].get('event')!='checked_turn_recovery':
                    self.events.append(dict(time=now,event='checked_turn_recovery',position=xy.tolist(),blocked=blocked))
                return dict(status='turn_recovery',forward=0.,yaw=float(turn[1]),scans=self.scans,plans=self.plans)
        if not clear:
            if now-self.last_plan>1.:
                self.path=[]
            result=self._stop('local_clearance_stop');result['detail']=self.stop_detail;return result
        self.velocity=command
        return dict(status='turn_recovery' if self.turn_recovery_active else 'travel',forward=float(command[0]),yaw=float(command[1]),scans=self.scans,plans=self.plans)

    def _command_clear(self,xy,yaw,command):
        # Sensor-derived map and live depth check the same stopping arc.
        p=xy.copy();heading=yaw;local=np.zeros(2);local_yaw=0.
        horizon=self.config.sensor_timeout+abs(command[0])/self.config.acceleration
        for t in np.arange(0.,horizon+.02,.02):
            v=np.sign(command[0])*max(0.,abs(command[0])-self.config.acceleration*max(0.,t-self.config.sensor_timeout))
            p+=v*.02*np.array([math.cos(heading),math.sin(heading)]);heading+=command[1]*.02
            local+=v*.02*np.array([math.cos(local_yaw),math.sin(local_yaw)]);local_yaw+=command[1]*.02
            cell=self.cell(p)
            if not(0<=cell[0]<self.free.shape[0] and 0<=cell[1]<self.free.shape[1] and self.free[cell]):
                self.stop_detail=dict(reason='map_braking_arc',cell=list(cell),position=p.tolist());return False
            if len(self.near_points) and np.any(np.linalg.norm(self.near_points[:,:2]-local,axis=1)<self.config.footprint_radius+.04):
                radial=np.linalg.norm(self.near_points[:,:2]-local,axis=1);self.stop_detail=dict(reason='depth_braking_arc',distance=float(radial.min()),point=self.near_points[int(radial.argmin())].tolist());return False
        return True
