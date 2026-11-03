"""Leave a close manipulation dock using only observed shape clearance."""
import numpy as np
from .dock_follower import DockFollower
from .placement_docks import plan_to_any_dock

class ObservedUndock:
    def __init__(self,docking):
        self.docking=docking;self.follower=None;self.last_frame=None
        self.velocity=np.zeros(2);self.started=None;self.error=None;self.budget=30.

    def step(self,packet,tracking,free,now):
        stop=lambda status:dict(status=status,forward=0.,yaw=0.)
        if not tracking.accepted:return stop('tracking_lost')
        T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        if self.started is None:self.started=now
        if now-self.started>self.budget:self.error='undock_timeout';return stop(self.error)
        dock=self.docking
        if self.follower is None:
            cells=np.argwhere(free);xy=dock.grid.cell_to_world(cells[:,::-1])
            delta=xy-pose[:2];distance=np.linalg.norm(delta,axis=1)
            ids=np.flatnonzero((distance>.15)&(distance<1.5))
            ids=ids[np.argsort(distance[ids])]
            candidates=[]
            for i in ids:
                # Preserve the current orientation where possible; the A* path
                # may use checked forward/reverse segments to reach open floor.
                p=np.r_[xy[i],pose[2]]
                if any(np.linalg.norm(p[:2]-q[:2])<.1 for q in candidates):continue
                candidates.append(p)
                if len(candidates)>=30:break
            dock.checker=dock.make_checker(pose,packet.arm.position)
            direct=None
            # Grid centers can miss the exact current heading by half a cell.
            # Test continuous straight departure poses before adding turns.
            forward=np.array([np.cos(pose[2]),np.sin(pose[2])])
            aligned=[]
            for distance in np.arange(.15,1.51,.05):
                for sign in (-1.,1.):
                    candidate=np.r_[pose[:2]+sign*distance*forward,pose[2]]
                    col,row=dock.grid.world_to_cell(candidate[None,:2])[0]
                    if 0<=row<free.shape[0] and 0<=col<free.shape[1] and free[row,col]:
                        aligned.append(candidate)
            for candidate in aligned+candidates:
                delta=candidate[:2]-pose[:2]
                lateral=abs(delta@np.array([-np.sin(pose[2]),np.cos(pose[2])]))
                if lateral<.025 and dock.checker.swept(pose,candidate):
                    direct=candidate;break
            if direct is not None:
                dock.path=[direct];dock.goal=direct
            elif not candidates or not plan_to_any_dock(dock,pose,packet.arm.position,candidates):
                self.error='undock_route_unverified';return stop(self.error)
            original=list(dock.path)
            dock.path=departure_route(pose,original,dock.checker)
            dock.goal=dock.path[-1]
            route=[pose]+dock.path
            distance=sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip(route,route[1:]))
            turns=sum(abs((b[2]-a[2]+np.pi)%(2*np.pi)-np.pi) for a,b in zip(route,route[1:]))
            self.budget=max(15.,distance/.08+turns/.3+3.*len(dock.path))
            dock.events.append(dict(event='departure_motion_budget',seconds=self.budget,distance_m=distance,turn_radians=turns,removed_waypoints=len(original)-len(dock.path),reason='Observed route length, turns, and measured-settling allowance'))
            self.follower=DockFollower(dock.path)
            dock.events.append(dict(event='observed_undock_to_transit',goal=dock.goal.tolist(),reason='Close manipulation dock is outside conservative room-transit clearance'))
        if packet.timestamp!=self.last_frame:
            dock.checker=dock.make_checker(pose,packet.arm.position);self.last_frame=packet.timestamp
        try:desired=np.array(self.follower.command(pose))
        except RuntimeError as exc:self.error=str(exc);return stop('undock_tracking_error')
        if self.follower.complete:return stop('undock_complete')
        limit=np.array([.4,.8])*.02
        command=self.velocity+np.clip(desired-self.velocity,-limit,limit)
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


def departure_route(start, route, checker):
    """No destination-facing requirement for leaving a dock into open floor."""
    path=[np.asarray(p).copy() for p in route]
    if len(path)>1 and np.linalg.norm(path[-1][:2]-path[-2][:2])<1e-6:
        # The preceding arrival pose was already swept-checked. Restoring the
        # departure heading is redundant; room navigation chooses its heading.
        path.pop()
    result=[np.asarray(start)]
    for point in path:
        if len(result)>1:
            a,b=result[-2:]
            wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
            ab=b[:2]-a[:2];bc=point[:2]-b[:2]
            collinear=abs(ab[0]*bc[1]-ab[1]*bc[0])<1e-7 and np.dot(ab,bc)>=0
            if collinear and abs(wrap(a[2]-b[2]))<1e-6 and abs(wrap(b[2]-point[2]))<1e-6 and checker.swept(a,point):
                result.pop()
        result.append(point)
    return result[1:]
