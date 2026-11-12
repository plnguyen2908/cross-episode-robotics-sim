"""Alternative facing docks around observed interior placement points."""
from dataclasses import dataclass
import numpy as np


@dataclass
class PlacementDock:
    pose: np.ndarray
    target: np.ndarray
    distance: float


def placement_docks(targets, current_pose, distances=(.58,.66,.50), angle_count=24,
                    target_spacing=.15, preferred_targets=None):
    """Sample all sides; face the chosen observed interior target at every dock.

    This proposes poses only. Observed swept collision/support checks, route
    planning and arm reach validation are mandatory before execution.
    """
    targets=np.asarray(targets,float)
    if targets.ndim!=2 or targets.shape[1]!=3 or not np.isfinite(targets).all():
        raise ValueError('Expected finite observed placement points')
    _,indices=np.unique(np.floor(targets/target_spacing).astype(int),axis=0,return_index=True)
    targets=targets[np.sort(indices)]
    preferred=np.asarray(preferred_targets if preferred_targets is not None else [],float).reshape(-1,3)
    result=[]
    for target in targets:
        for distance in distances:
            if distance<=0:raise ValueError('Dock distance must be positive')
            for facing in np.linspace(-np.pi,np.pi,angle_count,endpoint=False):
                xy=target[:2]-distance*np.array([np.cos(facing),np.sin(facing)])
                result.append(PlacementDock(np.r_[xy,facing],target.copy(),distance))
    def cost(candidate):
        turn=(candidate.pose[2]-current_pose[2]+np.pi)%(2*np.pi)-np.pi
        travel=np.linalg.norm(candidate.pose[:2]-current_pose[:2])+.12*abs(turn)
        # Try docks facing the perceived empty patch first, then other free
        # patches if the preferred one has no reachable collision-free dock.
        return (float(np.linalg.norm(preferred-candidate.target,axis=1).min()),travel) if len(preferred) else (0.,travel)
    return sorted(result,key=cost)


def retained_arm_clear(boxes, pose, support, base_height=0., margin=.05):
    """Is the carried arm posture clear of the observed support at this dock?

    ``boxes`` are calibrated robot boxes in the base frame at the retained
    joints; ``support`` holds measured support points in the dock's map frame.
    Arm planning starts from this posture, so a link resting within the margin
    of the support leaves no valid first motion. The base footprint is checked
    separately by the route layer.
    """
    support=np.asarray(support,float).reshape(-1,3)
    if not len(support):return True
    c,s=np.cos(pose[2]),np.sin(pose[2])
    local=support-np.r_[np.asarray(pose[:2],float),base_height]
    local=np.c_[c*local[:,0]+s*local[:,1],-s*local[:,0]+c*local[:,1],local[:,2]]
    for box in boxes:
        if box.name.startswith('robot_0/base/'):continue
        if np.all(np.abs((local-box.center)@box.axes)<=box.half+margin,axis=1).any():return False
    return True


def plan_to_any_dock(dock, pose, q, candidates, max_nodes=4000):
    """One shared A* search to any observed, facing dock, not one search per pose."""
    import heapq
    from .shaped_docking import rotation
    dock.checker=dock.make_checker(pose,q);dock.initial=np.asarray(pose).copy();dock.path=None
    goals=[np.asarray(g) for g in candidates if dock.checker.clear(g)]
    if not goals or not dock.checker.clear(pose):
        dock.events.append(dict(event='placement_route_unverified',detail=dock.checker.detail))
        return False
    R=rotation(pose[2]);xy=np.array([g[:2] for g in goals])
    def world(s):return np.r_[pose[:2]+R@(np.array(s[:2])*.1),pose[2]+s[2]*np.pi/4]
    def heuristic(s):return max(0.,float(np.linalg.norm(xy-world(s)[:2],axis=1).min())-.15)
    start=(0,0,0);queue=[(heuristic(start),0.,start)];cost={start:0.};parents={};cache={};visited=0
    while queue and visited<max_nodes:
        _,distance,s=heapq.heappop(queue)
        if distance>cost[s]+1e-9:continue
        visited+=1;p=world(s)
        nearby=np.flatnonzero(np.linalg.norm(xy-p[:2],axis=1)<.15)
        for index in nearby:
            goal=goals[index];delta=goal[:2]-p[:2]
            heading=np.arctan2(delta[1],delta[0]) if np.linalg.norm(delta)>.005 else p[2]
            connector=[np.r_[p[:2],heading],np.r_[goal[:2],heading],goal]
            if all(dock.checker.swept(a,b) for a,b in zip([p]+connector,connector)):
                states=[s]
                while s in parents:s=parents[s];states.append(s)
                dock.path=[world(k) for k in states[::-1]][1:]+connector;dock.goal=goal
                dock.events.append(dict(event='placement_multigoal_astar',visited=visited,goals=len(goals),goal=goal.tolist(),route=[v.tolist() for v in dock.path]))
                return True
        k=s[2];dx,dy=int(round(np.cos(k*np.pi/4))),int(round(np.sin(k*np.pi/4)))
        neighbors=[((s[0],s[1],(k+1)%8),.12),((s[0],s[1],(k-1)%8),.12),
                   ((s[0]+dx,s[1]+dy,k),np.hypot(dx,dy)*.1),
                   ((s[0]-dx,s[1]-dy,k),np.hypot(dx,dy)*.1+.04)]
        for nxt,weight in neighbors:
            nd=distance+weight
            if max(abs(nxt[0]),abs(nxt[1]))>25 or nd>=cost.get(nxt,float('inf')):continue
            key=(s,nxt)
            if key not in cache:cache[key]=dock.checker.swept(p,world(nxt))
            if not cache[key]:continue
            cost[nxt]=nd;parents[nxt]=s
            heapq.heappush(queue,(nd+heuristic(nxt),nd,nxt))
    dock.events.append(dict(event='placement_route_unverified',visited=visited,goals=len(goals),detail=dock.checker.detail))
    return False
