"""Shape-checked reuse of traveled SLAM routes on a frozen observed map.

The graph proposes a route; observed robot/payload swept geometry validates it.
It does not require the doorway to fit the larger circular exploration footprint.
No scene truth or replayed controls enter planning or following.
"""
import heapq
import time
import numpy as np
from scipy.spatial import cKDTree
from .shaped_docking import wrap
from .dock_follower import DockFollower


def se2(matrix):
    return np.r_[matrix[:2,3],np.arctan2(matrix[1,0],matrix[0,0])]


def drive_connection(start, end, checker):
    """Check the rotation/translation/rotation the forward/reverse base executes."""
    delta=end[:2]-start[:2]
    if np.linalg.norm(delta)<.005:
        return [end.copy()] if checker.swept(start,end) else None
    bearing=np.arctan2(delta[1],delta[0])
    headings=sorted((bearing,wrap(bearing+np.pi)),
                    key=lambda h:abs(wrap(h-start[2]))+abs(wrap(end[2]-h)))
    for heading in headings:
        path=[np.r_[start[:2],heading],np.r_[end[:2],heading],end.copy()]
        if all(checker.swept(a,b) for a,b in zip([start]+path,path)):
            return path
    return None


def plan_success_graph(memory, start, destination, anchor, checker, join_radius=1.5):
    """Lazy shortest-path search over measured poses with checked local joins.

    Thin stationary frames, retain turning poses, and check only candidate paths
    rather than collision-checking every pair of nearby scan frames. A rejected
    edge is removed from this search, not from the shared historical graph.
    """
    if memory is None or destination not in memory.stations or not memory.poses:
        return None,dict(reason='no_recorded_station_route')
    inverse=np.linalg.inv(anchor)
    all_poses=np.asarray([se2(inverse @ p) for p in memory.poses])
    station_nodes={s['node'] for stations in memory.stations.values() for s in stations}
    routes={}
    for a,b,r in memory.edges:routes.setdefault(r,[]).append((a,b))
    kept=[];links=[]
    for r,edges in routes.items():
        indices=[edges[0][0]]+[b for _,b in edges]
        chosen=[indices[0]]
        for i in indices[1:]:
            prev=all_poses[chosen[-1]];p=all_poses[i]
            if i in station_nodes or np.linalg.norm(p[:2]-prev[:2])>=.10 or abs(wrap(p[2]-prev[2]))>=.20:
                chosen.append(i)
        if chosen[-1]!=indices[-1]:chosen.append(indices[-1])
        offset=len(kept);kept.extend(chosen)
        links.extend((offset+i,offset+i+1,r) for i in range(len(chosen)-1))
    if not kept:return None,dict(reason='empty_recorded_route')
    poses=np.vstack([all_poses[kept],start]);source=len(kept)
    adjacency=[{} for _ in poses]
    def add(a,b,r):
        cost=float(np.linalg.norm(poses[a,:2]-poses[b,:2])+.12*abs(wrap(poses[b,2]-poses[a,2])))+1e-6
        adjacency[a][b]=(cost,r);adjacency[b][a]=(cost,r)
    for a,b,r in links:add(a,b,r)
    # Nearby historical visits are potential joins, not assumed free shortcuts.
    # The door operation can change which historical approach segment is free.
    # Allow local connections across visits/headings, then validate the actual
    # rotate/drive/rotate sweep lazily. A tiny same-heading join radius cannot
    # reconnect the outward retreat to the earlier kitchen path.
    for a,b in cKDTree(poses[:-1,:2]).query_pairs(join_radius):
        if b not in adjacency[a]:add(a,b,None)
    distances=np.linalg.norm(poses[:-1,:2]-start[:2],axis=1)
    entries=[]
    for i in np.argsort(distances):
        if distances[i]>join_radius:break
        if any(np.linalg.norm(poses[i,:2]-poses[j,:2])<.15 and abs(wrap(poses[i,2]-poses[j,2]))<.4 for j in entries):continue
        entries.append(i);add(source,int(i),None)
    goal_original={s['node'] for s in memory.stations[destination]}
    goals={i for i,j in enumerate(kept) if j in goal_original}
    if not entries or not goals:return None,dict(reason='no_local_graph_connection',entries=len(entries))
    # Floor underneath actual accepted poses is measured support, not invented
    # free cells; obstacle evidence is unchanged.
    for p in poses:checker.remember_support(p)
    cache={};rejected=0;searches=0;rejected_details=[]
    goal_xy=poses[list(goals),:2]
    def heuristic(i):return float(np.linalg.norm(goal_xy-poses[i,:2],axis=1).min())
    while True:
        searches+=1;queue=[(heuristic(source),0.,source)];costs={source:0.};parents={};end=None
        while queue:
            _,cost,node=heapq.heappop(queue)
            if cost>costs[node]+1e-9:continue
            if node in goals:end=node;break
            for nxt,(weight,r) in adjacency[node].items():
                value=cost+weight
                if value<costs.get(nxt,float('inf')):
                    costs[nxt]=value;parents[nxt]=(node,r)
                    heapq.heappush(queue,(value+heuristic(nxt),value,nxt))
        if end is None:
            return None,dict(reason='recorded_routes_blocked',rejected_edges=rejected,searches=searches,detail=checker.detail,rejected_details=rejected_details)
        chain=[end];route_ids=set()
        while chain[-1]!=source:
            node,r=parents[chain[-1]]
            if r is not None:route_ids.add(r)
            chain.append(node)
        chain.reverse();blocked=False;pieces=[]
        for a,b in zip(chain,chain[1:]):
            if (a,b) not in cache:cache[a,b]=drive_connection(poses[a],poses[b],checker)
            piece=cache[a,b]
            if piece is None:
                if len(rejected_details)<8:
                    rejected_details.append(dict(start=poses[a].tolist(),end=poses[b].tolist(),detail=checker.detail))
                adjacency[a].pop(b,None);adjacency[b].pop(a,None)
                rejected+=1;blocked=True;break
            pieces.extend(piece)
        if blocked:continue
        # Shortcut the selected path with the same swept connector. This omits
        # scan spins and short sampled steps, while retaining doorway bends.
        checkpoints=[poses[i] for i in chain[1:]]
        path=[];current=start;index=0
        while index<len(checkpoints):
            selected=None
            for last in range(len(checkpoints)-1,index-1,-1):
                candidate=drive_connection(current,checkpoints[last],checker)
                if candidate is not None:selected=(last,candidate);break
            if selected is None:
                return None,dict(reason='graph_shortcut_unavailable')
            last,piece=selected
            for p in piece:
                prev=path[-1] if path else start
                if np.linalg.norm(p[:2]-prev[:2])>.001 or abs(wrap(p[2]-prev[2]))>.005:path.append(p)
            current=checkpoints[last];index=last+1
        return path,dict(event='reuse_successful_route',strategy='shape_checked_success_graph',
                         destination=destination,route_ids=sorted(route_ids),graph_nodes=len(kept),
                         rejected_edges=rejected,searches=searches,waypoints=len(path),
                         cost_m=float(sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip([start]+path,path))))


class SuccessGraphTransit:
    """Plan once from the current observed posture; follow with SLAM at 50 Hz."""
    def __init__(self,memory,destination,anchor,checker,config):
        self.memory=memory;self.destination=destination;self.anchor=anchor
        self.checker=checker;self.config=config;self.events=[]
        self.follower=None;self.velocity=np.zeros(2);self.terminal=None

    def plan(self,pose):
        wall=time.monotonic()
        path,event=plan_success_graph(self.memory,pose,self.destination,self.anchor,self.checker)
        event['planning_wall_seconds']=time.monotonic()-wall;self.events.append(event)
        if path is None:return False
        self.follower=DockFollower(path,max_speed=self.config.speed,max_yaw_speed=self.config.yaw_speed,
                                   approach_gain=self.config.approach_gain,position_tolerance=.035,
                                   angle_tolerance=.04,recover_turn_drift=True,turn_drift_tolerance=.15)
        return True

    def step(self,packet,tracking,now):
        if self.follower is None:raise RuntimeError('Plan the observed route before execution')
        if not tracking.accepted:
            self.terminal='localization_lost';return dict(status=self.terminal,forward=0.,yaw=0.)
        desired=np.array(self.follower.command(se2(tracking.T_map_base)))
        limit=np.array([self.config.acceleration,self.config.yaw_acceleration])*.02
        self.velocity+=np.clip(desired-self.velocity,-limit,limit)
        if self.follower.complete and np.max(np.abs(self.velocity))<1e-8:
            self.terminal='known_destination_reached'
        return dict(status=self.terminal or 'follow_success_graph',forward=float(self.velocity[0]),yaw=float(self.velocity[1]),remaining_waypoints=len(self.follower.route))
