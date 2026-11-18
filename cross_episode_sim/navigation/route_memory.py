"""Episode-local successful routes, expressed in the continuous SLAM frame."""
import heapq
import numpy as np


def transform(points, matrix):
    points = np.asarray(points, float)
    return points @ matrix[:2, :2].T + matrix[:2, 3]


# A station entry recorded where its furniture was seen from, never reached.
SIGHTING = 'exploration_sighting'
# A station entry recorded where exploration stood at its furniture.
STANDOFF = 'exploration_standoff'


class RouteMemory:
    def __init__(self):
        self.routes = []
        self.nodes = []
        self.edges = []
        self.stations = {}
        self.poses = []

    def record(self, poses, anchor, start, end, start_kind='successful_transit_endpoint'):
        points = np.asarray([p[:2, 3] for p in poses])
        if len(points) < 2 or not start or not end:
            return
        # Retain actual traveled samples, not proposed paths or failed attempts.
        world = transform(points, anchor)
        self.routes.append((start, end, world))
        offset = len(self.nodes)
        self.nodes.extend(world.copy())
        self.poses.extend([anchor @ pose for pose in poses])
        route_id = len(self.routes)-1
        self.edges.extend((offset+i,offset+i+1,route_id) for i in range(len(world)-1))
        for label,index,pose,kind in ((start,offset,poses[0],start_kind),
                                      (end,len(self.nodes)-1,poses[-1],'successful_transit_endpoint')):
            global_pose = anchor @ pose
            self.stations.setdefault(label,[]).append(dict(node=index,
                pose=global_pose.copy(),kind=kind))

    def record_dock(self, poses, anchor, station):
        """Remember a successful local approach, without certifying future grasps."""
        if len(poses)<2:return
        self.record(poses,anchor,station,station)
        self.stations[station][-1]['kind']='successful_dock'

    def plan(self, xy, destination, anchor, segment_clear, connector, join_radius=1.5):
        """Shortest checked graph route, including the local A* connector cost.

        Nearby route samples are connected only after an explicit clearance
        check; proximity alone never creates an edge through a wall.
        """
        if not self.nodes or destination not in self.stations:
            return None
        from scipy.spatial import cKDTree
        points=transform(self.nodes,np.linalg.inv(anchor))
        adjacency=[[] for _ in points]
        def add(a,b,route):
            if segment_clear(points[a],points[b]):
                cost=float(np.linalg.norm(points[a]-points[b]))
                adjacency[a].append((b,cost,route));adjacency[b].append((a,cost,route))
        for a,b,route in self.edges:add(a,b,route)
        # Join recorded paths at nearby crossings, without mutating the frozen map.
        for a,b in cKDTree(points).query_pairs(.20):
            add(a,b,None)
        costs={};parents={};queue=[];connectors={}
        # Spatially distinct entry options avoid spending A* calls on duplicate frames.
        entries=[]
        for index in np.argsort(np.linalg.norm(points-xy,axis=1)):
            if np.linalg.norm(points[index]-xy)>join_radius:break
            if any(np.linalg.norm(points[index]-points[j])<.15 for j in entries):continue
            entries.append(index)
            path=connector(xy,points[index])
            if path is None or not len(path):continue
            path=np.asarray(path)
            cost=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
            costs[index]=cost;connectors[index]=path
            heapq.heappush(queue,(cost,index))
        goals={s['node'] for s in self.stations[destination]}
        end=None
        while queue:
            cost,node=heapq.heappop(queue)
            if cost!=costs.get(node):continue
            if node in goals:end=node;break
            for nxt,distance,route in adjacency[node]:
                new=cost+distance
                if new<costs.get(nxt,float('inf')):
                    costs[nxt]=new;parents[nxt]=(node,route)
                    heapq.heappush(queue,(new,nxt))
        if end is None:return None
        chain=[end];routes=set();reverse=False
        while chain[-1] in parents:
            previous,route=parents[chain[-1]]
            if route is not None:
                routes.add(route);reverse |= previous>chain[-1]
            chain.append(previous)
        chain.reverse()
        path=np.vstack((connectors[chain[0]],points[chain[1:]]))
        return dict(path=path,route_ids=sorted(routes),reverse=bool(reverse),
                    cost_m=costs[end],station_node=int(end),graph_nodes=len(points))

    def candidates(self, destination, anchor):
        inverse = np.linalg.inv(anchor)
        for start, end, points in reversed(self.routes):
            if end == destination:
                yield transform(points, inverse), False
            elif start == destination:
                yield transform(points[::-1], inverse), True


def episode_routes():
    from cross_episode_sim.runtime.episode import current_runtime
    runtime = current_runtime()
    if runtime is None:
        return None
    if not hasattr(runtime, 'route_memory'):
        runtime.route_memory = RouteMemory()
    return runtime.route_memory


def record_transit(slam, start, end, success):
    memory = episode_routes()
    if success and memory is not None and hasattr(slam, 'anchor'):
        memory.record(slam.trajectory, slam.anchor, start, end)


def record_exploration(slam, landmarks, success, *, destination_ready=True):
    """Seed transit memory from a completed sensor-confirmed station exploration."""
    memory=episode_routes()
    if not success or memory is None or not hasattr(slam,'anchor'):
        return dict(recorded=False,reason='Exploration unsuccessful or no persistent session')
    confirmed={item['label']:item for item in landmarks if item.get('observations',0)>=2}
    session=getattr(slam,'session',None)
    # A station-label handoff must not discard the earlier room crossing.
    trajectory=slam.trajectory if session is None else [np.linalg.inv(slam.anchor) @ p for p in session.trajectory]
    if not {'kitchen','dining table'} <= confirmed.keys() or len(trajectory)<2:
        return dict(recorded=False,reason='Missing confirmed stations or tracked path')
    views=np.asarray(confirmed['kitchen'].get('viewpoints',[]))
    if not len(views):return dict(recorded=False,reason='No observed kitchen viewpoint')
    poses=np.asarray(trajectory)
    distances=np.linalg.norm(poses[:,None,:2,3]-views[None,:,:2],axis=-1).min(axis=1)
    candidates=np.flatnonzero(distances<=.15)
    if not len(candidates):return dict(recorded=False,reason='No tracked path at observed kitchen viewpoint')
    first=int(candidates[0])
    if len(poses[first:])<2:return dict(recorded=False,reason='No station-to-station segment')
    destination='dining table' if destination_ready else 'exploration endpoint'
    # Exploration replaces the recognition viewpoints with its counter stand-off
    # once it has stood there. Without that visit the kitchen end is only where
    # the counter was recognised from. Say so; a loaded carry continues from there.
    kind=STANDOFF if confirmed['kitchen'].get('reached') else SIGHTING
    memory.record(poses[first:],slam.anchor,'kitchen',destination,start_kind=kind)
    # A recognition viewpoint can be across the room, or even through a door.
    # Keep the traveled graph, but do not alias a remote sighting as a reached
    # table destination. The existing fixed-map target navigator uses the
    # table's observed geometry until a successful transit/dock is recorded.
    return dict(recorded=True,source='successful_station_exploration',
                start='kitchen',start_kind=kind,end=destination,samples=len(poses)-first,
                continuous_session=session is not None)
