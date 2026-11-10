"""Navigate into an observed room footprint; no object stand-off or grasp."""
import numpy as np
from .house_exploration import HouseExplorer,HouseExplorationConfig,reachable_paths


class RoomFollower(HouseExplorer):
    def __init__(self,grid,footprint,resolution,config=None):
        super().__init__(grid,config or HouseExplorationConfig())
        self.footprint=frozenset(footprint);self.room_resolution=resolution
        self.phase='travel'

    def _lookahead_reached(self,xy):
        return self.goal is not None and np.linalg.norm(xy-self.goal)<self.config.waypoint_tolerance

    def _plan(self,xy,now):
        distance,_=self.layers(xy);costs,parents=reachable_paths(self.free,self.cell(xy),distance)
        self.last_plan=now;self.plans+=1
        nodes=np.argwhere(np.isfinite(costs));points=self.world(nodes)
        cells=np.floor(points/self.room_resolution+1e-7).astype(int)
        eligible=[i for i,cell in enumerate(cells) if tuple(cell) in self.footprint]
        if not eligible:return 'no_observed_route_into_room'
        best=min(eligible,key=lambda i:costs[tuple(nodes[i])]);node=tuple(nodes[best]);self.goal=points[best]
        path=[node]
        while node in parents:node=parents[node];path.append(node)
        remaining=list(self.world(path[::-1]));simplified=[np.asarray(xy).copy()]
        while remaining:
            j=len(remaining)-1
            while j>=0 and not self._line_free(simplified[-1],remaining[j]):j-=1
            if j<0:return 'start_connector_blocked'
            simplified.append(remaining[j]);remaining=remaining[j+1:]
        self.path=simplified[1:];self.last_progress=np.asarray(xy).copy();self.progress_time=now
        self.events.append(dict(time=now,event='room_route',route=[p.tolist() for p in simplified]))
        return None


class RoomApproach:
    def __init__(self,grid,footprint,resolution):
        self.follower=RoomFollower(grid,footprint,resolution)

    def step(self,packet,tracking,now):
        result=self.follower.step(packet,tracking,now)
        if result['status']=='arrived_frontier':return self.follower._stop('portal_crossed')
        return result
