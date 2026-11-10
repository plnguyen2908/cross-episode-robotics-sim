"""Explore a 2D map using two cached visual work-station landmarks."""
import numpy as np
from .semantic_navigation import SemanticNavigator
from ..perception.visual_targets import VisualTarget


class StationExplorer(SemanticNavigator):
    """No room classifier or per-frame vessel detector.

    Arrival certifies a route to the remembered table stand-off, not a fresh
    object observation or manipulation-ready pose. Local search follows later.
    """
    def __init__(self, grid, furniture, explorer=None, config=None, preferred_distance=1.1):
        kwargs={} if config is None else {'config':config}
        super().__init__(grid,None,'dining table',explorer,preferred_distance=preferred_distance,**kwargs)
        self.furniture=furniture
        self._reported_landmarks=None

    def observe(self,packet,tracking):
        self.furniture.observe(packet,tracking)
        summary=tuple((x['label'],x['observations']) for x in self.furniture.landmarks)
        if summary!=self._reported_landmarks:
            import json
            print(json.dumps(dict(event='station_landmarks',landmarks=list(summary),required=['kitchen','dining table'])),flush=True)
            self._reported_landmarks=summary
        confirmed={x['label']:x for x in self.furniture.landmarks if x['observations']>=2}
        if not {'kitchen','dining table'} <= confirmed.keys():return
        # Keep object identity stable while the follower uses this target.
        if not self.targets.targets:
            table=confirmed['dining table']
            self.targets.targets=[VisualTarget(
                track_id=0,label='dining table',position=np.asarray(table['position']),
                score=1.,observations=table['observations'],last_seen=table['last_seen'],
                history=[dict(timestamp=x['timestamp'],position=table['position']) for x in table['evidence']],
                navigation_memory=True)]

    def step(self,packet,tracking,now):
        command=super().step(packet,tracking,now)
        if command['status']=='at_target_standoff':
            self.terminal='stations_explored'
            return self.follower._stop(self.terminal)
        return command
