"""Explore a 2D map using two cached visual work-station landmarks."""
import json
import numpy as np
from .semantic_navigation import SemanticNavigator
from ..perception.visual_targets import VisualTarget

# A completed scan this close already covers the floor around the stand-off.
SCAN_REUSE_RADIUS=.5


class StationExplorer(SemanticNavigator):
    """No room classifier or per-frame vessel detector.

    The robot stands at the kitchen counter before it finishes at the dining
    table. Recognising the counter from another room confirms the landmark,
    but later docking needs the floor beside it in the map, so the counter
    stand-off is visited and scanned once.

    Arrival certifies a route to the remembered table stand-off, not a fresh
    object observation or manipulation-ready pose. Local search follows later.
    """
    def __init__(self, grid, furniture, explorer=None, config=None, preferred_distance=1.1):
        kwargs={} if config is None else {'config':config}
        super().__init__(grid,None,'dining table',explorer,preferred_distance=preferred_distance,**kwargs)
        self.furniture=furniture
        self._reported_landmarks=None
        # None until the counter is confirmed, then 'kitchen', 'kitchen_scan', 'dining table'.
        self.stage=None

    def _head_for(self,landmark,track_id):
        self.target_label=landmark['label']
        self.targets.targets=[VisualTarget(
            track_id=track_id,label=landmark['label'],position=np.asarray(landmark['position']),
            score=1.,observations=landmark['observations'],last_seen=landmark['last_seen'],
            history=[dict(timestamp=x['timestamp'],position=landmark['position']) for x in landmark['evidence']],
            navigation_memory=True)]

    def observe(self,packet,tracking):
        self.furniture.observe(packet,tracking)
        summary=tuple((x['label'],x['observations']) for x in self.furniture.landmarks)
        if summary!=self._reported_landmarks:
            print(json.dumps(dict(event='station_landmarks',landmarks=list(summary),required=['kitchen','dining table'])),flush=True)
            self._reported_landmarks=summary
        confirmed={x['label']:x for x in self.furniture.landmarks if x['observations']>=2}
        if self.stage is None and 'kitchen' in confirmed:
            # A counter stand-off reached earlier in this map need not be repeated.
            self.stage='dining table' if confirmed['kitchen'].get('reached') else 'kitchen'
            if self.stage=='kitchen':self._head_for(confirmed['kitchen'],1)
        # Keep object identity stable while the follower uses this target.
        if self.stage=='dining table' and not self.targets.targets and 'dining table' in confirmed:
            self._head_for(confirmed['dining table'],0)

    def _event(self,now,event,**fields):
        self.follower.events.append(dict(time=now,event=event,**fields))
        print(json.dumps(dict(event=event,**fields)),flush=True)

    def _at_kitchen(self,tracking,now):
        pose=tracking.T_map_base;xy=pose[:2,3].copy()
        landmark=next(x for x in self.furniture.landmarks if x['label']=='kitchen' and x['observations']>=2)
        # From here on the station is this reached pose, not where the counter was first seen from.
        landmark['viewpoints']=[pose[:3,3].tolist()];landmark['reached']=True
        self.targets.targets=[];self.mode='explore';self.alignment_started=None
        self.follower.target=None;self.follower.path=[];self.follower.goal=None;self.follower.terminal=None
        scanned=any(np.linalg.norm(xy-p)<SCAN_REUSE_RADIUS for p in self.explorer.scan_locations)
        self._event(now,'kitchen_standoff_reached',position=xy.tolist(),scan=not scanned)
        if scanned:self._leave_kitchen(now)
        else:
            self.stage='kitchen_scan';self.explorer.phase='scan';self.explorer.resume(now)
        return self.follower._stop('kitchen_standoff_reached')

    def _leave_kitchen(self,now):
        self.stage='dining table';self.explorer.phase='travel';self.explorer.resume(now)

    def _scan_kitchen(self,packet,tracking,now):
        if now-self.started>=self.follower.config.max_seconds:
            self.terminal='budget_exhausted';return self.follower._stop(self.terminal)
        self.observe(packet,tracking)
        command=self.explorer.step(packet,tracking,now)
        if command['status']=='scan_complete':
            self._event(now,'kitchen_scan_complete',position=tracking.T_map_base[:2,3].tolist())
            self._leave_kitchen(now)
        elif self.explorer.terminal=='budget_exhausted':
            self.terminal=self.explorer.terminal
        elif self.explorer.terminal:
            # No room to turn here. The approach still observed this floor;
            # an unfinished scan must not end the exploration.
            self._event(now,'kitchen_scan_abandoned',reason=self.explorer.terminal)
            self.explorer.terminal=None;self._leave_kitchen(now)
            return self.follower._stop('kitchen_scan_abandoned')
        return command

    def step(self,packet,tracking,now):
        if self.stage=='kitchen_scan' and not self.terminal:
            return self._scan_kitchen(packet,tracking,now)
        command=super().step(packet,tracking,now)
        if command['status']=='at_target_standoff':
            if self.stage=='kitchen':return self._at_kitchen(tracking,now)
            self.terminal='stations_explored'
            return self.follower._stop(self.terminal)
        return command
