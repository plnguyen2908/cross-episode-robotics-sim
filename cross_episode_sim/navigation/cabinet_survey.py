"""Look at the whole kitchen before choosing which storage cabinet to approach.

Navigation used to drive to the nearest confirmed 'handle on kitchen storage
cabinet'. From the kitchen station that is often a small compartment under the
stove top (fix57, fix59, fix60); from wide views showing several lower cabinets
Molmo2 picked a large cabinet 25/25 times. So: turn once in place at the start,
let the existing pointing supervisor point at every heading, then prefer the
handle it pointed at most often (wall-height handles excluded).

Onboard RGB-D, SLAM and the pointing supervisor only.
"""
import math
import numpy as np

MAX_HANDLE_HEIGHT=1.35
# The kitchen station faces the counter run. Facing away, Molmo points at the
# room door's handle (~1 m high) and those votes beat the big cabinet (fix59
# emulation). Count only handles within this bearing of the starting heading.
MAX_BEARING=math.radians(100)
# Asked for one "handle on kitchen storage cabinet", Molmo2 returns the left
# handle of stack_2's double door and almost never the right one (1/25
# emulated surveys). Asked for both handles it marks both (left 32/32, right
# 23/32 views), so the survey asks this and keeps the two handles apart.
SURVEY_PROMPT='both door handles of the large double-door lower kitchen cabinet'
VIEW_STEP=math.radians(30)
HANDLE_MERGE=.05      # the two handles of a double door are ~8 cm apart
CABINET_GROUP=.25


def pointed_handles(camera, points, T_map_base):
    """Map-frame positions of pointed pixels from measured, non-robot depth."""
    out=[]
    valid=camera.valid_depth.copy()
    if camera.robot_mask is not None:valid&=~camera.robot_mask
    for x,y in points:
        u,v=int(round(x)),int(round(y))
        if not (0<=v<valid.shape[0] and 0<=u<valid.shape[1] and valid[v,u]):continue
        z=float(camera.depth_m[v,u])
        if not .1<z<5.:continue
        q=np.array([(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z])
        b=camera.T_base_optical[:3,:3]@q+camera.T_base_optical[:3,3]
        out.append(np.asarray(T_map_base)[:3,:3]@b+np.asarray(T_map_base)[:3,3])
    return out


def choose_handle(handles, centre):
    """Most-pointed cabinet (handles within 25 cm), then its handle nearest the room centre."""
    if not handles:return None
    votes=lambda h:sum(o['votes'] for o in handles if np.linalg.norm(o['p'][:2]-h['p'][:2])<=CABINET_GROUP)
    top=max(handles,key=lambda h:(votes(h),h['votes']))
    cabinet=[h for h in handles if np.linalg.norm(h['p'][:2]-top['p'][:2])<=CABINET_GROUP]
    if centre is None:return max(cabinet,key=lambda h:h['votes'])
    return min(cabinet,key=lambda h:(float(np.linalg.norm(h['p'][:2]-np.asarray(centre))),-h['votes']))


def open_side_floor(grid, handle_xy, station_xy):
    """Signed preference for a handle toward the open side of its cabinet.

    Free observed floor beside the cabinet front (0.3-1.2 m sideways, 0.2-1.0 m
    out) on each side of the handle; returns (open side's free floor) projected
    on the handle's sideways offset from the cabinet centre is not known, so the
    handle with more free floor on its emptier side wins (it is further from the
    wall end of the counter run).
    """
    h=np.asarray(handle_xy,float);f=np.asarray(station_xy,float)-h
    if np.linalg.norm(f)<1e-6:return 0
    f/=np.linalg.norm(f);l=np.array([-f[1],f[0]])
    rr,cc=np.indices(grid.floor_views.shape)
    xy=grid.cell_to_world(np.stack([cc,rr],axis=-1))-h
    out=xy@f;side=xy@l
    free=(grid.floor_views>=grid.config.minimum_floor_views)&~(grid.evidence>=2).any(axis=2)
    band=(out>=.2)&(out<=1.)
    left=int((free&band&(side>=.3)&(side<=1.2)).sum());right=int((free&band&(side<=-.3)&(side>=-1.2)).sum())
    return min(left,right)


def room_centre(grid, station_xy, radius=3.5):
    """Centre of the observed free floor within `radius` of the kitchen station."""
    rr,cc=np.indices(grid.floor_views.shape)
    xy=grid.cell_to_world(np.stack([cc,rr],axis=-1))
    free=(grid.floor_views>=grid.config.minimum_floor_views)&~(grid.evidence>=2).any(axis=2)
    free&=np.linalg.norm(xy-np.asarray(station_xy,float),axis=-1)<=radius
    return xy[free].mean(axis=0) if free.any() else None


class CabinetSurvey:
    def __init__(self, label, yaw_speed=.35, yaw_acceleration=.8, angle=2*math.pi, max_seconds=30.):
        self.label=label;self.yaw_speed=yaw_speed;self.yaw_acceleration=yaw_acceleration
        self.angle=angle;self.max_seconds=max_seconds
        self.turned=0.;self.velocity=0.;self.last=None;self.start=None;self.done=False
        self.chosen=None;self.summary=None;self.origin=None
        self.handles=[];self.next_view=0.;self.views=[]

    @staticmethod
    def applies(target):
        return target.startswith('handle on ') and 'cabinet' in target and 'microwave' not in target

    def step(self, command, packet, tracking, now, pointing, navigator):
        if self.done:return command
        now=float(now)
        if self.start is None:self.start=now
        dt=.02 if self.last is None else max(0.,now-self.last);self.last=now
        if tracking.accepted and self.origin is None:
            T=np.asarray(tracking.T_map_base);self.origin=(T[:2,3].copy(),math.atan2(T[1,0],T[0,0]))
        if tracking.accepted and pointing is not None and self.turned>=self.next_view and self.turned<self.angle-1e-3:
            self.next_view+=VIEW_STEP;self._look(packet,tracking,pointing)
        if self.turned>=self.angle or now-self.start>=self.max_seconds:
            return self._finish(command,pointing,navigator)
        remaining=self.angle-self.turned
        target_speed=min(self.yaw_speed,math.sqrt(2*self.yaw_acceleration*max(remaining,0.)))
        step=self.yaw_acceleration*dt
        self.velocity=min(target_speed,self.velocity+step) if self.velocity<target_speed else max(target_speed,self.velocity-step)
        self.turned+=self.velocity*dt
        return dict(command,status='cabinet_survey',forward=0.,yaw=self.velocity)

    def _look(self, packet, tracking, pointing):
        from PIL import Image
        camera=packet.cameras['navigation'];pointer=pointing.pointer
        result=pointer.remote.call(image=np.asarray(camera.rgb),target=SURVEY_PROMPT) if getattr(pointer,'remote',None) is not None else pointer.point(Image.fromarray(camera.rgb),SURVEY_PROMPT)
        found=pointed_handles(camera,result['points'][:4],tracking.T_map_base)
        for p in found:
            for h in self.handles:
                if np.linalg.norm(h['p']-p)<HANDLE_MERGE:h['p']=(h['p']*h['votes']+p)/(h['votes']+1);h['votes']+=1;break
            else:self.handles.append(dict(p=np.asarray(p,float),votes=1))
        self.views.append(dict(turned=round(self.turned,2),points=len(result['points']),placed=len(found)))

    def _finish(self, command, pointing, navigator):
        self.done=True;self.velocity=0.
        def ahead(p):
            if self.origin is None:return True
            xy,heading=self.origin;d=np.asarray(p[:2])-xy
            return abs((math.atan2(d[1],d[0])-heading+math.pi)%(2*math.pi)-math.pi)<=MAX_BEARING
        low=[h for h in self.handles if float(h['p'][2])<=MAX_HANDLE_HEIGHT and ahead(h['p'])]
        grid=getattr(getattr(navigator,'semantic',None),'grid',None)
        centre=room_centre(grid,self.origin[0]) if (grid is not None and self.origin is not None) else None
        best=choose_handle(low,centre)
        if best is not None:
            from ..perception.visual_targets import VisualTarget
            targets=getattr(pointing,'targets',None)
            existing=[t for t in getattr(targets,'targets',[]) if t.label==self.label and np.linalg.norm(t.position-best['p'])<HANDLE_MERGE]
            if existing:
                self.chosen=existing[0];self.chosen.observations=max(self.chosen.observations,best['votes'])
            else:
                ids=[t.track_id for t in getattr(targets,'targets',[])]
                self.chosen=VisualTarget(track_id=max(ids,default=0)+1,label=self.label,position=np.asarray(best['p'],float),score=1.,observations=best['votes'],last_seen=float(self.last or 0.))
                if targets is not None:targets.targets.append(self.chosen)
            navigator.preferred_target=self.chosen
            # Hand the surveyed handle straight to the fixture skill, whose
            # redock approaches it (fix58 redocked from 1.9 m). Driving closer
            # here let the local approach and its nearest-handle confirmation
            # switch back to the small compartment (fix61: survey chose the big
            # cabinet, navigation ended at stack_3).
            navigator.found_target=self.chosen;navigator.object_search_success=True
            navigator.terminal='fixture_handle_found'
            if hasattr(navigator,'events'):navigator.events.append(dict(event='cabinet_survey_handoff',position=np.asarray(self.chosen.position,float).tolist(),observations=self.chosen.observations))
        self.summary=dict(turned_rad=round(self.turned,2),prompt=SURVEY_PROMPT,views=self.views,
            room_centre=None if centre is None else np.round(centre,3).tolist(),
            handles=[dict(position=np.round(h['p'],3).tolist(),votes=h['votes'],eligible=any(h is o for o in low)) for h in self.handles],
            chosen=None if self.chosen is None else np.round(self.chosen.position,3).tolist())
        print('CABINET_SURVEY',__import__('json').dumps(self.summary),flush=True)
        return dict(command,status='cabinet_survey_done',forward=0.,yaw=0.)
