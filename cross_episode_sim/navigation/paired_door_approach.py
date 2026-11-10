"""Observed adjacent-leaf selection and shape-checked local handle approach."""
import numpy as np
from .portal_approach import PortalApproach
from .observed_undock import ObservedUndock
from .shaped_docking import ShapedDocking


def adjacent_closed_leaf(tracks, opened, normal):
    n=np.asarray(normal,dtype=float)[:2];n=n/np.linalg.norm(n)
    tangent=np.array([-n[1],n[0]]);origin=np.asarray(opened['position'])
    candidates=[]
    for t in tracks:
        if t['portal']==opened['portal'] or t.get('duplicate_of') or not t.get('closed',True):continue
        if t.get('visual_category','closed_room_door')!='closed_room_door':continue
        if t.get('source')!=opened.get('source'):continue
        delta=np.asarray(t['position'])-origin
        # Nearby, same-plane room-door handles, not scene fixture identities.
        if abs(delta[2])<.15 and abs(delta[:2]@n)<.12 and .10<abs(delta[:2]@tangent)<.8:
            candidates.append(t)
    return min(candidates,key=lambda t:np.linalg.norm(np.asarray(t['position'])-origin),default=None)


class LocalDoorApproach(PortalApproach):
    def __init__(self,grid,target,normal,geometry,*,runtime_collision_stop=False):
        super().__init__(grid,target,geometry=geometry)
        self.normal=np.asarray(normal,dtype=float)[:2];self.normal/=np.linalg.norm(self.normal)
        self.local_motion=None;self.local_done=False
        self.runtime_collision_stop=runtime_collision_stop

    @property
    def freeze_navigation_map(self):
        # The door update is integrated before constructing this local move.
        # Follow its checked map; resume fusion when opening/crossing continues.
        return not self.local_done

    def step(self,packet,tracking,now):
        if not tracking.accepted:return self.follower._stop('tracking_lost')
        if self.local_done:return super().step(packet,tracking,now)
        if self.local_motion is None:
            target=self.follower.target.position[:2];pose=tracking.T_map_base
            if (pose[:2,3]-target)@self.normal<0:self.normal*=-1
            g=self.follower.grid;yy,xx=np.indices(g.floor_views.shape)
            xy=g.cell_to_world(np.stack((xx,yy),axis=-1));delta=xy-target
            side=delta@self.normal;lateral=delta@np.array([-self.normal[1],self.normal[0]])
            self.goals=(side>.5)&(side<.85)&(abs(lateral)<.4)
            self.goals &= (g.floor_views>=g.config.minimum_floor_views)&~(g.evidence>=2).any(axis=2)
            dock=ShapedDocking(g,self.geometry)
            self.local_motion=ObservedUndock(dock,max_distance=4.,goal_heading=np.arctan2(-self.normal[1],-self.normal[0]),goal_pose_required=True,runtime_collision_stop=self.runtime_collision_stop)
            self.follower.events=dock.events
        result=self.local_motion.step(packet,tracking,self.goals,now)
        self.follower.path=self.local_motion.docking.path or []
        if self.local_motion.error:
            self.follower.terminal=result['status']
            self.follower.stop_detail=self.local_motion.docking.checker.detail
        if result['status']=='undock_complete':
            self.local_done=True;self.align=True
            return self.follower._stop('local_door_approach_complete')
        return result


def left_observed_leaf(tracks, reference, base_xy):
    return observed_side_leaf(tracks,reference,base_xy,'left')


def observed_side_leaf(tracks, reference, base_xy, side):
    if side not in ('left','right'):raise ValueError('Expected observed left or right')
    center=np.asarray(reference['position'])[:2]
    forward=center-np.asarray(base_xy);forward=forward/np.linalg.norm(forward)
    left=np.array([-forward[1],forward[0]])
    candidates=[t for t in tracks if t.get('closed',True) and not t.get('duplicate_of')
                and t.get('visual_category','closed_room_door')=='closed_room_door'
                and t.get('source')==reference.get('source') and t.get('observations',0)>=2
                and np.linalg.norm(np.asarray(t['position'])[:2]-center)<.8]
    choose=max if side=='left' else min
    return choose(candidates,key=lambda t:(np.asarray(t['position'])[:2]-center)@left,default=reference)
