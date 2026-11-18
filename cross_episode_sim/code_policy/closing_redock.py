"""Dock at the exposed handle face of an observed open door before closing."""
import json
import numpy as np
from .redocking import SensorRedocking
from ..control.sensor_appliance_backend import camera_points


def closing_face(memory):
    """Transport the observed closed face by its measured handle/hinge arc."""
    hinge=np.asarray(memory['hinge'])
    a=np.asarray(memory['closed_handle'])[:2]-hinge[:2]
    b=np.asarray(memory['open_handle'])[:2]-hinge[:2]
    angle=np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)
    c,s=np.cos(angle),np.sin(angle)
    normal=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@memory['closed_normal']
    normal[2]=0.;normal/=np.linalg.norm(normal)
    return np.asarray(memory['open_handle']),normal


# The datagen oracle closes a door in strokes and stands, for each stroke, 0.60 m
# off the outer face the door will have in the MIDDLE of that stroke (22.5 deg
# toward closed for the first 45 deg), 0.20 m along the leaf. From there the arm
# reaches the still fully open leaf's far side, and a door that opens close to a
# wall leaves room to stand (this fridge door: 0.45 m to the wall at 90 deg).
# In this kitchen that exact spot leaves the base centre 0.30 m from the wall,
# just inside the docking margin, so spots for a further turned leaf follow.
MID_STROKE_DEGREES=(22.5,30.,37.5,45.)
MID_STROKE_ALONG_LEAF=(.20,.35,0.)


class ClosingRedocking(SensorRedocking):
    def __init__(self,runtime,sensors,mapper,memory):
        super().__init__(runtime,sensors,mapper)
        self.expected,self.approach_normal_map=closing_face(memory)
        self.memory=memory

    def mid_stroke_docks(self,distances):
        """Stand spots facing the leaf as it will be part-way through the first closing stroke."""
        hinge=np.asarray(self.memory['hinge'],float)[:2]
        opened=self.expected[:2]-hinge;closed=np.asarray(self.memory['closed_handle'],float)[:2]-hinge
        docks=[]
        for degrees in MID_STROKE_DEGREES:
            turn=np.sign(opened[0]*closed[1]-opened[1]*closed[0])*np.radians(degrees)
            c,s=np.cos(turn),np.sin(turn);R=np.array([[c,-s],[s,c]])
            leaf=R@opened;point=hinge+leaf;normal=R@self.approach_normal_map[:2];along=leaf/np.linalg.norm(leaf)
            for distance in distances:
                for shift in MID_STROKE_ALONG_LEAF:
                    xy=point+distance*normal+shift*along
                    docks.append(np.r_[xy,np.arctan2(point[1]-xy[1],point[0]-xy[0])])
        return docks

    def observe(self,target):
        # A plain move of this dock (back-off) keeps the base class's own target.
        if getattr(self,'plain_observe',False):return super().observe(target)
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not tracking.accepted:raise RuntimeError('Closing redock needs current estimated pose')
        T=tracking.T_map_base
        points=np.concatenate([camera_points(cam)[0] for cam in packet.cameras.values()])
        points=points@T[:3,:3].T+T[:3,3]
        distances=np.linalg.norm(points-self.expected,axis=1)
        if not len(points) or distances.min()>.15:
            # The remembered handle is on the leaf's far face. From the room
            # side of a door opened next to a wall it cannot be seen (fridge,
            # fix119: the close stopped here before any push). It was measured
            # by this robot while opening, so stand by the memory, as the
            # handle redock before closing already does.
            print('CLOSING_DOCK_EDGE_UNSEEN_USING_MEMORY',json.dumps(dict(target=target,
                nearest_depth_m=None if not len(points) else round(float(distances.min()),3))),flush=True)
            point=self.expected.copy()
        else:point=points[np.argmin(distances)].copy()
        self.target_point=point;self.target_label=target
        return packet,tracking,point,'observed_open_door'


# Stand spots for pushing a part-opened door further open from inside: this far
# from the remembered handle, in these directions (deg from the leaf's own line
# past its free edge; negative toward the inner face, positive toward the outer
# one), tried in order. The inner side comes first: from past the edge on the
# outer side the arm has to go round the free edge, which worked for the
# fridge at 25 deg (fix118; replay of fix126: 25 -> 80 deg from +30) but at
# 53 deg shoved the microwave door back to 33 deg on the way in (fix129).
PAST_EDGE_DISTANCE=.63
PUSH_OPEN_STAND_TURNS=(-60.,-30.,30.,0.)


class PastEdgeRedocking(ClosingRedocking):
    """Stand beside the free edge of a part-opened door, facing its handle."""
    strict_frontal=True          # exactly this spot: no bearing offsets
    no_closing_tiers=True        # and none of the closing-side fallback spots

    def __init__(self,runtime,sensors,mapper,memory,turn_degrees=30.):
        super().__init__(runtime,sensors,mapper,memory)
        hinge=np.asarray(memory['hinge'],float)[:2]
        along=self.expected[:2]-hinge;along/=np.linalg.norm(along)
        outer=np.asarray(self.approach_normal_map,float)[:2];outer/=np.linalg.norm(outer)
        turn=np.radians(turn_degrees)
        self.approach_normal_map=np.r_[np.cos(turn)*along+np.sin(turn)*outer,0.]

    def mid_stroke_docks(self,distances):return []
