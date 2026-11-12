"""Dock at the exposed handle face of an observed open door before closing."""
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


class ClosingRedocking(SensorRedocking):
    def __init__(self,runtime,sensors,mapper,memory):
        super().__init__(runtime,sensors,mapper)
        self.expected,self.approach_normal_map=closing_face(memory)

    def observe(self,target):
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not tracking.accepted:raise RuntimeError('Closing redock needs current estimated pose')
        T=tracking.T_map_base
        points=np.concatenate([camera_points(cam)[0] for cam in packet.cameras.values()])
        points=points@T[:3,:3].T+T[:3,3]
        distances=np.linalg.norm(points-self.expected,axis=1)
        if not len(points) or distances.min()>.15:
            raise RuntimeError('Previously opened door edge is not supported by current depth; obtain another view')
        point=points[np.argmin(distances)].copy()
        self.target_point=point;self.target_label=target
        return packet,tracking,point,'observed_open_door'
