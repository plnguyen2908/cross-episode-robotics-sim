"""Fit a horizontal support edge from onboard point clouds, without scene IDs."""
import numpy as np


def tabletop_front(points,target,robot_xy=(0.,0.)):
    import cv2
    points=np.asarray(points,float);target=np.asarray(target,float)
    nearby=points[np.isfinite(points).all(1)&
        (np.linalg.norm(points[:,:2]-target[:2],axis=1)<1.5)&
        (points[:,2]>target[2]-.35)&(points[:,2]<target[2]+.01)]
    if len(nearby)<100:raise ValueError('Insufficient observed tabletop points')
    levels,counts=np.unique(np.round(nearby[:,2]/.005),return_counts=True)
    # Prefer the upper broad plane when equally well observed chair seats and
    # tabletop are present; tiny object tops cannot displace a dominant table.
    height=float(levels[counts>=.5*counts.max()][-1]*.005)
    surface=nearby[abs(nearby[:,2]-height)<.008]
    xy=np.unique(np.round(surface[:,:2]/.01),axis=0)*.01
    if len(xy)<80:raise ValueError('Insufficient tabletop extent')
    corners=cv2.boxPoints(cv2.minAreaRect(xy.astype(np.float32))).astype(float)
    center=corners.mean(0);choices=[]
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        tangent=b-a;length=np.linalg.norm(tangent)
        if length<.30:continue
        tangent/=length;normal=np.array([-tangent[1],tangent[0]])
        midpoint=(a+b)/2
        if normal@(midpoint-center)<0:normal=-normal
        if normal@(np.asarray(robot_xy)-midpoint)<.02:continue
        along=(target[:2]-a)@tangent
        if not -.05<=along<=length+.05:continue
        edge=target[:2]+normal*((midpoint-target[:2])@normal)
        dock=target[:2]+.55*normal
        choices.append((float(np.linalg.norm(dock-robot_xy)),normal,edge))
    if not choices:raise ValueError('No visible facing tabletop edge')
    _,normal,edge=min(choices,key=lambda row:row[0])
    return dict(normal_base=[*normal,0.],edge_base=[*edge,height],
        height=height,points=len(surface),corners=corners.tolist(),source='observed horizontal tabletop boundary')


def packet_tabletop_front(packet,T,target):
    from ..control.sensor_appliance_backend import camera_points
    clouds=[camera_points(camera)[0] for camera in packet.cameras.values()]
    points=np.concatenate(clouds)
    local=(np.asarray(target)-T[:3,3])@T[:3,:3]
    front=tabletop_front(points,local)
    return front,T[:3,:3]@np.asarray(front['normal_base'])
