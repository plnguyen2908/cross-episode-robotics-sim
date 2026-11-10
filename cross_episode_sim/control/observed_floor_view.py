"""Wrist viewing poses from calibration and a robot-relative sensing target."""
import numpy as np


def wrist_floor_pose(tcp, optical, target):
    tcp=np.asarray(tcp,float);optical=np.asarray(optical,float)
    relative=np.linalg.inv(tcp)@optical
    forward=np.asarray(target,float)-optical[:3,3]
    if np.linalg.norm(forward)<.05:raise ValueError('Floor viewing target too close')
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.])
    if np.linalg.norm(right)<1e-6:right=optical[:3,0]-forward*(forward@optical[:3,0])
    right/=np.linalg.norm(right);down=np.cross(forward,right)
    result=tcp.copy();result[:3,:3]=np.column_stack((right,down,forward))@relative[:3,:3].T
    # Keep the camera center stationary while changing its viewing direction.
    result[:3,3]=optical[:3,3]-result[:3,:3]@relative[:3,3]
    return result
