"""Onboard robot-mask occlusion near a measured semantic target."""
import numpy as np


def robot_occlusion_fraction(camera,target_base,radius_pixels=24):
    if camera.robot_mask is None:return 0.
    optical=np.linalg.inv(camera.T_base_optical)@np.r_[target_base,1.]
    if optical[2]<=0:return 0.
    uv=camera.K@optical[:3];u,v=np.round(uv[:2]/uv[2]).astype(int)
    height,width=camera.robot_mask.shape
    if not (0<=u<width and 0<=v<height):return 0.
    patch=camera.robot_mask[max(0,v-radius_pixels):min(height,v+radius_pixels+1),
                            max(0,u-radius_pixels):min(width,u+radius_pixels+1)]
    return float(np.mean(patch))
