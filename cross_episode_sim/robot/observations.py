"""Two physical cameras, four synchronized observation modalities."""
import numpy as np

OBSERVATION_PANELS = ('navigation_rgb', 'navigation_depth', 'wrist_rgb', 'wrist_depth')
DEPTH_DISPLAY_MAX_M = 8.


def depth_rgb(depth, maximum=DEPTH_DISPLAY_MAX_M):
    """Fixed metric range; purple=near, yellow=far, invalid=black."""
    if maximum<=0:raise ValueError('Depth display maximum must be positive')
    depth=np.asarray(depth);valid=np.isfinite(depth)&(depth>0)
    t=np.clip(np.where(valid,depth,0)/maximum,0,1)
    anchors=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],float)
    x=t*4;i=np.minimum(x.astype(int),3);f=(x-i)[...,None]
    rgb=anchors[i]*(1-f)+anchors[i+1]*f;rgb[~valid]=0
    return rgb.astype(np.uint8)


def render_onboard_observations(renderer,data,profile,scene_option=None):
    from .cameras import onboard_camera_names
    nav,wrist=onboard_camera_names(profile)
    renderer.disable_depth_rendering()
    renderer.update_scene(data,camera=nav,scene_option=scene_option)
    rgb=renderer.render().copy()
    try:
        renderer.enable_depth_rendering()
        renderer.update_scene(data,camera=nav,scene_option=scene_option)
        depth=renderer.render().copy()
    finally:renderer.disable_depth_rendering()
    renderer.update_scene(data,camera=wrist,scene_option=scene_option)
    wrist_rgb=renderer.render().copy()
    try:
        renderer.enable_depth_rendering()
        renderer.update_scene(data,camera=wrist,scene_option=scene_option)
        wrist_depth=renderer.render().copy()
    finally:renderer.disable_depth_rendering()
    return dict(navigation_rgb=rgb,navigation_depth_m=depth,wrist_rgb=wrist_rgb,
                wrist_depth_m=wrist_depth,simulation_time=float(data.time))
