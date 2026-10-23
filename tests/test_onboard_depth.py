from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.robot.observations import render_onboard_observations,depth_rgb

class Renderer:
 def __init__(self,fail=False):self.depth=False;self.fail=fail
 def disable_depth_rendering(self):self.depth=False
 def enable_depth_rendering(self):self.depth=True
 def update_scene(self,data,camera,scene_option=None):self.camera=camera
 def render(self):
  if self.depth:
   if self.fail:raise RuntimeError('depth failed')
   return np.ones((3,4),np.float32)*2
  return np.zeros((3,4,3),np.uint8)

def test_synchronized_modalities_and_mode_cleanup():
 r=Renderer();o=render_onboard_observations(r,SimpleNamespace(time=1.5),SimpleNamespace(namespace='robot_0/'))
 assert o['simulation_time']==1.5 and o['navigation_depth_m'].shape==(3,4)
 assert o['wrist_depth_m'].shape==(3,4)
 assert not r.depth and r.camera=='robot_0/wrist_cam'
 r=Renderer(True)
 with pytest.raises(RuntimeError):render_onboard_observations(r,SimpleNamespace(time=0),SimpleNamespace(namespace='robot_0/'))
 assert not r.depth

def test_depth_palette_fixed_and_invalid_black():
 d=depth_rgb(np.array([[1.,4.,8.,np.nan,0.]]))
 assert d.shape==(1,5,3)
 assert (d[0,3:]==0).all()
 assert not np.array_equal(d[0,0],d[0,1])
