"""Initial observed-counter viewing direction for an out-of-view food target."""
import numpy as np
from PIL import Image
from cross_episode_sim.navigation.hybrid_navigation import HybridNavigator
from cross_episode_sim.control.sensor_appliance_backend import pixel_point
class CounterFoodNavigator(HybridNavigator):
 def step(self,packet,tracking,now):
  label=self.semantic.target_label
  if label.startswith('food container') and 'microwave' not in label and not getattr(self,'counter_view_done',False) and tracking.accepted:
   self.pointing.observe(packet,tracking)
   target=self._target(tracking.T_map_base[:2,3],now)
   if target is not None:self.counter_view_done=True
   else:
    if not hasattr(self,'counter_view_hint'):
     pointer=self.pointing.pointer.pointer
     camera=packet.cameras['navigation']
     result=pointer.point(Image.fromarray(camera.rgb),'kitchen countertop surface, excluding appliances and cabinet fronts')
     hints=[pixel_point(camera,camera.valid_depth,p) for p in result['points']]
     hints=[p for p in hints if p is not None and .4<p[2]<1.4 and np.linalg.norm(p[:2])>.4]
     self.counter_view_hint=(tracking.T_map_base[:3,:3]@hints[0]+tracking.T_map_base[:3,3]) if hints else None
     self.counter_view_started=now
     print('COUNTER_VIEW',None if self.counter_view_hint is None else self.counter_view_hint.tolist(),flush=True)
    if self.counter_view_hint is not None and now-self.counter_view_started<8.:
     command=self.local.face_point(packet,tracking,now,self.counter_view_hint)
     if command['status'] not in ('viewpoint_facing_ready','viewpoint_turn_blocked'):return command
    self.counter_view_done=True
  return super().step(packet,tracking,now)
