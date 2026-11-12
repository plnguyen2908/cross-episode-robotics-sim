"""Reuse the observed grocery track and approach its measured support front."""
import json
import numpy as np
from PIL import Image
from .redocking import SensorRedocking
from ..perception.portable_object_pointing import GroceryPointer
from ..perception.tabletop_front import packet_tabletop_front


class GroceryRedocking(SensorRedocking):
    strict_frontal=True

    def observe(self,target):
        self.target_confirmed=False
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        confirmed=None
        if tracking.accepted and self.target_label==target and self.target_point is not None:
            T=tracking.T_map_base
            local=(self.target_point-T[:3,3])@T[:3,:3]
            pointer=self.grocery_pointers.setdefault(target,GroceryPointer(self.pointer,target))
            for key,camera,support,component,measured in self.tracked_surface(packet,local):
                optical=camera.T_base_optical[:3,:3].T@(measured-camera.T_base_optical[:3,3])
                if optical[2]<=0:continue
                uv=camera.K@optical;uv=uv[:2]/uv[2]
                associated=(local[2]>support+.012 and
                    np.min(np.linalg.norm(component-local,axis=1))<.05)
                if associated:
                    result=dict(points=[uv.tolist()],category_verification=[dict(
                        accepted=True,source='prior semantic identity + fresh above-support connected surface')])
                else:
                    result=pointer.verify(Image.fromarray(camera.rgb),dict(points=[uv.tolist()],raw='projected observed target'))
                print('GROCERY_TRACK_CONFIRMATION',json.dumps(dict(camera=key,depth_gap_m=float(np.linalg.norm(measured-local)),
                    observed_support_z=support,object_surface_points=len(component),
                    evidence=result['category_verification'])),flush=True)
                if result['points']:
                    self.target_point=T[:3,:3]@measured+T[:3,3];self.target_confirmed=True
                    confirmed=(packet,tracking,self.target_point.copy(),key);break
        if confirmed is None:confirmed=super().observe(target)
        packet,tracking,point,_=confirmed
        front,normal=packet_tabletop_front(packet,tracking.T_map_base,point)
        self.approach_normal_map=normal;self.observed_support_front=front
        print('GROCERY_FRONTAL_SUPPORT',json.dumps(front),flush=True)
        return confirmed
