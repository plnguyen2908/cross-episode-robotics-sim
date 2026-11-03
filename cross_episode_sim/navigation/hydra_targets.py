"""Native Hydra object positions with independent fresh sensor confirmation."""
import numpy as np
from ..perception.visual_targets import VisualTarget,locate_detection


class HydraTargetMap:
    def __init__(self):
        self.targets=[];self.by_native_id={};self.next_id=0;self.last_update=None

    def update(self,objects,cameras,detections,tracking,timestamp):
        if not tracking.accepted:return
        previous={key:t.position.copy() for key,t in self.by_native_id.items()}
        live=set()
        for obj in objects:
            if obj['label'] not in {'cup','bowl'}:continue
            native_id=obj['id']
            position=np.asarray(obj['position'],float)
            if position.shape!=(3,) or not np.isfinite(position).all():continue
            live.add(native_id)
            target=self.by_native_id.get(native_id)
            if target is None:
                target=VisualTarget(self.next_id,obj['label'],position,0.,observations=0,last_seen=-float('inf'))
                target.native_id=native_id
                self.next_id+=1;self.by_native_id[native_id]=target
            target.position=position.copy()  # Native mapping output, not RGB centroid replacement.
        self.targets=[t for key,t in self.by_native_id.items() if key in live]
        for name,camera in cameras.items():
            T=np.linalg.inv(tracking.T_map_base@camera.T_base_optical)
            associations=[]
            for i,det in enumerate(detections.get(name,[])):
                if det.mask is None:continue
                measured=locate_detection(camera,det,tracking.T_map_base)
                if measured is None:continue
                for target in self.targets:
                    if target.label!=det.label:continue
                    distance=float(np.linalg.norm(target.position-measured))
                    if distance>.25:continue
                    q=T[:3,:3]@target.position+T[:3,3]
                    if q[2]<=0:continue
                    pixel=camera.K@q;u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
                    if not(0<=v<det.mask.shape[0] and 0<=u<det.mask.shape[1] and det.mask[v,u]):continue
                    associations.append((distance,i,target))
            used_detections=set();used_targets=set()
            for distance,i,target in sorted(associations,key=lambda row:row[0]):
                if i in used_detections or target.track_id in used_targets:continue
                used_detections.add(i);used_targets.add(target.track_id)
                if timestamp<=target.last_seen:continue
                target.last_seen=timestamp;target.observations+=1;target.score=detections[name][i].score
                target.history.append(dict(timestamp=timestamp,camera=name,match_distance_m=distance,position=target.position.tolist()))
        for key,target in self.by_native_id.items():
            if key in previous and target.last_seen!=timestamp:
                target.position=previous[key]  # Do not steer using an unconfirmed map-centroid drift.
        self.last_update=timestamp
