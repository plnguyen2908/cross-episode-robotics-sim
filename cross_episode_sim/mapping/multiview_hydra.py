"""Native Hydra mapping for independently calibrated onboard cameras."""
import numpy as np
from .hydra_process import NativeHydraProcess


def merge_camera_objects(primary,secondary,view,association_distance=.18):
    """Keep native positions; suppress same-category cross-view duplicates."""
    objects=list(primary)
    for obj in secondary:
        if any(o['label']==obj['label'] and np.linalg.norm(np.asarray(o['position'])-obj['position'])<association_distance for o in objects):
            continue
        objects.append(dict(obj,id=f'{view}/{obj["id"]}'))
    return objects


class MultiCameraHydraProcess(NativeHydraProcess):
    """One upstream mapper per calibration; navigation graph retains room nodes.

    Inputs are synchronized sensor cameras, estimated poses and visual labels.
    Object coordinates remain upstream Hydra outputs. Robot truth is unavailable.
    """
    def __init__(self,output,root=None):
        super().__init__(output,root)
        self.secondary={};self.worker_root=root

    def update_many(self,cameras,tracking,detections,timestamp):
        result=self.update(cameras['navigation'],tracking,detections.get('navigation',[]),timestamp)
        if result is None:return None
        result=dict(result);result['objects']=list(result['objects'])
        for name,camera in cameras.items():
            if name=='navigation':continue
            if name not in self.secondary:
                self.secondary[name]=NativeHydraProcess(self.output/name,root=self.worker_root)
            mapped=self.secondary[name].update(camera,tracking,detections.get(name,[]),timestamp)
            if mapped is not None:
                result['objects']=merge_camera_objects(result['objects'],mapped['objects'],name)
        return result

    def close(self):
        try:
            for mapper in getattr(self,'secondary',{}).values():mapper.close()
        finally:super().close()
