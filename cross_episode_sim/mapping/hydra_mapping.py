"""Sensor-only packet contract for native MIT-SPARK Hydra.

No simulator imports, true poses, room graph, or object identity queries.
"""
from pathlib import Path
import json
import numpy as np

# IDs 1/2 stay compatible with existing sensor recordings. Legacy ID 3
# (table) is deliberately unclassified in the object-mapping path.
LABELS={'cup':1,'bowl':2}


def prepare_frame(camera,tracking,detections,timestamp):
    if not tracking.accepted:return None
    T=np.asarray(tracking.T_map_base)@camera.T_base_optical
    if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid estimated camera pose')
    depth=np.asarray(camera.depth_m,dtype=np.float32).copy()
    valid=camera.valid_depth & np.isfinite(depth) & (depth>.1) & (depth<6.)
    if camera.robot_mask is not None:valid &= ~camera.robot_mask
    depth[~valid]=0.
    labels=np.zeros(depth.shape,dtype=np.int32)
    # A conflicting semantic mask is unknown, never whichever label was last.
    conflict=np.zeros(depth.shape,bool)
    for detection in detections:
        if detection.mask is None or detection.label not in LABELS:continue
        if detection.mask.shape!=depth.shape:raise ValueError('Mask/image shape mismatch')
        mask=np.asarray(detection.mask,bool)&valid
        code=LABELS[detection.label]
        conflict |= mask&(labels!=0)&(labels!=code)
        labels[mask & ~conflict]=code
    labels[conflict]=0
    return dict(rgb=np.asarray(camera.rgb,dtype=np.uint8),depth=depth,labels=labels,
                K=np.asarray(camera.K,dtype=float),T_map_optical=T,timestamp_ns=np.int64(round(timestamp*1e9)))


def refine_semantics(depth,labels,K=None,T=None):
    """Do not paint uncertain segmentation silhouettes onto background geometry.

    Keep geometry unchanged. Erode each predicted mask by one pixel, then reject
    semantic pixels touching a >10 cm depth discontinuity. No map/world IDs.
    """
    from scipy.ndimage import binary_erosion,maximum_filter,minimum_filter
    depth=np.asarray(depth);out=np.zeros_like(labels)
    valid=np.isfinite(depth)&(depth>0)
    low=minimum_filter(np.where(valid,depth,np.inf),size=3,mode='nearest')
    high=maximum_filter(np.where(valid,depth,-np.inf),size=3,mode='nearest')
    smooth=valid&((high-low)<.1)
    for code in LABELS.values():
        keep=binary_erosion(labels==code,structure=np.ones((3,3),bool))&smooth
        out[keep]=code
    return out


class HydraFrameWriter:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.count=0;self.last_timestamp=-1
        (self.output/'labels.json').write_text(json.dumps(LABELS,indent=2))

    def update(self,camera,tracking,detections,timestamp):
        frame=prepare_frame(camera,tracking,detections,timestamp)
        if frame is None:return None
        if frame['timestamp_ns']<=self.last_timestamp:raise ValueError('Frames must be time ordered')
        path=self.output/f'frame_{self.count:06d}.npz';np.savez_compressed(path,**frame)
        self.count+=1;self.last_timestamp=int(frame['timestamp_ns']);return path


class NativeHydraMapper:
    """Calls the actual compiled HydraPipeline, with no replacement mapper."""
    def __init__(self,first_frame,config_file=None):
        import yaml
        import spark_dsg  # Registers upstream graph types used by Hydra bindings.
        from hydra_python import HydraPipeline
        from hydra_python.pipeline import make_camera,load_configs
        K=first_frame['K'];h,w=first_frame['depth'].shape
        self.K=K.copy();self.shape=(h,w);self.last_timestamp=-1
        camera=make_camera(K[0,0],K[1,1],K[0,2],K[1,2],w,h,max_range=6.)
        if config_file:config=yaml.safe_load(Path(config_file).read_text())
        else:
            config=load_configs('habitat')
            config.pop('reconstruction',None)  # Legacy key added by upstream helper; active_window is current API.
            config.update(total_semantic_labels=3,invalid_labels=[],dynamic_labels=[],object_labels=list(LABELS.values()),surface_places_labels=[],label_names=[dict(label=0,name='unclassified')]+[dict(label=v,name=k) for k,v in LABELS.items()])
            config['frontend']['enable_mesh_objects']=True
            config['frontend']['objects']={'min_cluster_size':10,'cluster_tolerance':.1,'bounding_box_type':'AABB'}
            config['active_window']['type']='ReconstructionModule'
            config['active_window']['volumetric_map'].update(voxel_size=.03,truncation_distance=.12,with_semantics=True)
            config['active_window']['full_update_separation_s']=0.
            config['active_window']['tsdf']['semantic_integrator']={'type':'SingleLabelIntegrator','labels_to_ignore':[0]}
            config['frontend']['freespace_places']['gvd'].update(min_distance_m=.06,min_diff_m=.03)
            config['frontend']['type']='GraphBuilder'
            config['backend']['type']='BackendModule'
            config['backend']['optimize_on_lc']=False
            config['enable_lcd']=False
        self.config=config
        self.pipeline=HydraPipeline.from_config(yaml.safe_dump(config),camera,use_step_mode=True)

    def update(self,frame):
        from scipy.spatial.transform import Rotation
        if frame['depth'].shape!=self.shape or not np.allclose(frame['K'],self.K):raise ValueError('Camera calibration changed')
        stamp=int(frame['timestamp_ns'])
        if stamp<=self.last_timestamp:raise ValueError('Frames must be time ordered')
        T=frame['T_map_optical'];q=Rotation.from_matrix(T[:3,:3]).as_quat()
        result=self.pipeline.step(stamp,T[:3,3],q[[3,0,1,2]],frame['depth'],refine_semantics(frame['depth'],frame['labels'],frame['K'],T),frame['rgb'])
        if not result:raise RuntimeError('Native Hydra rejected sensor frame')
        self.last_timestamp=stamp
        return self.pipeline.graph

    def save(self,output):
        import yaml
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        self.pipeline.save(output)
        self.pipeline.save_graph(str(output/'scene_graph.json'),True)
        (output/'effective_config.yaml').write_text(yaml.safe_dump(self.config))
