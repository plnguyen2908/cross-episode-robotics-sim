"""RGB detections + measured depth; no simulator IDs or world poses."""
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from .surfaces import table_surface_mask,project_mask,compact_surface,surface_overlap


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = None


class ObjectDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class GroundingDinoDetector:
    def __init__(self, device='cuda', model='IDEA-Research/grounding-dino-tiny', threshold=.3,verify_tables=True):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch=torch; self.device=device; self.threshold=threshold
        self.processor=AutoProcessor.from_pretrained(model)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device).eval()
        self.table_verifier=TableVerifier(device) if verify_tables else None

    def detect(self, rgb, text=None):
        from PIL import Image
        inputs=self.processor(images=Image.fromarray(rgb),text=text or 'a cup. a mug. a drinking glass. a wine glass. a bowl. a dining table. a kitchen countertop. the floor. a robot arm. a chair.',return_tensors='pt').to(self.device)
        with self.torch.inference_mode(): outputs=self.model(**inputs)
        result=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=self.threshold,text_threshold=.25,target_sizes=[rgb.shape[:2]])[0]
        detections=[]
        for box,score,label in zip(result['boxes'],result['scores'],result['text_labels']):
            label=label.lower()
            if any(word in label for word in ('countertop','floor','robot','chair')):continue
            category=next((canonical for word,canonical in [('table','dining table'),('bowl','bowl'),('mug','cup'),('cup','cup'),('glass','cup')] if word in label),None)
            if category:
                coordinates=tuple(box.cpu().tolist())
                if category=='dining table' and self.table_verifier is not None and not self.table_verifier.accept(rgb,coordinates):continue
                detection=Detection(category,float(score),coordinates)
                # Grounding DINO can emit nested duplicates for one object.
                if any(d.label==category and box_iou(d.box,coordinates)>.5 for d in detections):continue
                detections.append(detection)
        return detections


def locate_detection(camera, detection, T_map_base):
    """Robust central-box depth cluster, not an object segmentation oracle.

    Returns an observed surface point, not the object's physical centre or grasp.
    Reject sparse/noisy depth rather than fabricating a location.
    """
    if detection.label=='dining table':
        points=table_points(camera,detection,T_map_base)
        return np.median(points,axis=0) if len(points) else None
    h,w=camera.depth_m.shape
    x0,y0,x1,y1=np.asarray(detection.box,float)
    if not np.isfinite([x0,y0,x1,y1]).all() or x1<=x0 or y1<=y0:return None
    dx=(x1-x0)*.2 if detection.mask is None else 0.
    dy=(y1-y0)*.2 if detection.mask is None else 0.
    x0,x1=max(0,int(x0+dx)),min(w,int(np.ceil(x1-dx)))
    y0,y1=max(0,int(y0+dy)),min(h,int(np.ceil(y1-dy)))
    if x1<=x0 or y1<=y0:return None
    v,u=np.mgrid[y0:y1,x0:x1];z=camera.depth_m[y0:y1,x0:x1]
    valid=camera.valid_depth[y0:y1,x0:x1]&np.isfinite(z)&(z>.1)&(z<6.)
    if detection.mask is not None:
        if detection.mask.shape!=camera.depth_m.shape:return None
        valid &= detection.mask[y0:y1,x0:x1]
    if camera.robot_mask is not None:
        if camera.robot_mask[y0:y1,x0:x1].mean()>.25:return None
        valid &= ~camera.robot_mask[y0:y1,x0:x1]
    if valid.sum()<12:return None
    if detection.mask is not None and detection.label in {'cup','bowl'}:
        # Validate the COMPLETE mask before selecting its central depth cluster.
        # Otherwise a thin slice of a huge counter can look like a small vessel.
        zz=z[valid];uu=u[valid];vv=v[valid];K=camera.K
        whole=np.column_stack(((uu-K[0,2])*zz/K[0,0],(vv-K[1,2])*zz/K[1,1],zz))
        T=T_map_base@camera.T_base_optical
        whole=whole@T[:3,:3].T+T[:3,3]
        low,high=np.percentile(whole,[5,95],axis=0)
        if np.max(high-low)>.75 or high[2]<.05 or high[2]-low[2]<.015:return None
    depths=z[valid];median=np.median(depths)
    valid &= np.abs(z-median)<max(.08,.04*median)
    if valid.sum()<12:return None
    z=z[valid];u=u[valid];v=v[valid];K=camera.K
    points=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
    T=T_map_base@camera.T_base_optical
    points=points@T[:3,:3].T+T[:3,3]
    # Reject physically implausible vessel hypotheses from masks spanning
    # walls/floors/counters. This uses measured extent, never scene identities.
    if detection.mask is not None and detection.label in {'cup','bowl'}:
        span=np.percentile(points,95,axis=0)-np.percentile(points,5,axis=0)
        if np.max(span)>.75:return None
    point=np.median(points,axis=0)
    # Broad physical category priors in the gravity-aligned SLAM frame, not
    # scene-specific coordinates. A floor patch is not a tabletop.
    if detection.label=='dining table' and not .45<point[2]<1.25:return None
    return point


def table_points(camera,detection,T_map_base):
    mask=np.zeros(camera.depth_m.shape,bool)
    if detection.mask is not None:mask[:]=detection.mask
    else:
        h,w=mask.shape;x0,y0,x1,y1=detection.box
        mask[max(0,int(y0)):min(h,int(np.ceil(y1))),max(0,int(x0)):min(w,int(np.ceil(x1)))]=True
    mask &= camera.valid_depth
    if camera.robot_mask is not None:mask &= ~camera.robot_mask
    T=T_map_base@camera.T_base_optical
    # A detector box legitimately includes floor between table legs. The
    # support-fraction test applies to an actual segmented object mask only.
    mask=table_surface_mask(camera.depth_m,mask,camera.K,T,min_surface_fraction=.2 if detection.mask is not None else 0.)
    return compact_surface(project_mask(camera.depth_m,mask,camera.K,T)) if mask.any() else np.empty((0,3))


@dataclass
class VisualTarget:
    track_id: int
    label: str
    position: np.ndarray
    score: float
    observations: int=1
    last_seen: float=0.
    history: list=field(default_factory=list)
    surface_points: list=field(default_factory=list)
    native_id: str | None=None
    navigation_memory: bool=False


class VisualTargetMap:
    def __init__(self):self.targets=[]

    def update(self, camera, detections, tracking, timestamp):
        if not tracking.accepted:return []
        updated=[]
        for detection in sorted(detections,key=lambda d:-d.score):
            surface=table_points(camera,detection,tracking.T_map_base) if detection.label=='dining table' else np.empty((0,3))
            position=(np.median(surface,axis=0) if len(surface) else None) if detection.label=='dining table' else locate_detection(camera,detection,tracking.T_map_base)
            if position is None:continue
            def projects_into_mask(target):
                if detection.mask is None:return True
                T=np.linalg.inv(tracking.T_map_base@camera.T_base_optical)
                q=T[:3,:3]@target.position+T[:3,3]
                if q[2]<=0:return False
                pixel=camera.K@q;u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
                return 0<=v<detection.mask.shape[0] and 0<=u<detection.mask.shape[1] and detection.mask[v,u]
            if detection.label=='dining table':
                candidates=[t for t in self.targets if t.label==detection.label and surface_overlap(surface,t.surface_points)>=.25]
            else:
                candidates=[t for t in self.targets if t.label==detection.label and np.linalg.norm(t.position-position)<.35 and projects_into_mask(t)]
            if candidates:
                target=min(candidates,key=lambda t:np.linalg.norm(t.position-position))
                if timestamp<=target.last_seen:continue
                target.position=.7*target.position+.3*position;target.observations+=1
                target.score=max(target.score,detection.score);target.last_seen=timestamp
            else:
                target=VisualTarget(len(self.targets),detection.label,position,detection.score,last_seen=timestamp)
                self.targets.append(target)
            if len(surface):
                combined=np.vstack((np.asarray(target.surface_points).reshape(-1,3),surface))
                target.surface_points=compact_surface(combined).tolist()
                target.position=np.median(target.surface_points,axis=0)
            target.history.append(dict(timestamp=timestamp,box=list(detection.box),position=position.tolist(),camera=camera.frame_id))
            updated.append(target)
        return updated


def box_iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    intersection=max(0,x1-x0)*max(0,y1-y0)
    return intersection/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


class TableVerifier:
    """Contrastive RGB crop classification; no scene labels or simulator inputs."""
    def __init__(self,device='cuda'):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch=torch;self.device=device
        name='openai/clip-vit-large-patch14'
        self.processor=CLIPProcessor.from_pretrained(name)
        self.model=CLIPModel.from_pretrained(name).to(device).eval()
        self.labels=['a dining table with chairs for eating meals',
                     'a kitchen countertop with cupboards and appliances',
                     'a floor in a room','a robot arm and gripper',
                     'a kitchen cabinet or cupboard']
        with torch.inference_mode():
            inputs=self.processor(text=self.labels,padding=True,return_tensors='pt').to(device)
            features=self.model.get_text_features(**inputs)
            self.text_features=features/features.norm(dim=-1,keepdim=True)
        self.last_scores=None

    def accept(self,rgb,box):
        from PIL import Image
        h,w=rgb.shape[:2];x0,y0,x1,y1=box
        pad=.2*max(x1-x0,y1-y0)
        bounds=(max(0,int(x0-pad)),max(0,int(y0-pad)),min(w,int(x1+pad)),min(h,int(y1+pad)))
        if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:return False
        crop=Image.fromarray(rgb).crop(bounds)
        with self.torch.inference_mode():
            inputs=self.processor(images=crop,return_tensors='pt').to(self.device)
            feature=self.model.get_image_features(**inputs);feature=feature/feature.norm(dim=-1,keepdim=True)
            scores=(feature@self.text_features.T)[0].float().cpu().numpy()
        self.last_scores=dict(zip(self.labels,map(float,scores)))
        return int(np.argmax(scores))==0 and scores[0]-max(scores[1:])>.01
