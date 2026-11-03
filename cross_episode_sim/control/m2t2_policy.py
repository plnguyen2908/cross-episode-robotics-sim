"""Pretrained M2T2 target proposals from observed metric points; no simulator API."""
from pathlib import Path
import numpy as np


def filter_observed_contacts(candidates, target_surface, tolerance=.015):
    """Keep learned contacts on the sensed target, not a nearby object.

    M2T2 contacts are scene points. A broad centroid-radius prefilter alone
    can include a neighboring bowl when the requested target is a small cup.
    """
    from scipy.spatial import cKDTree
    points=np.asarray(target_surface,float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError('Need observed target surface')
    if not candidates:return []
    distance=cKDTree(points).query(np.array([row['contact'] for row in candidates]))[0]
    return [row for row,gap in zip(candidates,distance) if gap<=tolerance]


class M2T2Policy:
    def __init__(self, checkpoint, repository, device='cuda:0', seed=0):
        import sys
        sys.path.insert(0,str(Path(repository).resolve()))
        import torch
        from omegaconf import OmegaConf
        from m2t2.m2t2 import M2T2
        torch.manual_seed(seed)
        self.torch=torch; self.device=device
        self.cfg=OmegaConf.load(Path(repository)/'config.yaml')
        self.model=M2T2.from_config(self.cfg.m2t2)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model.load_state_dict(state['model'])
        self.model=self.model.to(device).eval()

    def predict(self, points, rgb, *, held_points=None, ee_pose=None):
        """Points and TCP pose share a gravity-aligned, metric robot-base frame.

        held_points must be perceived object points, never mesh/pose ground truth.
        Returns M2T2 gripper-root poses (NOT our robot's TCP poses).
        """
        t=self.torch
        from m2t2.dataset_utils import sample_points
        p=t.as_tensor(points,dtype=t.float32)
        color=t.as_tensor(rgb,dtype=t.float32)/255
        color=(color-t.tensor([.485,.456,.406]))/t.tensor([.229,.224,.225])
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<32 or not t.isfinite(p).all():
            raise ValueError('Need finite metric scene points')
        inputs=t.cat([p-p.mean(0),color],dim=1)
        index=sample_points(p,self.cfg.data.num_points)
        ee=t.eye(4) if ee_pose is None else t.as_tensor(ee_pose,dtype=t.float32)
        obj=t.rand(1024,6);bottom=t.zeros(3)
        if held_points is not None:
            hp=t.as_tensor(held_points,dtype=t.float32)
            bottom=hp.min(0).values;bottom[:2]=t.unique((hp[:,:2]/.01).round(),dim=0).mean(0)*.01
            local=(hp-ee[:3,3])@ee[:3,:3]
            obj=t.cat([local-local.mean(0),t.zeros_like(local)],dim=1)
            obj=obj[sample_points(obj,1024)]
        data={k:v.unsqueeze(0).to(self.device) for k,v in dict(inputs=inputs[index],points=p[index],
            object_inputs=obj,ee_pose=ee,bottom_center=bottom,cam_pose=t.eye(4),task_is_place=t.tensor(float(held_points is not None))).items()}
        with t.inference_mode():out=self.model.infer(data,self.cfg.eval)
        def arrays(key):return [x.detach().cpu().numpy() for x in out[key][0]]
        return {key:arrays(key) for key in ('grasps','grasp_confidence','grasp_contacts','placements','placement_confidence','placement_contacts')}


def grasp_to_tcp(grasp, contact_depth=.1034):
    """M2T2 Panda-root → contact center; map closing x to our TCP closing y.

    Our tool frame has +z along approach and +y between opposing pads.
    This is a static gripper convention conversion, not an object-pose offset.
    """
    g=np.asarray(grasp);out=g.copy()
    out[:3,3]=g[:3,3]+contact_depth*g[:3,2]
    out[:3,:3]=g[:3,:3]@np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    return out


def parallel_gripper_symmetries(candidates):
    """Equivalent parallel-jaw grasps, each requiring robot motion validation.

    A half-turn about the approach axis swaps the opposing pads without changing
    the contact center or closing line. It does not assert arm reachability.
    """
    result=[]
    for index,row in enumerate(candidates):
        for flipped in (False,True):
            tcp=np.asarray(row['tcp'],float).copy()
            if flipped:tcp[:3,:3]=tcp[:3,:3]@np.diag([-1.,-1.,1.])
            result.append(dict(row,tcp=tcp.tolist(),model_tcp=row['tcp'],parallel_jaw_flip=flipped,prediction_index=index))
    return result
