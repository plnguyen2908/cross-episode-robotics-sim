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


def propose_pick_grasps(policy, points, rgb, target, samples=3, max_samples=8):
    """Retain the normal samples; retry empty target proposals before motion.

    Small objects can be missed by M2T2's random scene-point subsampling.
    Extra samples reuse the same observations and loaded model. They do not
    widen target acceptance, relax collisions, or synthesize oracle grasps.
    """
    import torch
    if samples < 1 or max_samples < samples:
        raise ValueError('Require 1 <= samples <= max_samples')
    rows=[]; diagnostics=[]
    for sample in range(max_samples):
        if sample >= samples and rows:
            break
        if sample:
            torch.manual_seed(sample)
        out=policy.predict(points,rgb)
        before=len(rows)
        for group,(poses,scores,contacts) in enumerate(zip(out['grasps'],out['grasp_confidence'],out['grasp_contacts'])):
            for pose,score,contact in zip(poses,scores,contacts):
                if np.linalg.norm(contact-target)<.16:
                    rows.append(dict(group=sample*10000+group,sampling_pass=sample,
                        confidence=float(score),contact=contact.tolist(),grasp=pose.tolist(),
                        tcp=grasp_to_tcp(pose).tolist()))
        diagnostics.append(dict(sample=sample,total_grasps=sum(map(len,out['grasps'])),
                                target_candidates=len(rows)-before))
    rows.sort(key=lambda row:-row['confidence'])
    return dict(candidates=rows,total_grasps=sum(row['total_grasps'] for row in diagnostics),
                sampling_passes=len(diagnostics),sampling_diagnostics=diagnostics,
                empty_target_retry=len(diagnostics)>samples)


M2T2_REPO, M2T2_REVISION = 'wentao-yuan/m2t2', 'a20325dcd19cd1b838274974d0ebe35bfe383796'


def default_checkpoint():
    """m2t2.pth: $M2T2_CHECKPOINT (eval_env.sh sets it), else the pinned revision in the Hugging Face cache.
    Only read when no M2T2 service is configured (the evaluation uses the service)."""
    import os
    if os.environ.get('M2T2_CHECKPOINT'):return os.environ['M2T2_CHECKPOINT']
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(M2T2_REPO, 'm2t2.pth', revision=M2T2_REVISION)
        if isinstance(found, str):return found
    except Exception:
        pass
    return str(Path(os.environ.get('HF_HOME', Path.home()/'.cache'/'huggingface'))/'hub'/'models--wentao-yuan--m2t2'
               /'snapshots'/M2T2_REVISION/'m2t2.pth')


class M2T2Policy:
    def __init__(self, checkpoint, repository, device='cuda:0', seed=0):
        from cross_episode_sim.perception.rpc import configured_client
        self.remote=configured_client('m2t2');self.seed=seed;self.remote_first=True
        if self.remote is not None:
            import torch
            torch.manual_seed(seed)
            return
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
        if self.remote is not None:
            import torch
            first=self.remote_first;self.remote_first=False
            return self.remote.call(points=np.asarray(points),rgb=np.asarray(rgb),held_points=held_points,ee_pose=ee_pose,seed=int(torch.initial_seed()),initial_sample=first)
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


def valid_grasp_poses(candidates):
    """Reject malformed transforms before rotation/IK code; keep valid proposals."""
    valid=[];rejected=[]
    for index,row in enumerate(candidates):
        try:
            tcp=np.asarray(row['tcp'],float)
            good=(tcp.shape==(4,4) and np.isfinite(tcp).all()
                  and np.allclose(tcp[3],[0,0,0,1],atol=1e-5)
                  and np.allclose(tcp[:3,:3].T@tcp[:3,:3],np.eye(3),atol=.005)
                  and abs(np.linalg.det(tcp[:3,:3])-1.)<.005)
        except (KeyError,ValueError,TypeError):good=False
        if good:valid.append(row)
        else:rejected.append(dict(index=index,reason='invalid_grasp_transform'))
    return valid,rejected


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
