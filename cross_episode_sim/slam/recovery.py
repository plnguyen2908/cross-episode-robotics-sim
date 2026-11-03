"""Learned recovery proposals, accepted only against measured RGB-D geometry.

No simulator state is accepted. VGGT supplies an initial relative pose; measured
depth supplies metric scale and geometric refinement. The learned point cloud is
never fused into the RGB-D map.
"""
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def planar(T, mount):
    base = T @ np.linalg.inv(mount)
    yaw = np.arctan2(base[1, 0], base[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    base[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    base[2, 3] = 0
    return base @ mount


class VGGTProposal:
    """Upstream VGGT pair inference. Lazy-loaded; no graph or scene truth."""
    def __init__(self, upstream, weights, device='cuda'):
        self.upstream, self.weights, self.device = Path(upstream), Path(weights), device
        self.model = None

    def __call__(self, reference_rgb, current_rgb, reference_depth):
        import torch
        from PIL import Image
        if self.model is None:
            sys.path.insert(0, str((self.upstream/'third_party/vggt').resolve()))
            from vggt.models.vggt import VGGT
            self.model = VGGT()
            self.model.load_state_dict(torch.load(self.weights, map_location='cpu'))
            self.model = self.model.eval().to(device=self.device, dtype=torch.bfloat16)
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        h, w = reference_depth.shape
        width, height = 518, round(h*518/w/14)*14
        if height > 518:
            raise ValueError('Recovery currently supports landscape camera images')
        imgs = [np.asarray(Image.fromarray(x).resize((width, height), Image.Resampling.BICUBIC)) for x in (reference_rgb, current_rgb)]
        inputs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(self.device)/255
        with torch.inference_mode():
            pred = self.model(inputs)
            extrinsics, _ = pose_encoding_to_extri_intri(pred['pose_enc'], inputs.shape[-2:])
        ext = extrinsics[0].float().cpu().numpy()
        if not np.isfinite(ext).all():
            return None
        matrices = np.tile(np.eye(4), (2, 1, 1)); matrices[:, :3] = ext
        delta = matrices[0] @ np.linalg.inv(matrices[1])  # current optical -> reference optical
        inferred = pred['depth'][0, 0, ..., 0].float().cpu().numpy()
        observed = cv2.resize(reference_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        valid = (observed > .65) & (observed < 6) & np.isfinite(inferred) & (inferred > .01)
        if valid.sum() < 1000:
            return None
        delta[:3, 3] *= np.median(observed[valid]/inferred[valid])
        return delta


class VerifiedRGBDRecovery:
    def __init__(self, K, mount, proposer, max_references=12):
        self.K, self.mount, self.proposer = K.copy(), mount.copy(), proposer
        self.references = deque(maxlen=max_references)
        self.pending = None
        self.events = []
        self.counter = 0
        self.attempts = 0

    def remember(self, rgb, depth, pose):
        self.counter += 1
        if not self.references or self.counter % 10 == 0:
            clean = rgb.copy(); clean[depth <= 0] = 0
            self.references.append((clean, depth.copy(), pose.copy()))

    def cloud(self, depth):
        h, w = depth.shape
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])
        cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth.astype(np.float32)), intr, depth_scale=1., depth_trunc=6., stride=3)
        cloud = cloud.voxel_down_sample(.035)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.12, max_nn=30))
        return cloud

    def verify(self, ref_rgb, ref_depth, rgb, depth, delta):
        """Refine a proposal using observed depth, then independently test RGB/depth."""
        if delta.shape != (4,4) or not np.isfinite(delta).all() or not np.allclose(delta[3],[0,0,0,1]):
            return None, {'reason': 'invalid pose proposal'}
        target, source = self.cloud(ref_depth), self.cloud(depth)
        if min(len(target.points), len(source.points)) < 200:
            return None, {'reason': 'insufficient measured geometry'}
        result = o3d.pipelines.registration.registration_icp(source, target, .15, delta,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        if not np.isfinite(result.transformation).all():
            return None, {'reason': 'nonfinite depth alignment'}
        # Enforce known planar base kinematics in the relative base frame.
        transform = self.mount@result.transformation@np.linalg.inv(self.mount)
        yaw = np.arctan2(transform[1, 0], transform[0, 0]); c, s = np.cos(yaw), np.sin(yaw)
        transform[:3, :3] = [[c,-s,0],[s,c,0],[0,0,1]]; transform[2, 3] = 0
        transform = np.linalg.inv(self.mount)@transform@self.mount
        p = np.asarray(source.points); q = p@transform[:3, :3].T+transform[:3, 3]
        distances, nearest = cKDTree(np.asarray(target.points)).query(q)
        # A turn changes the visible frustum. Points outside the reference view
        # cannot count as alignment failures; require adequate overlapping area.
        h,w=ref_depth.shape;safe=np.maximum(q[:,2],1e-6)
        u=np.rint(q[:,0]*self.K[0,0]/safe+self.K[0,2]).astype(int)
        v=np.rint(q[:,1]*self.K[1,1]/safe+self.K[1,2]).astype(int)
        overlap=(q[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
        measured=ref_depth[np.clip(v,0,h-1),np.clip(u,0,w-1)]
        overlap &= (measured>0)&(q[:,2]<measured+.1)
        valid = (distances < .06)&overlap
        coverage=float(overlap.mean())
        fitness = float(valid.sum()/max(1,overlap.sum()))
        rmse = float(np.sqrt(np.mean(distances[valid]**2))) if valid.any() else float('inf')
        # Reject underconstrained alignments (e.g. sliding along a blank wall).
        normals = np.asarray(target.normals)[nearest[valid]]@self.mount[:3, :3].T
        qb = q[valid]@self.mount[:3, :3].T+self.mount[:3, 3]
        J = np.column_stack([normals[:, 0], normals[:, 1], -normals[:, 0]*qb[:, 1]+normals[:, 1]*qb[:, 0]])
        H = J.T@J/max(1, len(J)); diag = np.sqrt(np.maximum(np.diag(H), 1e-12))
        eig = np.linalg.eigvalsh(H/(diag[:, None]*diag[None, :]))
        # Projective verification compares measured depth AND actual image colors.
        checks = []
        for src_d, dst_d, src_rgb, dst_rgb, T in [(depth,ref_depth,rgb,ref_rgb,transform), (ref_depth,depth,ref_rgb,rgb,np.linalg.inv(transform))]:
            h,w = src_d.shape; vv,uu = np.mgrid[0:h:4,0:w:4]; z=src_d[::4,::4].ravel()
            xyz=np.column_stack([(uu.ravel()-self.K[0,2])*z/self.K[0,0],(vv.ravel()-self.K[1,2])*z/self.K[1,1],z])
            moved=xyz@T[:3,:3].T+T[:3,3]; safe=np.maximum(moved[:,2],1e-6)
            u=np.rint(moved[:,0]*self.K[0,0]/safe+self.K[0,2]).astype(int); v=np.rint(moved[:,1]*self.K[1,1]/safe+self.K[1,2]).astype(int)
            inside=(z>0)&(moved[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
            u=np.clip(u,0,w-1);v=np.clip(v,0,h-1)
            observed=dst_d[v,u]; available=inside&(observed>0)
            good=available&(np.abs(observed-moved[:,2])<.06)
            color=np.mean(np.abs(src_rgb[::4,::4].reshape(-1,3).astype(float)-dst_rgb[v,u].astype(float)),axis=1)/255
            checks.append((float(good.sum()/max(1,available.sum())),float(np.mean(color[good]<.12)) if good.any() else 0.,int(good.sum())))
        info={'fitness':fitness,'coverage':coverage,'rmse':rmse,'observability':float(eig[0]),'projective':checks}
        if coverage < .2 or fitness < .65 or rmse > .035 or eig[0] < .01 or min(x[0] for x in checks)<.65 or min(x[1] for x in checks)<.65 or min(x[2] for x in checks)<500:
            return None, dict(info, reason='geometric or photometric verification rejected')
        return transform, info

    def recover(self, rgb, depth):
        self.attempts += 1
        if not self.references:
            return None, 0., float('inf'), 'recovery has no reference'
        if self.pending is None and self.attempts % 5 != 1:
            return None, 0., float('inf'), 'recovery awaiting next proposal'
        clean=rgb.copy();clean[depth<=0]=0
        for index in dict.fromkeys([len(self.references)-1, max(0,len(self.references)-3)]):
            ref_rgb,ref_depth,ref_pose=self.references[index]
            start=time.monotonic()
            delta=self.proposer(ref_rgb,clean,ref_depth)
            if delta is None:continue
            proposal_seconds=time.monotonic()-start
            start=time.monotonic()
            refined,info=self.verify(ref_rgb,ref_depth,clean,depth,delta)
            info.update(proposal_seconds=proposal_seconds,verification_seconds=time.monotonic()-start)
            self.events.append(dict(attempt=self.attempts,reference=index,**info))
            if refined is None:continue
            pose=planar(ref_pose@refined,self.mount)
            if self.pending is not None:
                change=np.linalg.inv(self.pending)@pose
                angle=np.arccos(np.clip((np.trace(change[:3,:3])-1)/2,-1,1))
                if np.linalg.norm(change[:3,3])<.15 and angle<.2:
                    self.pending=None
                    return pose,info['fitness'],info['rmse'],'VGGT recovery verified by measured RGB-D'
            self.pending=pose
            return None,info['fitness'],info['rmse'],'VGGT recovery awaiting second observation'
        self.pending=None
        return None,0.,float('inf'),'VGGT recovery rejected'
