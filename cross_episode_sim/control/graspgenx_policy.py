"""GraspGen-X pick proposals for the Robotiq 2F-85 from an observed object cloud.

Input is the segmented, sensed target surface in a gravity-aligned metric frame.
No simulator object pose or mesh is used. Placement stays with M2T2.
"""
import os
from pathlib import Path
import numpy as np

GRIPPER='robotiq_2f_85'
# GraspGen-X canonical frame: origin at the 2F-85 base link, +Z approach, +X closing.
# Our tool frame (gripper/grasp_site): +Z approach, +Y closing, 0.155 m ahead of the
# simulated base link. Measured pad centres: 0.140 m (simulated gripper) versus the
# 0.130 m sweep-volume centre GraspGen-X conditions on, so the pad centres are
# aligned by placing our base link 0.010 m behind the predicted one.
PAD_CENTRE_SHIFT=.010
TOOL_FROM_GRASP=np.array([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,1.,.155-PAD_CENTRE_SHIFT],[0.,0.,0.,1.]])
PAD_CENTRE_IN_TOOL=np.array([0.,0.,.140-.155])
M2T2_CONTACT_DEPTH=.1034


def graspgenx_to_tcp(pose):
    return np.asarray(pose,float)@TOOL_FROM_GRASP


def tcp_to_m2t2_root(tcp):
    """Inverse of m2t2_policy.grasp_to_tcp, so existing consumers stay unchanged."""
    out=np.asarray(tcp,float).copy()
    out[:3,:3]=out[:3,:3]@np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    out[:3,3]=out[:3,3]-M2T2_CONTACT_DEPTH*out[:3,2]
    return out


def proposal_rows(poses,scores,tags,object_points,sampling_pass=0):
    """Rows in the schema the pick executor already consumes."""
    from scipy.spatial import cKDTree
    tree=cKDTree(np.asarray(object_points,float));rows=[]
    for index,(pose,score,tag) in enumerate(zip(poses,scores,tags)):
        tcp=graspgenx_to_tcp(pose)
        centre=tcp[:3,3]+tcp[:3,:3]@PAD_CENTRE_IN_TOOL
        contact=tree.data[tree.query(centre)[1]]
        rows.append(dict(group=sampling_pass*10000+index,sampling_pass=sampling_pass,
            confidence=float(score),contact=contact.tolist(),tcp=tcp.tolist(),
            grasp=tcp_to_m2t2_root(tcp).tolist(),graspgenx_pose=np.asarray(pose,float).tolist(),
            proposal_source='graspgenx_'+str(tag)))
    return rows


def endpoint_proposals(client,surface,passes=2,threshold=.7,minimum=20,fallback_passes=0,fallback_threshold=.3):
    """Proposal rows for one segmented target surface from a graspgenx endpoint.

    M2T2 returns only grasps above its own confidence mask; GraspGen-X scores
    every sample. Keep its documented 0.7 threshold, or the best `minimum` if
    fewer pass. Shared by the vessel and generic-object pick proposal scripts.

    With `fallback_passes` (total seeds sampled) the confident rows come back
    unchanged and marked fallback=False, followed by every other sample scoring
    at least `fallback_threshold`, marked fallback=True. A narrow object that
    fits between the open fingers gets its confident grasps low on the side,
    where the support blocks the hand, while its rim and upper-side grasps
    score below the threshold; the executor decides when to try those.
    """
    rows=[]
    for sample in range(max(passes,fallback_passes)):
        out=client.call(object_points=surface,seed=sample)
        rows.extend(proposal_rows(out['grasps'],out['confidence'],out['tags'],surface,sample))
    rows.sort(key=lambda row:-row['confidence'])
    first=[row for row in rows if row['sampling_pass']<passes]
    confident=[row for row in first if row['confidence']>=threshold]
    primary=confident if len(confident)>=minimum else first[:minimum]
    if not fallback_passes:return primary
    kept={row['group'] for row in primary}
    return [dict(row,fallback=False) for row in primary]+[dict(row,fallback=True) for row in rows
        if row['group'] not in kept and row['confidence']>=fallback_threshold]


def slide_along_handle(rows,surface,target,elongation=2.):
    """Move grasps along a straight handle bar so they sit at the pointed spot.

    GraspGen-X spreads its grasps over the whole bar while the executor keeps
    only grasps within 3 cm of the pointed handle spot. On a bar (first
    principal axis at least `elongation` times the second) each grasp is
    translated along the bar axis to the target's position on that axis,
    clipped to the observed bar ends. Orientation is unchanged and the contact
    is recomputed on the observed surface. Other handles are returned as is.
    """
    from scipy.spatial import cKDTree
    surface=np.asarray(surface,float);target=np.asarray(target,float)
    _,singular,axes=np.linalg.svd(surface-surface.mean(0),full_matrices=False)
    if len(singular)<2 or singular[0]<elongation*max(singular[1],1e-9):return rows
    axis=axes[0];along=(surface-surface.mean(0))@axis
    goal=float(np.clip((target-surface.mean(0))@axis,along.min(),along.max()))
    tree=cKDTree(surface);out=[]
    for row in rows:
        tcp=np.asarray(row['tcp'],float).copy()
        centre=tcp[:3,3]+tcp[:3,:3]@PAD_CENTRE_IN_TOOL
        shift=goal-float((centre-surface.mean(0))@axis)
        tcp[:3,3]+=shift*axis;centre=centre+shift*axis
        out.append(dict(row,tcp=tcp.tolist(),grasp=tcp_to_m2t2_root(tcp).tolist(),
            contact=tree.data[tree.query(centre)[1]].tolist(),handle_slide=float(shift)))
    return out


def handle_proposals(client,surface,normal,passes=3,facing=-.5,minimum_points=100,target=None):
    """Grasp rows for a protruding fixture handle from a graspgenx endpoint.

    GraspGen-X is object-centric: given the handle together with its backing
    panel it scores almost nothing on the handle, so only the separated handle
    surface is sent. It then samples grasps from every side, including through
    the door behind the handle; rows whose approach does not face the observed
    panel (approach . outward normal > `facing`) are dropped. Rows carry the
    tcp/confidence/contact fields the handle executor already consumes, most
    confident first. Panel alignment, centring, collision-checked approach and
    the encoder grasp check downstream are unchanged. With `target` (the
    pointed handle spot) grasps on a straight bar are slid to it.
    """
    surface=np.asarray(surface,np.float32);normal=np.asarray(normal,float)
    if surface.ndim!=2 or surface.shape[1]!=3 or len(surface)<12:return []
    cloud=surface
    if len(cloud)<minimum_points:
        # A thin handle seen from afar can have few depth points. Repeat the
        # measured points; no surface is invented.
        cloud=np.concatenate([cloud]*int(np.ceil(minimum_points/len(cloud))))
        # The service drops exact duplicates ("requires at least 100 object
        # points; received 56", fix75), so repeats get 0.2 mm of jitter, far
        # below depth noise.
        cloud=(cloud+np.random.default_rng(0).normal(0.,2e-4,cloud.shape)).astype(np.float32)
        cloud[:len(surface)]=surface
    rows=[]
    for sample in range(passes):
        try:out=client.call(object_points=cloud,seed=sample)
        except RuntimeError as error:
            # The service prunes points before sampling, so a thin handle can
            # fall under its 100-point minimum even when more were sent
            # ("received 79", fix78; "received 56", fix75). That is "no
            # GraspGen-X proposal", and the caller's M2T2 fallback runs.
            if 'requires at least' not in str(error):raise
            print('GRASPGENX_HANDLE_TOO_SPARSE',str(error)[:160],flush=True)
            break
        rows.extend(proposal_rows(out['grasps'],out['confidence'],out['tags'],surface,sample))
    rows=[row for row in rows if np.asarray(row['tcp'])[:3,2]@normal<=facing]
    if target is not None:
        # The executor keeps grasps within 3 cm of its handle target: the
        # pointed spot, or the middle of the observed bar when it recentres
        # (microwave, fridge, drawers). Offer each grasp at both; a pointed
        # spot 11.7 cm from the middle dropped all 124 microwave grasps.
        middle=surface[np.argmin(np.linalg.norm(surface-np.median(surface,axis=0),axis=1))]
        rows=[dict(row,handle_slide_target=name) for name,spot in (('pointed',target),('middle',middle))
              for row in slide_along_handle(rows,surface,spot)]
    rows.sort(key=lambda row:-row['confidence'])
    return rows


def split_plausible(rows,envelope,centers,half,rotations,shoulder,margin=.01):
    """Split pick rows into (plausible, implausible) wrist poses, order kept.

    Implausible means a corner of the open-gripper `envelope` (tool frame) at
    the grasp pose lies within `margin` of an observed obstacle box, or the
    wrist would enter from the far side of the grasp as seen from the shoulder.
    A screen for ordering and for bounding the fallback tier only: plausible
    rows still have to pass IK, swept collision, linkage and physical grip checks.
    """
    envelope=np.asarray(envelope,float);centers=np.asarray(centers,float).reshape(-1,3)
    half=np.asarray(half,float).reshape(-1,3)+margin;rotations=np.asarray(rotations,float).reshape(-1,3,3)
    def implausible(row):
        tcp=np.asarray(row['tcp'],float);reach=tcp[:2,3]-np.asarray(shoulder,float)[:2]
        if tcp[:2,2]@reach<-.5*np.linalg.norm(reach):return True
        corners=envelope@tcp[:3,:3].T+tcp[:3,3]
        local=np.einsum('pbi,bij->pbj',corners[:,None]-centers[None],rotations)
        return bool((abs(local)<=half[None]).all(-1).any())
    flags=[implausible(row) for row in rows]
    return [row for row,flag in zip(rows,flags) if not flag],[row for row,flag in zip(rows,flags) if flag]


class GraspGenXPolicy:
    def __init__(self, repository='external/GraspGenX', planner='graspmoe', num_grasps=200):
        from graspgenx import get_checkpoints_version_dir
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg
        root=Path(get_checkpoints_version_dir())
        if not (root/'gen'/'config.yaml').is_file():
            root=root.parent/'GraspGenXModel'/root.name
        cfg=load_model_cfg(os.path.join(root,'gen'),os.path.join(root,'dis'),None,None)
        assets=Path(repository).resolve()/'assets'
        self.sampler=GraspGenXSampler(cfg,GRIPPER,assets_dir=str(assets),diffusion_grasps=num_grasps)
        self.planner=planner;self.num_grasps=num_grasps

    def predict(self, object_points, seed=0):
        import torch
        from graspgenx.samplers import run_planner_on_object
        points=np.asarray(object_points,np.float32)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<32 or not np.isfinite(points).all():
            raise ValueError('Need a finite observed object cloud of at least 32 points')
        torch.manual_seed(int(seed));np.random.seed(int(seed))
        centre=points.mean(0)
        grasps,scores,tags,_=run_planner_on_object(points-centre,self.sampler,planner=self.planner,
            grasp_threshold=-1.,num_grasps=self.num_grasps)
        grasps=np.asarray(grasps,float).reshape(-1,4,4).copy()
        grasps[:,:3,3]+=centre;grasps[:,3]=[0.,0.,0.,1.]
        return dict(grasps=grasps,confidence=np.asarray(scores,float),tags=list(tags))
