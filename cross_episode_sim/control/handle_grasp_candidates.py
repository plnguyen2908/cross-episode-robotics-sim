"""Refine and deduplicate learned handle poses before collision planning."""
import numpy as np
from cross_episode_sim.perception.handle_surface import align_grasp_to_panel,center_grasp_on_handle


def refine_handle_candidates(rows,surface,normal,*,align_panel=False,center_handle=False,limit=None):
    result=[]
    for row in rows:
        pose=np.asarray(row['tcp'],float).copy()
        try:
            if align_panel:pose=align_grasp_to_panel(pose,normal)
            if center_handle:pose=center_grasp_on_handle(pose,surface)
        except ValueError:continue
        # Alignment/centering can collapse distinct M2T2 proposals to one pose.
        if any(np.linalg.norm(pose[:3,3]-np.asarray(r['tcp'])[:3,3])<.005 and
               np.linalg.norm(pose[:3,:3]-np.asarray(r['tcp'])[:3,:3])<.15 for r in result):continue
        result.append(dict(row,tcp=pose.tolist(),model_tcp=row.get('model_tcp',row['tcp']),
                           raw_model_tcp=row.get('raw_model_tcp',row['tcp']),
                           orientation_refinement='observed panel normal' if align_panel else row.get('orientation_refinement','unchanged')))
        if limit is not None and len(result)>=limit:break
    return result


def handle_depth_variants(row,backoff=0.,*,allow_shallow_fallback=False):
    """Try the requested grasp first, then a 5 mm shallower contact if enabled.

    This changes the proposed pose, never world geometry or collision margins.
    Every variant still needs a validated approach and insertion path.
    """
    offsets=[backoff]
    if allow_shallow_fallback and backoff==0.:offsets.append(.005)
    for offset in offsets:
        pose=np.asarray(row['tcp'],float).copy()
        pose[:3,3]-=offset*pose[:3,2]
        yield dict(row,executed_tcp=pose.tolist(),contact_backoff=offset)
