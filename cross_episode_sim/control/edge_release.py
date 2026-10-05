"""Supported near-edge targets for a flat object held from its exposed side."""
import numpy as np


def flat_edge_targets(local_vertices, rotation, preferred, low, high, outward,
                      local_com, margin=.03, max_shift=.10):
    """Keep the COM on the support while giving the side grasp overhang space.

    This is geometric candidate generation, not a collision-free arm plan.
    The caller must check other objects and plan the loaded arm/release path.
    """
    rotated=np.asarray(local_vertices)@np.asarray(rotation).T
    normal=np.asarray(outward,dtype=float)
    axis=int(np.argmax(np.abs(normal)));tangent=1-axis
    if not np.isclose(abs(normal[axis]),1.) or not np.isclose(normal[tangent],0.):
        raise ValueError('Expected an axis-aligned support edge')
    low,high=np.asarray(low),np.asarray(high)
    edge=high[axis] if normal[axis]>0 else low[axis]
    boundary=edge*normal[axis]
    projection=rotated[:,:2]@normal
    extent=float(np.ptp(projection))
    if np.ptp(rotated[:,tangent])+.02>high[tangent]-low[tangent]:
        return []
    com=np.asarray(rotation)@np.asarray(local_com)
    results=[]
    # Start with a small overhang; retain at least 65% of projected support.
    for overhang in (.02,.04,0.,.06):
        if overhang>.35*extent:continue
        point=np.asarray(preferred,dtype=float).copy()
        point[axis]=normal[axis]*(boundary+overhang-projection.max())
        point[tangent]=np.clip(point[tangent],low[tangent]+.01-rotated[:,tangent].min(),
                               high[tangent]-.01-rotated[:,tangent].max())
        point[2]=high[2]-rotated[:,2].min()
        if np.linalg.norm(point[:2]-np.asarray(preferred)[:2])>max_shift:continue
        world_com=point+com
        if np.any(world_com[:2]<low[:2]+margin) or np.any(world_com[:2]>high[:2]-margin):continue
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=point
        results.append(pose)
    return results
