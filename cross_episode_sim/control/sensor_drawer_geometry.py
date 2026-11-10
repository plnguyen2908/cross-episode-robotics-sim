"""Occupancy-preserving observed collision boxes with fine robot clearance.

Resolution changes discretization only, never removes sensed obstacle points.
"""
import numpy as np

def merge_occupied(voxels):
 remaining=set(map(tuple,np.asarray(voxels,dtype=int)))
 for lo in sorted(remaining):
  if lo not in remaining:continue
  hi=list(lo)
  for axis in (0,1,2):
   while True:
    nxt=hi[axis]+1;ranges=[range(lo[j],hi[j]+1) if j!=axis else [nxt] for j in range(3)]
    if not all((x,y,z) in remaining for x in ranges[0] for y in ranges[1] for z in ranges[2]):break
    hi[axis]=nxt
  for x in range(lo[0],hi[0]+1):
   for y in range(lo[1],hi[1]+1):
    for z in range(lo[2],hi[2]+1):remaining.remove((x,y,z))
  yield np.array(lo),np.array(hi)


def adaptive_partitions(points, robot_boxes, near_distance=.12):
    near=np.zeros(len(points),bool)
    for box in robot_boxes:
        near |= np.linalg.norm(np.maximum(np.abs((points-box.center)@box.axes)-box.half,0),axis=1)<near_distance
    for pts,resolution in [(points[near],.002),(points[~near],.015)]:
        vox=np.unique(np.floor(pts/resolution).astype(int),axis=0)
        for lo,hi in merge_occupied(vox):
            yield lo,hi,resolution


def unique_voxels(voxels, *, return_index=False):
    """Exact lexicographic integer-row deduplication using scalar sorting.

    Packing preserves every occupied cell and the first occurrence index. Large
    coordinate spans use NumPy's general path to avoid integer overflow.
    """
    rows=np.asarray(voxels)
    if rows.ndim!=2 or rows.shape[1]!=3 or not np.issubdtype(rows.dtype,np.signedinteger):
        raise ValueError('Expected signed integer (N, 3) voxel coordinates')
    if not len(rows):
        return (rows.copy(),np.empty(0,dtype=np.intp)) if return_index else rows.copy()
    lo=rows.min(axis=0);hi=rows.max(axis=0)
    spans=[int(b)-int(a)+1 for a,b in zip(lo,hi)]
    if spans[0]*spans[1]*spans[2]>np.iinfo(np.int64).max:
        return np.unique(rows,axis=0,return_index=return_index)
    shifted=rows.astype(np.int64)-lo.astype(np.int64)
    packed=(shifted[:,0]*spans[1]+shifted[:,1])*spans[2]+shifted[:,2]
    codes,first=np.unique(packed,return_index=True)
    unique=rows[first]
    return (unique,first) if return_index else unique
