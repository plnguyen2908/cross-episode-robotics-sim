"""Private test-fixture restoration of prior observed mapping, not scene truth."""
import json
import numpy as np


def restore_start_context(grid, recording, timestamp):
    rows=[json.loads(line) for line in (recording/'diagnostics.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r.get('accepted') and r.get('pose') is not None and r.get('map') and r['timestamp']<=timestamp+1e-6]
    if not rows:raise ValueError('No observed context at/before selected reset frame')
    row=rows[-1];anchor=np.asarray(row['pose'])
    with np.load(recording/row['map']) as old:
        rr,cc=np.indices(grid.floor_views.shape)
        xy=grid.cell_to_world(np.c_[cc.ravel(),rr.ravel()]);counts=[]
        half=grid.config.resolution/2
        for offset in ((-half,-half),(-half,half),(half,-half),(half,half)):
            world=(xy+offset)@anchor[:2,:2].T+anchor[:2,3]
            cells=np.floor((world-old['origin'])/float(old['resolution'])).astype(int)
            valid=(cells[:,0]>=0)&(cells[:,1]>=0)&(cells[:,0]<old['floor'].shape[1])&(cells[:,1]<old['floor'].shape[0])
            values=np.zeros(len(xy),np.uint16)
            values[valid]=old['floor'][cells[valid,1],cells[valid,0]];counts.append(values)
        grid.floor_views[:]=np.maximum(grid.floor_views,np.min(counts,axis=0).reshape(grid.floor_views.shape))
    # Keep only already-traveled footprints; never consume later trajectory.
    inverse=np.linalg.inv(anchor);poses=[]
    # Offline component fixtures may preserve the earlier SLAM trajectory
    # separately from their final floor grid. These are occupied footprints,
    # not inferred free space or simulator base poses. Never read future poses.
    history=recording/'support_history.json'
    if history.exists():
        extra=json.loads(history.read_text())
        rows=rows+[r for r in extra if r.get('accepted') and r.get('pose') is not None
                   and r['timestamp']<=timestamp+1e-6]
    for r in rows:
        T=inverse@np.asarray(r['pose']);poses.append([T[0,3],T[1,3],np.arctan2(T[1,0],T[0,0])])
    grid.inherited_support_poses=np.asarray(poses)
    return anchor
