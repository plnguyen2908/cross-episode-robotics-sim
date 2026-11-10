"""Restore measured floor coverage for isolated recorded-state regressions.

Diagnostic snapshots contain the observed SLAM map, not simulator geometry.
Preserve coverage conservatively when reanchoring it into the test's local frame.
"""
import json
import numpy as np


def restore_recorded_floor(grid, recording, *, last=False):
    rows=[json.loads(line) for line in (recording/'diagnostics.jsonl').read_text().splitlines()]
    rows=[row for row in rows if row.get('accepted') and row.get('pose') is not None and row.get('map')]
    row=rows[-1 if last else 0]
    anchor=np.asarray(row['pose'])
    with np.load(recording/row['map']) as old:
        rr,cc=np.indices(grid.floor_views.shape)
        xy=grid.cell_to_world(np.c_[cc.ravel(),rr.ravel()]);counts=[]
        half=grid.config.resolution/2
        for offset in ((-half,-half),(-half,half),(half,-half),(half,half)):
            world=(xy+offset)@anchor[:2,:2].T+anchor[:2,3]
            cells=np.floor((world-old['origin'])/float(old['resolution'])).astype(int)
            valid=(cells[:,0]>=0)&(cells[:,1]>=0)&(cells[:,0]<old['floor'].shape[1])&(cells[:,1]<old['floor'].shape[0])
            values=np.zeros(len(xy),np.uint16)
            values[valid]=old['floor'][cells[valid,1],cells[valid,0]]
            counts.append(values)
        grid.floor_views[:]=np.maximum(grid.floor_views,np.min(counts,axis=0).reshape(grid.floor_views.shape))
    return anchor
