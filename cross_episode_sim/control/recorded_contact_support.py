"""Restore recorded sensor floor coverage for standalone contact continuations."""
import json
from pathlib import Path
import h5py
import numpy as np


def recorded_contact_support(grasp_run,*,observed_root=None):
    """Use the last accepted map/estimated pose before this recorded grasp.

    This is development-harness context, not fixture truth or a reference route.
    Live execution already carries this region in its episode runtime.
    """
    grasp_run=Path(grasp_run)
    with h5py.File(grasp_run/'state_recording/states.h5') as states:
        start=float(states['times'][0])
    for parent in ([Path(observed_root)] if observed_root is not None else grasp_run.parents):
        record=parent/'state_recording'
        diagnostics=record/'diagnostics.jsonl'
        if not diagnostics.exists():continue
        entries=[json.loads(line) for line in diagnostics.read_text().splitlines()]
        eligible=[e for e in entries if e['timestamp']<=start and e.get('accepted')
                  and e.get('pose') is not None and e.get('map')]
        if not eligible:continue
        event=max(eligible,key=lambda e:e['timestamp'])
        saved=np.load(record/event['map'])
        rows,cols=np.where(saved['floor']>=int(saved['minimum_floor_views']))
        resolution=float(saved['resolution'])
        xy=saved['origin']+np.c_[cols,rows]*resolution
        from shapely.geometry import box
        from shapely.ops import unary_union
        from .contact_retreat import transform_support_region
        region=unary_union([box(x,y,x+resolution,y+resolution) for x,y in xy])
        result=transform_support_region(region,np.linalg.inv(np.asarray(event['pose'])))
        return result,dict(source=str(record/event['map']),timestamp=event['timestamp'],
                           frame='recorded estimated base pose before grasp',floor_cells=len(xy))
    return None,None
