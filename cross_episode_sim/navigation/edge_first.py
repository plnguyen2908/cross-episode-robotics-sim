"""Among several instances on one support, take the one nearest its front edge."""
import numpy as np
from .docking_candidates import observed_surface

GROUP_RADIUS=.60  # instances this close to the current one share its support region
GROUP_HEIGHT=.10
EDGE_MARGIN=.05   # smaller depth differences are measurement noise: keep the current one


def edge_depth(packet,T_map_base,position):
    """Metres from position to the observed front edge of its support, or None.

    Calibrated onboard RGB-D only. None means the edge beside this position
    was not verified from the present view.
    """
    position=np.asarray(position,float)
    try:point,_,normal=observed_surface(packet,T_map_base,position)
    except ValueError:return None
    return float((point[:2]-position[:2])@normal)


def edge_first(current,candidates,packet,T_map_base,depth=edge_depth):
    """Return (choice, measured instances).

    An instance standing behind another needs a longer reach and has the front
    one in the arm's way. Change from current only when another instance of
    its group measurably sits nearer the edge the base docks against.
    """
    group=[c for c in candidates if c is not current
           and np.linalg.norm(c.position[:2]-current.position[:2])<=GROUP_RADIUS
           and abs(c.position[2]-current.position[2])<=GROUP_HEIGHT]
    if not group:return current,[]
    rows=[(t,depth(packet,T_map_base,t.position)) for t in [current]+group]
    measured=[dict(track_id=getattr(t,'track_id',None),position=np.asarray(t.position,float).tolist(),edge_depth_m=d) for t,d in rows]
    known=[(d,index) for index,(_,d) in enumerate(rows) if d is not None]
    if not known:return current,measured
    best,index=min(known)
    held=rows[0][1]
    if index and (held is None or best+EDGE_MARGIN<held):return rows[index][0],measured
    return current,measured
