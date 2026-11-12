"""Small gravity-release proposals from observed support and payload geometry."""
import numpy as np

from .observed_placement import supported_interior, unoccupied_contacts


def table_drop_candidates(surface, scene, held, tcp, *, clearance=.05):
    """Keep the held orientation; put its measured bottom above free support.

    These are geometric proposals, not M2T2 outputs. The caller must plan the
    arm and payload path and score physical support after the release.
    """
    surface=np.asarray(surface,float).reshape(-1,3)
    held=np.asarray(held,float).reshape(-1,3)
    tcp=np.asarray(tcp,float)
    if len(surface)<3 or len(held)<4:return []
    center=(held[:,:2].min(0)+held[:,:2].max(0))/2
    radius=float(np.linalg.norm(held[:,:2]-center,axis=1).max()+.03)
    _,indices=np.unique(np.floor(surface/.04).astype(int),axis=0,return_index=True)
    contacts=surface[indices]
    contacts=contacts[supported_interior(surface,contacts,radius)]
    contacts=contacts[unoccupied_contacts(contacts,scene,radius,float(np.ptp(held[:,2])+clearance))]
    contacts=contacts[np.argsort(np.linalg.norm(contacts[:,:2]-center,axis=1))][:24]
    result=[]
    for contact in contacts:
        goal=tcp.copy()
        goal[:3,3]+=np.r_[contact[:2]-center,contact[2]+clearance-held[:,2].min()]
        result.append(dict(group=2000,confidence=0.,contact=contact.tolist(),tcp=goal.tolist(),
                           release_clearance_m=clearance,proposal_source='observed_table_small_drop',
                           learned_proposal=False))
    return result
