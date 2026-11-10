"""Short contact approach from calibrated FK and observed collision checks."""
from cross_episode_sim.control.contact_retreat import tool_pose
import logging
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from cross_episode_sim.control.observed_grasp_planner import smooth_joint_path


def plan_contact_line(planner,joints,target,spacing=.005):
    q=np.asarray(joints,float).copy();geometry=planner.geometry
    start=tool_pose(geometry,q)
    target=np.asarray(target,float)
    distance=np.linalg.norm(target[:3,3]-start[:3,3])
    angle=Rotation.from_matrix(target[:3,:3]@start[:3,:3].T).as_rotvec()
    if distance>.15 or np.linalg.norm(angle)>.2:
        raise RuntimeError('Contact-line fallback is limited to a short aligned approach')
    lo,hi=geometry.limits.T;segments=[]
    for fraction in np.linspace(0,1,max(2,int(np.ceil(distance/spacing))+1))[1:]:
        position=start[:3,3]+fraction*(target[:3,3]-start[:3,3])
        rotation=Rotation.from_rotvec(fraction*angle).as_matrix()@start[:3,:3]
        def residual(x):
            T=tool_pose(geometry,x)
            return np.r_[T[:3,3]-position,.2*Rotation.from_matrix(rotation@T[:3,:3].T).as_rotvec()]
        from .local_ik import local_ik
        from types import SimpleNamespace
        waypoint=np.eye(4);waypoint[:3,3]=position;waypoint[:3,:3]=rotation
        candidate=local_ik(geometry,q,waypoint)
        fit=(SimpleNamespace(x=candidate,status=1) if candidate is not None else
             least_squares(residual,np.clip(q,lo+.001,hi-.001),bounds=(lo+.001,hi-.001),max_nfev=80))
        error=residual(fit.x)
        if (np.linalg.norm(error[:3])>.002 or np.linalg.norm(error[3:])>.005) and fit.status==0:
            # An exhausted optimizer budget is not geometric unreachability.
            # Continue from its best iterate only on this uncommon failure;
            # keep the same pose tolerance, limits, and subsequent swept checks.
            logging.getLogger(__name__).warning('Contact-line IK budget exhausted; continuing with unchanged tolerances and collision checks')
            fit=least_squares(residual,fit.x,bounds=(lo+.001,hi-.001),max_nfev=220)
            error=residual(fit.x)
        if np.linalg.norm(error[:3])>.002 or np.linalg.norm(error[3:])>.005:
            raise RuntimeError('Contact-line IK unreachable')
        path=smooth_joint_path(q,fit.x)
        dense=np.linspace(q,fit.x,max(2,int(np.ceil(np.abs(fit.x-q).sum()/.001))+1))
        if not planner.clear(path) or not planner.clear(dense):
            raise RuntimeError('Contact-line sweep blocked: '+str(getattr(planner,'last_rejection',None)))
        segments.append(path if not segments else path[1:]);q=fit.x
    return np.concatenate(segments)
