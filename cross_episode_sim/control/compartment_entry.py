"""Preflight loaded alignment and insertion without stepping the live world."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.control.collision_probe import forward_collision


def plan_empty_withdrawal(c, waypoints):
    """Preflight complete empty-hand exit paths, including shelf-clearance lifts."""
    if getattr(c, 'holding_loaf', False) or getattr(c, 'attached', False):
        raise RuntimeError('Compartment withdrawal requires a released object')
    live = c.data
    addresses = [c.model.jnt_qposadr[c.model.joint('robot_0/'+name).id]
                 for name in c.planner.names]
    errors = []
    for lift in (0., .02, .04, .06):
        probe = mujoco.MjData(c.model)
        mujoco.mj_copyData(probe, c.model, live)
        try:
            c.data = probe
            goals = []
            if lift:
                first = c.tcp().copy(); first[2, 3] += lift
                goals.append(first)
            for waypoint in waypoints:
                goal = waypoint.copy(); goal[2, 3] += lift
                goals.append(goal)
            result = []
            for goal in goals:
                path = c.plan_contact_path('withdraw from released table object', goal)
                result.append((goal, path))
                if len(path):
                    probe.qpos[addresses] = path[-1]
                    mujoco.mj_forward(c.model, probe)
            return result
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            c.data = live
    raise RuntimeError('No native-mesh-clear empty compartment withdrawal: '+str(errors))


def plan_entry(controller, front, destination, via=()):
    c=controller
    addresses=[c.model.jnt_qposadr[c.model.joint('robot_0/'+name).id]
               for name in c.planner.names]
    positions=c.data.qpos[addresses].copy()
    goal=list(front[:3,3]-[0,0,c.embodiment.planner_tool_offset()])+list(
        Rotation.from_matrix(front[:3,:3]).as_quat(scalar_first=True))
    cache=getattr(c,'_compartment_alignment_cache',{})
    key=(c.object_name,tuple(c.data.qpos),tuple(front.ravel()))
    cached=cache.get(key)
    if isinstance(cached,str):
        raise RuntimeError(cached)
    if cached is None:
        try:
            path=c.planner.plan(positions.tolist(),goal)
        except RuntimeError as exc:
            # Conservative fixture boxes can close off an open compartment.
            # A joint interpolation is eligible only after the same dense
            # native-mesh validation below (including the carried object).
            basket = getattr(c, 'object_info', {}).get(c.object_name, {}).get('category') == 'basket'
            attempts = [(2., True), (6., True)]
            if basket:
                # Attached-basket spheres can overlap the arm despite clear
                # native geometry. An unconstrained IK seed is only eligible
                # after the dense robot/payload/contents checks below.
                attempts.append((6., False))
            for index, (radius, conservative_self_check) in enumerate(attempts):
                try:
                    end=np.asarray(c.nearby_ik(front,positions,trust_radius=radius,
                                              preserve_self_clearance=conservative_self_check))
                    if not conservative_self_check:
                        c.record(basket_alignment_requires_native_self_check=True)
                    break
                except RuntimeError as ik_error:
                    if index == len(attempts)-1:
                        cache[key]=str(ik_error)
                        raise
            count=max(6,int(np.ceil(np.max(np.abs(end-positions))/.01)))
            path=[positions+f*(end-positions) for f in np.linspace(0,1,count+1)[1:]]
            c.record(compartment_alignment_method='native-mesh checked joint interpolation',
                     conservative_planner_error=str(exc))
        cache[key]=path
    else:
        path=cached
    if not len(path):
        raise RuntimeError('cuRobo failed to plan compartment alignment')
    live=c.data
    probe=mujoco.MjData(c.model)
    mujoco.mj_copyData(probe,c.model,live)
    relative=np.linalg.inv(c.tcp()) @ c.bread_pose()
    obj_address=c.model.jnt_qposadr[c.model.joint(c.object_joint).id]
    previous=positions
    try:
        c.data=probe
        for end in path:
            end=np.asarray(end)
            count=max(1,int(np.ceil(np.max(np.abs(end-previous))/.01)))
            for fraction in np.linspace(0,1,count+1)[1:]:
                probe.qpos[addresses]=previous+fraction*(end-previous)
                mujoco.mj_forward(c.model,probe)
                obj=c.tcp() @ relative
                probe.qpos[obj_address:obj_address+3]=obj[:3,3]
                probe.qpos[obj_address+3:obj_address+7]=Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
                forward_collision(c.model,probe)
                if hasattr(c,'project_payload_contents'):
                    c.project_payload_contents(probe,reference=live)
                if c.navigation_penetration(probe,True)>.001 or c.robot_self_penetration(probe)>.0005:
                    raise RuntimeError('Actual-mesh collision in contact move: compartment alignment '+
                                       str(c.contact_pair_detail(probe,True)))
            previous=end
        if via or not np.allclose(front,destination,atol=1e-10,rtol=0.):
            for waypoint in (*via,destination):
                insertion=c.plan_contact_path('insert held object into compartment',waypoint)
                if len(insertion):
                    probe.qpos[addresses]=insertion[-1]
                    mujoco.mj_forward(c.model,probe)
                    obj=c.tcp() @ relative
                    probe.qpos[obj_address:obj_address+3]=obj[:3,3]
                    probe.qpos[obj_address+3:obj_address+7]=Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)
                    forward_collision(c.model,probe)
    finally:
        c.data=live
    return path
