"""Measured receptacle geometry and placement candidates for a shared scene."""
import numpy as np


def raise_hand_above_rim(tcp, hand_vertices, candidate, rim_height, clearance=.008):
    """Minimum vertical release offset that keeps the hand above a basket rim."""
    local=(np.asarray(hand_vertices)-tcp[:3,3]) @ tcp[:3,:3]
    projected=local @ candidate[:3,:3].T+candidate[:3,3]
    result=candidate.copy()
    result[2,3]+=max(0.,float(rim_height+clearance-projected[:,2].min()))
    return result


def compartment_mouth(controller, role, tcp):
    """Keep the held object outside the front plane while centering in the aperture."""
    c=controller
    _,lo,hi=surface(c,role)
    outward=np.asarray(c.task_outward[role])
    axis=int(np.argmax(np.abs(outward)))
    current=c.bread_pose()
    local=(c.bread_vertices()-current[:3,3]) @ current[:3,:3]
    placed=tcp @ c.grasp_relative
    offsets=local @ placed[:3,:3].T+placed[:3,3]-tcp[:3,3]
    edge=(hi if outward[axis]>0 else lo)[axis]*outward[axis]
    required=edge-float(np.min(offsets[:,:2] @ outward))+.03
    mouth=tcp.copy()
    mouth[:2,3]+=outward*max(0.,required-float(mouth[:2,3] @ outward))
    return mouth


def compartment_front(controller, role, tcp):
    """Align outside the measured open door, accounting for the held footprint."""
    from cross_episode_sim.manipulation.edge_access import collision_vertices

    c=controller
    _,lo,hi=surface(c,role)
    outward=np.asarray(c.task_outward[role])
    axis=int(np.argmax(np.abs(outward)))
    front=tcp.copy()
    front[axis,3]=(hi if outward[axis]>0 else lo)[axis]+outward[axis]*.30
    binding=c.manifest.get('fixture_bindings',{}).get(role)
    if binding is None:return front
    door=collision_vertices(c.model,c.data,c.descendants(binding['body']))
    current=c.bread_pose()
    local=(c.bread_vertices()-current[:3,3]) @ current[:3,:3]
    placed=tcp @ c.grasp_relative
    offsets=local @ placed[:3,:3].T+placed[:3,3]-tcp[:3,3]
    required=float(np.max(door[:,:2] @ outward)-np.min(offsets[:,:2] @ outward)+.03)
    if 'joint' in binding:
        joint=c.model.joint(binding['joint'])
        tangent=1-axis
        payload_range=front[tangent,3]+np.array([offsets[:,tangent].min(),offsets[:,tangent].max()])
        if (abs(float(c.data.qpos[joint.qposadr[0]]))>np.radians(85)
                and (payload_range[1]<door[:,tangent].min()-.03
                     or payload_range[0]>door[:,tangent].max()+.03)):
            # With the door fully aside, enter at the selected shelf position.
            # A mandatory detour around the opposite edge can exceed arm reach
            # on a wide cabinet. Full robot/payload swept checks still follow.
            return compartment_mouth(c, role, tcp)
        # Approach around the free side of a partly opened door. A straight
        # insertion at the final shelf x can intersect its diagonal panel.
        tangent=1-axis
        anchor=c.data.xanchor[c.model.joint(binding['joint']).id,tangent]
        edge=lo[tangent]-.03 if anchor>(lo[tangent]+hi[tangent])/2 else hi[tangent]+.03
        front[tangent,3]=edge-(placed[tangent,3]-tcp[tangent,3])
        payload_range=front[tangent,3]+np.array([offsets[:,tangent].min(),offsets[:,tangent].max()])
        if (payload_range[1]<door[:,tangent].min()-.03
                or payload_range[0]>door[:,tangent].max()+.03):
            return front
    front[:2,3]+=outward*max(0.,required-float(front[:2,3] @ outward))
    return front


def surface(controller, role):
    spec = controller._surface_specs[role]
    gid = controller.model.geom(spec['floor_geom']).id
    corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    vertices = controller.data.geom_xpos[gid] + (
        controller.model.geom_aabb[gid,:3]+corners*controller.model.geom_aabb[gid,3:]
        ) @ controller.data.geom_xmat[gid].reshape(3,3).T
    return gid, vertices.min(0), vertices.max(0)


def footprint_on_surface(controller, gid, xy, radius):
    """Check the actual rotated slab, rather than its oversized world AABB."""
    center=controller.data.geom_xpos[gid]
    rotation=controller.data.geom_xmat[gid].reshape(3,3)
    corners=np.array([[xy[0]+sx*radius[0],xy[1]+sy*radius[1],center[2]]
                      for sx in (-1,1) for sy in (-1,1)])
    local=(corners-center) @ rotation
    aabb=controller.model.geom_aabb[gid]
    return bool(np.all(np.abs(local[:,:2]-aabb[:2])<=aabb[3:5]+1e-9))


def candidates(controller, role, obj):
    from scipy.spatial.transform import Rotation

    from cross_episode_sim.manipulation.edge_access import collision_vertices
    from cross_episode_sim.tasks.breakfast.scene import bounds
    gid, lo, hi = surface(controller, role)
    body=controller.data.body(obj)
    vertices=collision_vertices(controller.model,controller.data,controller.descendants(obj))
    local=(vertices-body.xpos) @ body.xmat.reshape(3,3)
    initial=controller.object_info[obj]['initial_quaternion']
    placed=local @ Rotation.from_quat(initial,scalar_first=True).as_matrix().T
    extent=placed.max(0)-placed.min(0)
    # Leave release/rocking clearance at cabinet edges instead of sampling
    # the same lip-adjacent row and rejecting every pose after the fact.
    radius = extent[:2]/2 + (.03 if role == 'kitchen_cabinet' else .012)
    spec = controller._surface_specs[role]
    if spec.get('burner_site'):
        xy = controller.data.site(spec['burner_site']).xpos[:2]
        return [np.array([*xy, hi[2]])]
    # Sample near the accessible front first, rejecting occupied footprints.
    obstacles=[]
    for other in controller.manifest['bindings']:
        if other['body'] in (obj, spec['body']): continue
        a,b=bounds(controller.model,controller.data,other['body'])
        if b[2] > hi[2] and a[2] < hi[2]+extent[2]: obstacles.append((a,b))
    result=[]
    for y in np.linspace(lo[1]+radius[1],hi[1]-radius[1],5):
        for x in np.linspace(lo[0]+radius[0],hi[0]-radius[0],7):
            p=np.array([x,y,hi[2]])
            if np.any(p[:2]-radius<lo[:2]) or np.any(p[:2]+radius>hi[:2]): continue
            if not footprint_on_surface(controller,gid,p[:2],radius): continue
            if any(np.all(p[:2]+radius>a[:2]) and np.all(p[:2]-radius<b[:2]) for a,b in obstacles): continue
            result.append(p)
    if not result: raise RuntimeError('No clear supported placement inside '+role)
    if role.endswith('_basket'):
        # Pack from one side; a centered first item can block both remaining
        # top-down approaches even when two items fit in the basket.
        rim = collision_vertices(controller.model, controller.data, controller.descendants(spec['body']))
        if extent[2] > float(rim[:, 2].max()-hi[2])+.02:
            # Tall items retain the accessible corner-first approach.
            return sorted(result, key=lambda p: (float(p[0]), float(p[1])))
        outward = np.asarray(getattr(controller, 'task_outward', {}).get(role, (0., -1.)))
        # Fill the far side first so subsequent descents do not reach across
        # an existing item. When occupied, prefer maximum separation.
        def order(p):
            gaps = [np.linalg.norm(np.maximum(np.maximum(a[:2]-p[:2], p[:2]-b[:2]), 0.))
                    for a,b in obstacles if np.all(b[:2]>lo[:2]) and np.all(a[:2]<hi[:2])]
            return (-min(gaps) if gaps else 0., float(p[:2] @ outward), float(p[0]))
        return sorted(result,key=order)
    center=(lo+hi)/2
    return sorted(result,key=lambda p:float(np.linalg.norm(p[:2]-center[:2])))


def supported_by_floor(controller, object_bodies, floor_geom):
    """Require upward support from the selected slab, not another fixture part."""
    for contact in controller.data.contact:
        if contact.dist > .002:
            continue
        a,b=int(contact.geom1),int(contact.geom2)
        if a == floor_geom and controller.model.geom_bodyid[b] in object_bodies:
            if contact.frame[2] > .7:
                return True
        if b == floor_geom and controller.model.geom_bodyid[a] in object_bodies:
            if -contact.frame[2] > .7:
                return True
    return False
