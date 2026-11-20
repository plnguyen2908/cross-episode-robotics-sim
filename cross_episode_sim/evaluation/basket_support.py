"""Private scoring for released objects contained in a basket, including stacked and leaning ones."""
import mujoco
import numpy as np
from .grasp_contact import contact_root_lookup


def basket_release_score(model,data,targets,basket):
    token=basket.replace(' ','_')
    roots=[int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
    lookup=contact_root_lookup(model.body_parentid,roots)
    floors=[g for g in range(model.ngeom) if token in (model.geom(g).name or '') and (model.geom(g).name or '').endswith('_floor')]
    pads={i for i in range(model.nbody) if (model.body(i).name or '').startswith('robot_0/gripper/')}
    held=set();edges=[];supported=set()
    for c in data.contact:
        if c.dist>.002:continue
        g1,g2=int(c.geom1),int(c.geom2);b1,b2=map(int,model.geom_bodyid[[g1,g2]])
        r1,r2=int(lookup[b1]),int(lookup[b2])
        if b1 in pads and r2>=0:held.add(r2)
        if b2 in pads and r1>=0:held.add(r1)
        if abs(c.frame[2])<.7:continue
        lower,upper=(g1,g2) if c.frame[2]>0 else (g2,g1)
        parent,child=int(lookup[model.geom_bodyid[lower]]),int(lookup[model.geom_bodyid[upper]])
        if child<0:continue
        if lower in floors:supported.add(child)
        elif parent>=0 and parent!=child:edges.append((parent,child))
    for _ in roots:
        new=supported|{child for parent,child in edges if parent in supported}
        if new==supported:break
        supported=new
    # A released item wedged between other contents and a wall has no
    # near-vertical support normal, yet the basket alone holds it up. Count it
    # when everything it touches is this basket or contents the basket supports.
    def leaning(root):
        touching=False
        for c in data.contact:
            if c.dist>.002:continue
            g1,g2=int(c.geom1),int(c.geom2);r1,r2=int(lookup[model.geom_bodyid[g1]]),int(lookup[model.geom_bodyid[g2]])
            if (r1==root)==(r2==root):continue
            other,partner=(g2,r2) if r1==root else (g1,r1)
            if partner not in supported and token not in (model.geom(other).name or ''):return False
            touching=True
        return touching
    basket_roots={int(lookup[model.geom_bodyid[g]]) for g in floors}
    for _ in roots:
        new={root for root in set(roots)-supported-held-basket_roots if leaning(root)}
        if not new:break
        supported|=new
    accepted=[]
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for root in set(targets)&supported-held:
        vertices=[]
        for g in range(model.ngeom):
            if lookup[model.geom_bodyid[g]]!=root or not(model.geom_contype[g] or model.geom_conaffinity[g]):continue
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=model.geom_dataid[g];start=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
                local=model.mesh_vert[start:start+n]
            else:local=model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:]
            vertices.extend(local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
        if not vertices:continue
        contained=False
        for floor in floors:
            local=(np.asarray(vertices)-data.geom_xpos[floor])@data.geom_xmat[floor].reshape(3,3)
            center,half=model.geom_aabb[floor,:3],model.geom_aabb[floor,3:]
            contained |= bool(np.all(abs(local[:,:2]-center[:2])<=half[:2]+.003))
        j=int(model.body_jntadr[root]);adr=int(model.jnt_dofadr[j])
        stable=np.linalg.norm(data.qvel[adr:adr+3])<.03 and np.linalg.norm(data.qvel[adr+3:adr+6])<.1
        if contained and stable:accepted.append(int(root))
    return dict(success=bool(accepted),initially_held=sorted(map(int,targets)),supported=sorted(supported),
                released_supported=accepted,stacked_support_allowed=True,scoring='held up by the basket alone (floor, walls or supported contents), full XY containment, released and stable')
