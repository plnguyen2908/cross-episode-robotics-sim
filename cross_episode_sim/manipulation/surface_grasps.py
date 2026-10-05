"""Geometry-based opposing-contact candidates, independent of semantic object class."""
import numpy as np
import mujoco
import trimesh
from scipy.spatial import cKDTree


def sample_collision_surface(model, data, geom_ids, count=1800, seed=7):
    triangles=[]
    for g in sorted(geom_ids):
        mesh=int(model.geom_dataid[g])
        if mesh<0:
            kind=int(model.geom_type[g]); size=model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                surface=trimesh.creation.box(extents=2*size)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                surface=trimesh.creation.icosphere(subdivisions=3, radius=size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                surface=trimesh.creation.icosphere(subdivisions=3)
                surface.vertices *= size
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                surface=trimesh.creation.cylinder(radius=size[0], height=2*size[1], sections=64)
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                surface=trimesh.creation.capsule(radius=size[0], height=2*size[1], count=[32,32])
                surface.vertices[:,2] -= surface.bounds[:,2].mean()
            else:
                raise ValueError(f'Unsupported object collision geom type: {kind}')
            vertices=surface.vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
            triangles.append(vertices[surface.faces])
            continue
        a,n=int(model.mesh_vertadr[mesh]),int(model.mesh_vertnum[mesh])
        vertices=model.mesh_vert[a:a+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        a,n=int(model.mesh_faceadr[mesh]),int(model.mesh_facenum[mesh])
        triangles.append(vertices[model.mesh_face[a:a+n]])
    triangles=np.concatenate(triangles)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=np.linalg.norm(cross,axis=1)
    keep=area>1e-12
    triangles,cross,area=triangles[keep],cross[keep],area[keep]
    rng=np.random.default_rng(seed)
    ids=rng.choice(len(triangles),count,p=area/area.sum())
    u=np.sqrt(rng.random(count));v=rng.random(count)
    points=(1-u[:,None])*triangles[ids,0]+(u*(1-v))[:,None]*triangles[ids,1]+(u*v)[:,None]*triangles[ids,2]
    normals=cross[ids]/area[ids,None]
    return points,normals


def candidates(points,normals,max_width=.075,seed=7):
    """Sample opposing surface contacts and roll the approach around their closing axis.

    Local contact geometry, not whole-object width, determines eligibility.
    These hypotheses still require collision, IK, and physical qualification.
    """
    tree=cKDTree(points)
    rng=np.random.default_rng(seed)
    pairs=[]
    used=set()
    low,high=points.min(0),points.max(0)
    for i in rng.permutation(len(points)):
        js=np.asarray(tree.query_ball_point(points[i],max_width),dtype=int)
        delta=points[js]-points[i]
        width=np.linalg.norm(delta,axis=1)
        direction=delta/np.maximum(width[:,None],1e-12)
        align_a=-(direction@normals[i])
        align_b=np.sum(direction*normals[js],axis=1)
        valid=(width>.008)&(align_a>.90)&(align_b>.90)
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        score=np.minimum(align_a[ids],align_b[ids])
        k=ids[np.argmax(score)]
        j=js[k]
        center=(points[i]+points[j])/2
        if min(points[i,2],points[j,2])<low[2]+.015:continue
        key=tuple(np.round(center/.008).astype(int))+tuple(np.round(np.abs(direction[k])/.2).astype(int))
        if key in used:continue
        used.add(key)
        pairs.append((center,direction[k],float(width[k]),float(min(align_a[k],align_b[k])),points[i],points[j]))
    result=[]
    for center,y,width,quality,a,b in pairs:
        # A frame around the closing axis samples ALL approach rolls.
        reference=np.array([0.,0.,-1.])
        if abs(y@reference)>.95:reference=np.array([0.,-1.,0.])
        z0=reference-y*(y@reference);z0/=np.linalg.norm(z0)
        z1=np.cross(y,z0)
        for roll in np.linspace(0,2*np.pi,12,endpoint=False):
            z=np.cos(roll)*z0+np.sin(roll)*z1
            if z[2]>.15:continue # approaches from below the tabletop are unsuitable here
            x=np.cross(y,z)
            rotation=np.column_stack([x,y,z])
            family='top' if z[2]<-.9 else ('side' if abs(z[2])<.2 else 'oblique')
            height=float((center[2]-low[2])/max(high[2]-low[2],1e-6))
            result.append(dict(center=center.tolist(),rotation=rotation.tolist(),
                tcp=(center+.019*z).tolist(),width=width,quality=quality,
                contacts=[a.tolist(),b.tolist()],height_fraction=height,family=family))
    # Interleave approach families and height bins so a cap-only solution cannot exhaust the budget.
    buckets={}
    for c in result:
        key=(c['family'],min(4,int(c['height_fraction']*5)))
        buckets.setdefault(key,[]).append(c)
    for values in buckets.values():values.sort(key=lambda c:-c['quality'])
    ordered=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                c=buckets[key].pop(0);c['id']=len(ordered);ordered.append(c)
    return ordered
