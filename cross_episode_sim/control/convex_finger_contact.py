"""Convex finger-mesh narrow phase for tiny observed appliance controls."""
import numpy as np
from scipy.spatial import ConvexHull
from .observed_hinged_planner import ObservedHingedPlanner


def convex_shape(vertices):
    vertices=np.asarray(vertices,float);hull=ConvexHull(vertices)
    edges=np.concatenate([vertices[hull.simplices[:,i]]-vertices[hull.simplices[:,(i+1)%3]] for i in range(3)])
    return vertices[hull.vertices],hull.equations[:,:3],edges


def convex_box_overlap(shape, center, axes, half, margin=0.):
    """SAT: convex polyhedron versus an OBB, in the finger's local frame."""
    vertices,normals,edges=shape
    candidates=np.concatenate([normals,axes.T,np.cross(edges[:,None,:],axes.T[None,:,:]).reshape(-1,3)])
    norms=np.linalg.norm(candidates,axis=1);candidates=candidates[norms>1e-9]/norms[norms>1e-9,None]
    # Repeated triangle edges only duplicate axes; removing them is exact for
    # unchanged rows and is not a collision-margin approximation.
    candidates=np.unique(candidates,axis=0)
    # Small 3-D products must not launch a large BLAS thread pool at every
    # swept sample. Explicit sums preserve the SAT without that overhead.
    projection=(vertices[:,None,:]*candidates[None,:,:]).sum(2)
    mid=(np.asarray(center)[None,:]*candidates).sum(1)
    transformed=np.einsum('ni,ij->nj',candidates,axes,optimize=False)
    radius=(np.abs(transformed)*np.asarray(half)).sum(1)
    return not np.any((projection.max(0)+margin<mid-radius)|(mid+radius+margin<projection.min(0)))


class ObservedControlPlanner(ObservedHingedPlanner):
    def world_clear_box(self,box,margin=.005):
        if super().world_clear_box(box,margin):return True
        shape=getattr(self.finger_geometry,'convex_shapes',{}).get(box.name)
        if shape is None:return False
        ids=self.tree.query_ball_point(box.center,np.linalg.norm(box.half)+self.max_diagonal+margin)
        # A long counter gives the KD-tree a large query radius. Do not run
        # mesh GJK against every distant voxel returned by that broad phase.
        ids=np.asarray(ids,dtype=int)
        world_half=np.einsum('nij,nj->ni',np.abs(self.rotations[ids]),self.half[ids])
        robot_half=np.abs(box.axes)@box.half
        ids=ids[np.all(np.abs(self.centers[ids]-box.center)<=world_half+robot_half+margin,axis=1)]
        for index in ids:
            center=(self.centers[index]-box.center)@box.axes
            axes=box.axes.T@self.rotations[index]
            if shape.overlap(center,axes,self.half[index],margin):
                self.last_rejection=dict(kind='observed_finger_convex',part=box.name,obstacle=self.boxes[index].name,
                                         obstacle_center=self.centers[index].tolist())
                return False
        return True


class NativeConvexShape:
    """MuJoCo convex-distance kernel on copied robot vertices and sensed boxes.

    This tiny collision model contains no environment bodies or state. It is
    not the live simulator, and never reads simulator contact truth. Native GJK
    avoids quadratic face/edge SAT work on the detailed gripper meshes.
    """
    def __init__(self,vertices):
        import mujoco
        self.mujoco=mujoco
        text=' '.join(format(float(x),'.12g') for x in np.asarray(vertices).ravel())
        self.model=mujoco.MjModel.from_xml_string(
            '<mujoco><asset><mesh name="finger" maxhullvert="-1" vertex="'+text+'"/></asset>'
            '<worldbody><geom type="mesh" mesh="finger"/><geom type="box" size=".01 .01 .01"/></worldbody></mujoco>')
        self.data=mujoco.MjData(self.model);mujoco.mj_forward(self.model,self.data)

    def overlap(self,center,axes,half,margin):
        self.model.geom_size[1]=half;self.model.geom_rbound[1]=np.linalg.norm(half)
        self.data.geom_xpos[1]=center;self.data.geom_xmat[1]=np.asarray(axes).ravel()
        return self.mujoco.mj_geomDistance(self.model,self.data,0,1,float(margin+1e-6),None)<margin
