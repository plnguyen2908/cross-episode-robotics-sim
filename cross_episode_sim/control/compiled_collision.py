"""Compiled, float64 sphere/observed-box predicate. No reduced sweep sampling.

Axis-aligned bounds are conservative broad phase only. Rotated boxes still use
exact signed box distance. No fast-math approximation or scene truth is used.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def sphere_box_collision(spheres, centers, half, rotations, world_half, margin, supports):
    for i in range(len(spheres)):
        x,y,z,r=spheres[i]
        if r<=0:
            continue
        fixed=False
        for support in supports:
            if max(abs(x-support[0]),abs(y-support[1]),abs(z-support[2]),abs(r-support[3]))<=1e-5:
                fixed=True
                break
        reach=r+margin
        for j in range(len(centers)):
            dx=x-centers[j,0];dy=y-centers[j,1];dz=z-centers[j,2]
            if (abs(dx)>world_half[j,0]+reach+1e-12 or
                abs(dy)>world_half[j,1]+reach+1e-12 or
                abs(dz)>world_half[j,2]+reach+1e-12):
                continue
            if fixed and centers[j,2]+half[j,2]<=.03 and centers[j,2]-half[j,2]>=-.06:
                continue
            a=abs(dx*rotations[j,0,0]+dy*rotations[j,1,0]+dz*rotations[j,2,0])-half[j,0]
            b=abs(dx*rotations[j,0,1]+dy*rotations[j,1,1]+dz*rotations[j,2,1])-half[j,1]
            c=abs(dx*rotations[j,0,2]+dy*rotations[j,1,2]+dz*rotations[j,2,2])-half[j,2]
            gap=np.sqrt(max(a,0.)**2+max(b,0.)**2+max(c,0.)**2)+min(max(a,b,c),0.)-r
            if gap<margin:
                return i,j,gap
    return -1,-1,np.inf


def build_box_bvh(centers, half, rotations, leaf_size=8):
    """Static conservative hierarchy; leaves retain every observed box."""
    world_half=np.einsum('nij,nj->ni',np.abs(rotations),half)
    lower=centers-world_half;upper=centers+world_half
    bounds=[];children=[];spans=[];ordered=[]
    def build(ids):
        index=len(bounds)
        bounds.append(np.stack([lower[ids].min(0),upper[ids].max(0)]))
        children.append([-1,-1]);spans.append([0,0])
        if len(ids)<=leaf_size:
            spans[index]=[len(ordered),len(ids)];ordered.extend(ids)
        else:
            axis=int(np.argmax(np.ptp(centers[ids],axis=0)))
            ids=ids[np.argsort(centers[ids,axis],kind='stable')];mid=len(ids)//2
            children[index]=[build(ids[:mid]),build(ids[mid:])]
        return index
    if len(centers):build(np.arange(len(centers)))
    return (np.asarray(bounds,dtype=np.float64).reshape(-1,2,3),
            np.asarray(children,dtype=np.int64).reshape(-1,2),
            np.asarray(spans,dtype=np.int64).reshape(-1,2),np.asarray(ordered,dtype=np.int64))


@njit(cache=True)
def sphere_box_bvh_collision(spheres,centers,half,rotations,margin,supports,bounds,children,spans,ordered):
    if not len(bounds):return -1,-1,np.inf
    stack=np.empty(len(bounds),dtype=np.int64)
    for i in range(len(spheres)):
        x,y,z,r=spheres[i]
        if r<=0:continue
        fixed=False
        for support in supports:
            if max(abs(x-support[0]),abs(y-support[1]),abs(z-support[2]),abs(r-support[3]))<=1e-5:
                fixed=True;break
        reach=r+margin+1e-12
        top=1;stack[0]=0
        while top:
            top-=1;node=stack[top]
            if (x<bounds[node,0,0]-reach or x>bounds[node,1,0]+reach or
                y<bounds[node,0,1]-reach or y>bounds[node,1,1]+reach or
                z<bounds[node,0,2]-reach or z>bounds[node,1,2]+reach):continue
            if children[node,0]>=0:
                stack[top]=children[node,0];stack[top+1]=children[node,1];top+=2;continue
            start,count=spans[node]
            for k in range(start,start+count):
                j=ordered[k]
                if fixed and centers[j,2]+half[j,2]<=.03 and centers[j,2]-half[j,2]>=-.06:continue
                dx=x-centers[j,0];dy=y-centers[j,1];dz=z-centers[j,2]
                a=abs(dx*rotations[j,0,0]+dy*rotations[j,1,0]+dz*rotations[j,2,0])-half[j,0]
                b=abs(dx*rotations[j,0,1]+dy*rotations[j,1,1]+dz*rotations[j,2,1])-half[j,1]
                c=abs(dx*rotations[j,0,2]+dy*rotations[j,1,2]+dz*rotations[j,2,2])-half[j,2]
                gap=np.sqrt(max(a,0.)**2+max(b,0.)**2+max(c,0.)**2)+min(max(a,b,c),0.)-r
                if gap<margin:return i,j,gap
    return -1,-1,np.inf


@njit(cache=True)
def point_box_minimum(points, center, axes, half, padding=.004):
    """Exact signed point/OBB minimum, float64 without temporary cloud arrays."""
    best=np.inf
    for i in range(len(points)):
        x=points[i,0]-center[0];y=points[i,1]-center[1];z=points[i,2]-center[2]
        a=abs(x*axes[0,0]+y*axes[1,0]+z*axes[2,0])-half[0]
        b=abs(x*axes[0,1]+y*axes[1,1]+z*axes[2,1])-half[1]
        c=abs(x*axes[0,2]+y*axes[1,2]+z*axes[2,2])-half[2]
        ap=max(a,0.);bp=max(b,0.);cp=max(c,0.)
        gap=np.sqrt(ap*ap+bp*bp+cp*cp)+min(max(a,b,c),0.)-padding
        best=min(best,gap)
    return best
