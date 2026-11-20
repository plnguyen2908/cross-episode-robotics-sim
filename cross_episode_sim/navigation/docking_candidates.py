"""Shared surface-relative docking geometry; no simulator or map access."""
import numpy as np


def surface_docks(point, lower, upper, outward, distances, laterals):
    """Original household oracle candidate order, supplied by either geometry source."""
    point=np.asarray(point,float);outward=np.asarray(outward,float)
    axis=int(np.argmax(np.abs(outward)));edge=(upper if outward[axis]>0 else lower)[axis]
    for distance in distances:
        for lateral in laterals:
            xy=point[:2].copy();xy[axis]=edge+outward[axis]*distance;xy[1-axis]+=lateral
            yield np.array([*xy,np.arctan2(point[1]-xy[1],point[0]-xy[0])])


def household_surface_docks(point,lower,upper,outward,*,fridge=False,burner=False,pickup=False):
    distances=(((.50,.43,.35,.55) if pickup else (.43,.50,.35,.55)) if fridge else
               ((.26,.30,.35,.43) if upper[2]>=.88 or burner else (.35,.43,.50,.28)))
    if burner:distances=(*distances,.50,.60,.70)
    laterals=(0.,-.10,.10,-.20,-.30,-.40) if fridge else (0.,-.10,.10)
    yield from surface_docks(point,lower,upper,outward,distances,laterals)


def frontal_surface_docks(point,lower,upper,outward):
    """Centered, continuously oriented pickup poses; no oblique fallback."""
    point=np.asarray(point,float);normal=np.asarray(outward,float)
    normal=normal/np.linalg.norm(normal)
    edge=float(np.where(normal>=0,np.asarray(upper)[:2],np.asarray(lower)[:2])@normal)
    for distance in (.26,.30,.35,.43,.50,.55,.60):
        xy=point[:2]+(edge-point[:2]@normal+distance)*normal
        yield np.r_[xy,np.arctan2(-normal[1],-normal[0])]


def standoff_surface_docks(point,outward,gaps):
    """The same centered frontal poses at the given distances from the target."""
    point=np.asarray(point,float);normal=np.asarray(outward,float)[:2]
    normal=normal/np.linalg.norm(normal)
    for gap in gaps:
        yield np.r_[point[:2]+gap*normal,np.arctan2(-normal[1],-normal[0])]


def fit_surface_frame(points,target,base_xy):
    """Fit an observed cabinet face and countertop edge in arbitrary map axes."""
    points=np.asarray(points);target=np.asarray(target)
    nearby=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.45)&(points[:,2]>target[2]-.20)&(points[:,2]<target[2]-.01)]
    if len(nearby)<100:raise ValueError('Insufficient observed support')
    bins=np.round(nearby[:,2]/.003).astype(int);values,counts=np.unique(bins,return_counts=True);height=values[counts.argmax()]*.003
    support=nearby[np.abs(nearby[:,2]-height)<.006]
    face=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<1.)&(points[:,2]>.30)&(points[:,2]<height-.08),:2]
    if len(face)<100:raise ValueError('Insufficient observed counter face')
    # Bound runtime without changing spatial sampling or using simulator labels.
    face=face[::max(1,len(face)//6000)]
    rng=np.random.default_rng(0);best=None
    view=np.asarray(base_xy)-target[:2];view/=max(np.linalg.norm(view),1e-9)
    for _ in range(200):
        a,b=face[rng.choice(len(face),2,replace=False)];t=b-a;length=np.linalg.norm(t)
        if length<.25:continue
        normal=np.array([-t[1],t[0]])/length
        if abs(normal@view)<.3:continue
        if normal@view<0:normal=-normal
        if not -.65<float((target[:2]-a)@normal)<.10:continue
        mask=np.abs((face-a)@normal)<.015
        if best is None or mask.sum()>best.sum():best=mask
    if best is None or best.sum()<100 or best.mean()<.35:raise ValueError('Counter orientation unverified')
    center=face[best].mean(0);_,_,vt=np.linalg.svd(face[best]-center,full_matrices=False);normal=vt[-1]
    if normal@view<0:normal=-normal
    tangent=np.array([normal[1],-normal[0]])
    support=support[np.abs((support[:,:2]-target[:2])@tangent)<.20]
    if len(support)<100:raise ValueError('Support edge not observed')
    edge=max(float(np.median(face[best]@normal)),float(np.quantile(support[:,:2]@normal,.99)))
    point=np.r_[target[:2]+(edge-target[:2]@normal)*normal,height]
    return point,point.copy(),normal


def fit_open_edge(clouds,target,base_xy,cell=.02,reach=1.2):
    """Fit the support edge nearest target from sight lines that pass it.

    clouds holds one (camera origin, points) pair per camera. A table has no
    cabinet face under its edge for fit_surface_frame. A point measured below
    the top proves its sight line crossed the top's height in open air, so top
    cells bordering such crossings are a measured edge; top cells that end at
    the image border or behind an object are not.
    """
    target=np.asarray(target,float);base_xy=np.asarray(base_xy,float)
    clouds=[(np.asarray(origin,float),np.asarray(points,float)) for origin,points in clouds]
    clouds=[(origin,points[np.linalg.norm(points[:,:2]-target[:2],axis=1)<reach+1.]) for origin,points in clouds]
    nearby=np.concatenate([points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.45)&(points[:,2]>target[2]-.20)&(points[:,2]<target[2]-.01)] for _,points in clouds])
    if len(nearby)<100:raise ValueError('Insufficient observed support')
    bins=np.round(nearby[:,2]/.003).astype(int);values,counts=np.unique(bins,return_counts=True);height=values[counts.argmax()]*.003
    low=target[:2]-reach;size=int(np.ceil(2*reach/cell));top=np.zeros((size,size),bool);passed=np.zeros((size,size),bool)
    def mark(grid,xy):
        index=np.floor((xy-low)/cell).astype(int);index=index[((index>=0)&(index<size)).all(1)];grid[index[:,0],index[:,1]]=True
    support=np.concatenate([points[np.abs(points[:,2]-height)<.006,:2] for _,points in clouds])
    for origin,points in clouds:
        mark(top,points[np.abs(points[:,2]-height)<.01,:2])
        # A camera at or under the top sees below it without passing the edge.
        if origin[2]<=height+.05:continue
        below=points[points[:,2]<height-.10]
        mark(passed,origin[:2]+(below[:,:2]-origin[:2])*((origin[2]-height)/(origin[2]-below[:,2]))[:,None])
    passed&=~top;beside=np.zeros_like(passed);beside[1:-1,1:-1]=passed[1:-1,1:-1]
    for dx,dy in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        beside[1:-1,1:-1]|=passed[1+dx:size-1+dx,1+dy:size-1+dy]
    edge=low+(np.argwhere(top&beside)+.5)*cell
    if len(edge)<10:raise ValueError('Support edge not observed')
    passed_xy=low+(np.argwhere(passed)+.5)*cell;top_xy=low+(np.argwhere(top)+.5)*cell
    rng=np.random.default_rng(0);remaining=np.ones(len(edge),bool);found=[]
    # A table shows up to four edges; fit each and keep the one nearest target.
    for _ in range(4):
        pool=np.flatnonzero(remaining);best=None
        if len(pool)<10:break
        for _ in range(300):
            a,b=edge[rng.choice(pool,2,replace=False)];t=b-a;length=np.linalg.norm(t)
            if length<.20:continue
            mask=remaining&(np.abs((edge-a)@np.array([-t[1],t[0]]))<cell*length)
            if best is None or mask.sum()>best.sum():best=mask
        if best is None or best.sum()<10:break
        remaining&=~best
        center=edge[best].mean(0);_,_,vt=np.linalg.svd(edge[best]-center,full_matrices=False);tangent,normal=vt[0],vt[1]
        along=(edge[best]-center)@tangent;at=float((target[:2]-center)@tangent)
        def across(xy):
            local=xy-center;offset=local@normal
            return offset[(local@tangent>=along.min())&(local@tangent<=along.max())&(np.abs(offset)<.10)]
        outside=across(passed_xy)
        if len(outside)<10:continue
        # Outward is the side sight lines pass on; the top lies on the other.
        if np.median(outside)<0:normal=-normal;outside=-outside
        if (outside>0).mean()<.8 or (across(top_xy)<cell).mean()<.8:continue
        # The edge has to be measured beside the target, not only further along.
        if not along.min()-.10<=at<=along.max()+.10 or np.count_nonzero(np.abs(along-at)<.25)<6:continue
        strip=support[np.abs((support-center)@tangent-at)<.25];strip=(strip-center)@normal;strip=strip[np.abs(strip)<3*cell]
        if len(strip)<20:continue
        depth=float(center@normal+np.quantile(strip,.99)-target[:2]@normal)
        if not -.10<depth<.65 or (base_xy-target[:2])@normal<depth:continue
        found.append((depth,normal))
    if not found:raise ValueError('Support edge orientation unverified')
    depth,normal=min(found,key=lambda item:item[0])
    point=np.r_[target[:2]+depth*normal,height]
    return point,point.copy(),normal


def observed_surface(packet,T_map_base,target):
    """Estimate the support frame from calibrated onboard RGB-D, not map axes."""
    clouds=[]
    for c in packet.cameras.values():
        valid=c.valid_depth.copy()
        if c.robot_mask is not None:valid &= ~c.robot_mask
        v,u=np.where(valid);z=c.depth_m[v,u]
        xyz=np.column_stack(((u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z))
        T=T_map_base@c.T_base_optical;clouds.append((T[:3,3],xyz@T[:3,:3].T+T[:3,3]))
    try:return fit_surface_frame(np.concatenate([points for _,points in clouds]),target,T_map_base[:2,3])
    except ValueError as face:
        # Legged tables have no face under the edge; use the edge itself.
        try:return fit_open_edge(clouds,target,T_map_base[:2,3])
        except ValueError as edge:raise ValueError(f'{face}; {edge}') from None
