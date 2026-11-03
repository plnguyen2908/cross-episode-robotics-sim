"""Robot-only IK and collision-validated motion for sensor-proposed grasp poses."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def merge_voxels(voxels):
    """Partition occupied integer cells into disjoint boxes, without filling holes."""
    remaining=set(map(tuple,np.asarray(voxels,dtype=int)));result=[]
    while remaining:
        lo=min(remaining);hi=list(lo)
        for axis in (0,1,2):
            while True:
                nxt=hi[axis]+1
                ranges=[range(lo[j],hi[j]+1) if j!=axis else [nxt] for j in range(3)]
                cells=[(x,y,z) for x in ranges[0] for y in ranges[1] for z in ranges[2]]
                if not all(c in remaining for c in cells):break
                hi[axis]=nxt
        for x in range(lo[0],hi[0]+1):
            for y in range(lo[1],hi[1]+1):
                for z in range(lo[2],hi[2]+1):remaining.remove((x,y,z))
        result.append((np.array(lo),np.array(hi)))
    return result


def smooth_joint_path(start,end,dt=.02,velocity=1.,acceleration=2.):
    delta=np.asarray(end)-start
    duration=max(.3,float(np.max(1.875*np.abs(delta)/velocity)),float(np.max(np.sqrt(5.774*np.abs(delta)/acceleration))))
    count=int(np.ceil(duration/dt))+1;t=np.linspace(0,1,count)
    s=10*t**3-15*t**4+6*t**5
    return np.asarray(start)[None]+s[:,None]*delta


class ObservedGraspPlanner:
    """No simulator handle. Calibrated FK, robot joint limits, observed cuboids only.

    Fast path: bounded smooth joint interpolation, checked at every command and
    additional <=0.001-rad summed-joint intervals. Optional fallback: one cuRobo cspace attempt.
    No unsafe straight-line fallback is executed.
    """
    def __init__(self,arm,geometry,boxes,allow_curobo_fallback=False, fixed_floor_support_spheres=None):
        self.fixed_floor_support_spheres=np.asarray(fixed_floor_support_spheres if fixed_floor_support_spheres is not None else []).reshape(-1,4)
        self.allow_curobo_fallback=allow_curobo_fallback
        self.arm=arm;self.geometry=geometry;self.boxes=boxes;self.dt=.02;self.events=[]
        self.centers=np.array([b.pose[:3] for b in boxes]);self.half=np.array([b.dims for b in boxes])/2
        self.rotations=Rotation.from_quat(np.array([b.pose[3:] for b in boxes]),scalar_first=True).as_matrix()
        self.tree=cKDTree(self.centers);self.max_diagonal=float(np.linalg.norm(self.half,axis=1).max())

    def clear(self,positions,margin=.005):
        import torch
        from curobo.types import JointState
        qs=np.atleast_2d(positions)
        if len(qs)>64:
            return all(self.clear(qs[i:i+64],margin) for i in range(0,len(qs),64))
        state=JointState.from_position(torch.as_tensor(qs,dtype=torch.float32,device='cuda'),joint_names=self.arm.names)
        with torch.no_grad():
            spheres=self.arm.planner.kinematics.compute_kinematics(state).robot_spheres.reshape(len(qs),-1,4)
            cfg=self.arm.planner.kinematics.get_self_collision_config();pairs=cfg.collision_pairs.long()
            radii=spheres[...,3]+cfg.sphere_padding
            gaps=(spheres[:,pairs[:,0],:3]-spheres[:,pairs[:,1],:3]).norm(dim=-1)-radii[:,pairs[:,0]]-radii[:,pairs[:,1]]
            if float(gaps.min())<margin:
                self.last_rejection={'kind':'self_collision_margin','gap':float(gaps.min())};return False
            spheres=spheres.cpu().numpy()
        return self.world_clear(spheres,margin)

    def world_clear(self,spheres,margin=.005):
        flat=spheres.reshape(-1,4);flat=flat[flat[:,3]>0]
        # Fixed base/link spheres repeat across trajectory samples. Exact
        # deduplication avoids rechecking them without changing swept coverage.
        if len(flat)>256:flat=np.unique(flat,axis=0)
        for offset in range(0,len(flat),1024):
            block=flat[offset:offset+1024]
            # Size-bucket broad phase: a distant large wall must not inflate
            # every small voxel's query radius. Exact narrow phase is unchanged.
            if not hasattr(self,'_world_groups'):
                radii=np.linalg.norm(self.half,axis=1)
                bins=np.ceil(np.log2(np.maximum(radii,.025)/.025)).astype(int)
                self._world_groups=[]
                for key in np.unique(bins):
                    ids=np.flatnonzero(bins==key)
                    self._world_groups.append((ids,cKDTree(self.centers[ids]),float(radii[ids].max())))
            sphere_parts=[];box_parts=[]
            for ids,tree,radius in self._world_groups:
                neighbors=tree.query_ball_point(block[:,:3],block[:,3]+radius+margin)
                counts=np.array([len(v) for v in neighbors])
                if not counts.sum():continue
                sphere_parts.append(np.repeat(np.arange(len(block)),counts))
                box_parts.append(ids[np.concatenate([v for v in neighbors if len(v)]).astype(int)])
            if not sphere_parts:continue
            sphere_ids=np.concatenate(sphere_parts);box_ids=np.concatenate(box_parts)
            delta=np.abs(np.einsum('bi,bij->bj',block[sphere_ids,:3]-self.centers[box_ids],self.rotations[box_ids]))-self.half[box_ids]
            distance=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)-block[sphere_ids,3]
            # Only calibrated, stationary base spheres may touch the observed floor.
            # Moving arm/payload spheres and all non-floor obstacles stay checked.
            supports=getattr(self,'fixed_floor_support_spheres',np.empty((0,4)))
            if len(supports):
                support_tree=getattr(self,'_support_tree',None)
                if support_tree is None:
                    support_tree=cKDTree(supports);self._support_tree=support_tree
                fixed=support_tree.query(block,p=np.inf)[0]<=1e-5
                floor=(self.centers[box_ids,2]+self.half[box_ids,2]<=.03)&(self.centers[box_ids,2]-self.half[box_ids,2]>=-.06)
                distance[fixed[sphere_ids]&floor]=np.inf
            worst=int(distance.argmin())
            if distance[worst]<margin:
                self.last_rejection={'kind':'observed_world_margin','gap':float(distance[worst]),'obstacle':self.boxes[box_ids[worst]].name,'sphere':block[sphere_ids[worst]].tolist()};return False
        return True

    def world_clear_box(self,box,margin=.005):
        """Exact OBB separating-axis test; retain payload padding and margin."""
        R=np.einsum('ij,njk->nik',box.axes.T,self.rotations)
        ab=np.abs(R)+1e-9
        t=(self.centers-box.center)@box.axes
        ha=box.half+margin;hb=self.half
        separated=np.any(np.abs(t)>ha+np.einsum('nij,nj->ni',ab,hb),axis=1)
        separated |= np.any(np.abs(np.einsum('ni,nij->nj',t,R))>hb+np.einsum('nij,i->nj',ab,ha),axis=1)
        for i in range(3):
            k,l=(i+1)%3,(i+2)%3
            for j in range(3):
                u,v=(j+1)%3,(j+2)%3
                lhs=np.abs(t[:,l]*R[:,k,j]-t[:,k]*R[:,l,j])
                radius=ha[k]*ab[:,l,j]+ha[l]*ab[:,k,j]+hb[:,u]*ab[:,i,v]+hb[:,v]*ab[:,i,u]
                separated |= lhs>radius
        if not separated.all():
            self.last_rejection={'kind':'observed_payload_box','obstacle':self.boxes[int(np.flatnonzero(~separated)[0])].name}
            return False
        return True

    def solve(self,q,position,rotation):
        lo,hi=self.geometry.limits[:,0]+.001,self.geometry.limits[:,1]-.001
        def residual(x):
            T=self.geometry.fk(x)[self.geometry.tcp_body]@self.geometry.tcp_local
            return np.r_[T[:3,3]-position,.2*Rotation.from_matrix(rotation@T[:3,:3].T).as_rotvec()]
        rng=np.random.default_rng(0);solutions=[];self.ik_diagnostics=[]
        for seed in [q]+[rng.uniform(lo,hi) for _ in range(5)]:
            fit=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=100)
            error=residual(fit.x)
            if np.linalg.norm(error[:3])>.003 or np.linalg.norm(error[3:])>.01:continue
            if self.clear(fit.x):solutions.append(fit.x)
            else:self.ik_diagnostics.append(self.last_rejection)
        return sorted(solutions,key=lambda x:np.linalg.norm(x-q))

    def plan(self,q,position,rotation):
        start=time.monotonic();solutions=self.solve(q,position,rotation)
        if not solutions:raise RuntimeError('No collision-clear IK solution: '+str(self.ik_diagnostics[:3]))
        for goal in solutions:
            path=smooth_joint_path(q,goal)
            dense=np.linspace(q,goal,max(2,int(np.ceil(np.sum(np.abs(goal-q))/.001))+1))
            if self.clear(path) and self.clear(dense):
                self.events.append(dict(method='checked_quintic',wall_seconds=time.monotonic()-start,simulated_seconds=(len(path)-1)*self.dt))
                return path
        if not self.allow_curobo_fallback:
            raise RuntimeError('Reachable IK, but no validated smooth joint path')
        # Try one selected joint goal rather than five retries for every IK seed.
        import torch
        from curobo.types import JointState
        def state(x):return JointState.from_position(torch.tensor(np.asarray(x)[None],device='cuda',dtype=torch.float32),joint_names=self.arm.names)
        result=self.arm.planner.plan_cspace(state(solutions[0]),state(q),max_attempts=1)
        if result is None or not result.success.any():raise RuntimeError('No collision-checked path to reachable pose')
        b,s=result.success.nonzero(as_tuple=True);end=int(result.interpolated_last_tstep[b[0],s[0]].item());tr=result.interpolated_trajectory
        indices=[tr.joint_names.index(n) for n in self.arm.names];raw=tr.position[b[0],s[0],:end,indices].cpu().numpy()
        old=np.arange(len(raw))*self.arm.dt;times=np.arange(0,old[-1]+.019,.02)
        path=np.column_stack([np.interp(times,old,raw[:,j]) for j in range(7)])
        if not self.clear(path):raise RuntimeError('Interpolated cuRobo commands failed collision validation')
        self.events.append(dict(method='curobo_cspace',wall_seconds=time.monotonic()-start,simulated_seconds=(len(path)-1)*self.dt))
        return path


def observed_target_mask(points,target,support):
    """Refine a semantic map target to its connected depth surface above support.

    The map centroid can be biased by a partial view. Do not use a fixed cylinder
    around it: that can leave the target's rim classified as an external obstacle.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    local=(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.28)&(points[:,2]>support+.004)&(points[:,2]<target[2]+.20)
    indices=np.flatnonzero(local)
    if len(indices)<20:raise RuntimeError('Insufficient observed target surface')
    vox,inverse=np.unique(np.floor(points[indices]/.006).astype(int),axis=0,return_inverse=True)
    centers=(vox+.5)*.006;tree=cKDTree(centers);pairs=tree.query_pairs(.0125,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(vox),len(vox)))
    _,labels=connected_components(graph,directed=False)
    choices=[]
    for label in np.unique(labels):
        group=centers[labels==label]
        if len(group)<20:continue
        distance=np.linalg.norm(group-target,axis=1).min()
        if distance<.12:choices.append((distance,label))
    if not choices:raise RuntimeError('No connected observed object near semantic target')
    _,label=min(choices);mask=np.zeros(len(points),bool);mask[indices]=labels[inverse]==label
    return mask


def build_observed_boxes(grid,T_map_base,points,target):
    """Compress sensor occupancy exactly; retain support and non-target surfaces."""
    from curobo._src.geom.types import Cuboid
    R=T_map_base[:3,:3];indices=np.argwhere(grid.evidence>=2)
    step=np.array([grid.config.resolution,grid.config.resolution,grid.config.height_resolution])
    origin=np.r_[grid.origin,grid.config.min_obstacle_height]
    xyz_indices=indices[:,[1,0,2]]
    centers=origin+(xyz_indices+.5)*step
    base=(centers-T_map_base[:3,3])@R
    keep=(np.linalg.norm(base[:,:2],axis=1)<1.4)&(np.linalg.norm(base-target,axis=1)>.65)
    quat=Rotation.from_matrix(R.T).as_quat(scalar_first=True).tolist();boxes=[]
    for i,(lo,hi) in enumerate(merge_voxels(xyz_indices[keep])):
        center=origin+(lo+hi+1)*step/2;center=(center-T_map_base[:3,3])@R
        boxes.append(Cuboid(name=f'map_{i}',pose=center.tolist()+quat,dims=((hi-lo+1)*step+.01).tolist()))
    near=points[np.linalg.norm(points[:,:2]-target[:2],axis=1)<.2]
    levels,counts=np.unique(np.round(near[:,2]/.005),return_counts=True);support=float(levels[counts.argmax()]*.005)
    contact=observed_target_mask(points,target,support)
    vox=np.unique(np.floor(points[~contact]/.025).astype(int),axis=0)
    for i,(lo,hi) in enumerate(merge_voxels(vox)):
        boxes.append(Cuboid(name=f'depth_{i}',pose=((lo+hi+1)*.025/2).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.025+.003).tolist()))
    return boxes,dict(fine_voxels=len(vox),coarse_voxels=int(keep.sum()),merged_boxes=len(boxes),support_z=support,target_surface_points=int(contact.sum()),target_surface_bounds=[points[contact].min(0).tolist(),points[contact].max(0).tolist()])
