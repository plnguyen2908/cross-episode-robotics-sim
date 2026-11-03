"""Observed held-object geometry; robot calibration and estimated map only."""
from dataclasses import replace
import numpy as np
from cross_episode_sim.navigation.shaped_docking import RobotBox,ObservedShapeChecker,ShapedDocking

def invalidate_moved_surface(grid, object_points, static_points, source_resolution):
    """Invalidate cached voxels explained only by the picked RGB-D component.

    Static depth samples protect mixed voxels (including the supporting counter).
    Set invalidated cells to unknown, never free; floor evidence is unchanged.
    Inputs must share the current estimated map frame.
    """
    from scipy.spatial import cKDTree
    object_points=np.asarray(object_points);static_points=np.asarray(static_points)
    if len(object_points)<20 or len(static_points)<20:raise ValueError('Need object and static depth evidence')
    ij=np.argwhere(grid.evidence>=2)
    centers=np.c_[grid.origin[0]+(ij[:,1]+.5)*grid.config.resolution,grid.origin[1]+(ij[:,0]+.5)*grid.config.resolution,grid.config.min_obstacle_height+(ij[:,2]+.5)*grid.config.height_resolution]
    radius=float(np.linalg.norm(np.asarray(source_resolution))*.5)
    target=cKDTree(object_points).query(centers)[0]
    static=cKDTree(static_points).query(centers)[0]
    margin=np.linalg.norm([grid.config.resolution,grid.config.resolution,grid.config.height_resolution])*.5
    changed=ij[(target<=radius+margin)&(static>radius+margin)]
    grid.evidence[tuple(changed.T)]=0
    return len(changed)

class ObservedPayload:
    def __init__(self,geometry,points_tool):
        self.geometry=geometry
        p=np.asarray(points_tool,float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<20 or not np.isfinite(p).all():raise ValueError('Need observed held-object points')
        lo,hi=p.min(0),p.max(0)
        self.center=(lo+hi)/2;self.half=(hi-lo)/2+.02
        self.points_tool=p
        self.hull_equations=None
    def box(self,q):
        T=self.geometry.fk(q)[self.geometry.tcp_body]@self.geometry.tcp_local
        return RobotBox('observed_payload',T[:3,:3]@self.center+T[:3,3],T[:3,:3],self.half)
    def filter_packet(self,packet):
        box=self.box(packet.arm.position);cameras={}
        for name,c in packet.cameras.items():
            v,u=np.indices(c.depth_m.shape);z=c.depth_m
            points=np.stack([(u-c.K[0,2])*z/c.K[0,0],(v-c.K[1,2])*z/c.K[1,1],z],axis=-1)
            points=points@c.T_base_optical[:3,:3].T+c.T_base_optical[:3,3]
            # One centimetre for depth-edge rasterization/attachment residual.
            # Collision shape and its larger safety margin are unchanged.
            mask=np.all(np.abs((points-box.center)@box.axes)<=box.half+.01,axis=-1)
            cameras[name]=replace(c,valid_depth=c.valid_depth&~mask)
        return replace(packet,cameras=cameras)
    def checker(self,grid,q,pose,support):
        return ObservedShapeChecker(grid,self.geometry.boxes(q)+[self.box(q)],pose,support_poses=support)


def reconstruct_held_points(packet, geometry, radius=.35):
    """Connected observed surface near the tool, excluding calibrated robot.

    For an already-held object only. No simulator object lookup. Fail closed
    when no substantial local surface remains; proximity is not grasp success.
    """
    from scipy.spatial import cKDTree
    q=packet.arm.position;T=geometry.fk(q)[geometry.tcp_body]@geometry.tcp_local
    clouds=[]
    for camera in packet.cameras.values():
        valid=camera.valid_depth.copy()
        if camera.robot_mask is not None:valid &= ~camera.robot_mask
        v,u=np.where(valid);z=camera.depth_m[v,u]
        xyz=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z]@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
        keep=np.linalg.norm(xyz-T[:3,3],axis=1)<radius
        for box in geometry.boxes(q):keep &= ~np.all(np.abs((xyz-box.center)@box.axes)<=box.half+.01,axis=1)
        clouds.append(xyz[keep])
    points=np.concatenate(clouds)
    if len(points)<50:raise ValueError('Insufficient observed held-object surface')
    _,ids=np.unique(np.floor(points/.005).astype(int),axis=0,return_index=True);points=points[ids]
    tree=cKDTree(points);seen=np.zeros(len(points),bool);groups=[]
    for i in range(len(points)):
        if seen[i]:continue
        pending=[i];seen[i]=True;group=[]
        while pending:
            j=pending.pop();group.append(j)
            for k in tree.query_ball_point(points[j],.018):
                if not seen[k]:seen[k]=True;pending.append(k)
        if len(group)>=30 and np.min(np.linalg.norm(points[group]-T[:3,3],axis=1))<.18:groups.append(group)
    if not groups:raise ValueError('No connected observed surface near tool')
    held=points[max(groups,key=len)]
    if np.max(np.linalg.norm(held-T[:3,3],axis=1))>=radius-.01:
        raise ValueError('Observed payload reaches reconstruction boundary')
    return (held-T[:3,3])@T[:3,:3]


class PayloadDocking(ShapedDocking):
    """Shared base routing with the observed held surface in every sweep."""
    def __init__(self, grid, payload):
        super().__init__(grid, payload.geometry)
        self.payload = payload

    def make_checker(self, pose, q):
        self.record_support(pose)
        support = list(getattr(self.grid, 'inherited_support_poses', [])) + self.support_poses
        return self.payload.checker(self.grid, q, pose, support)


def payload_clearance(payload, q, obstacle_points, padding=.075, surface_hull=False):
    """Check a proposed carried pose against observed points, never object truth.

    Padding includes the carry map's voxel uncertainty and safety margin.
    This complements arm planning, which alone does not cover a held object.
    """
    box=payload.box(q)
    points=np.asarray(obstacle_points)
    local=(points-box.center)@box.axes
    near=np.all(np.abs(local)<=box.half+padding,axis=1)
    if not surface_hull:return not near.any()
    if not near.any():return True
    if payload.hull_equations is None:
        from scipy.spatial import ConvexHull
        payload.hull_equations=ConvexHull(payload.points_tool).equations
    equations=payload.hull_equations
    tool=local[near]+payload.center
    hit=np.all(tool@equations[:,:3].T+equations[:,3]<=padding,axis=1)
    return not hit.any()


def refine_held_payload(packet, geometry, prior_points_tool):
    """Expand attachment from a matching live connected surface near the tool.

    A poor match is rejected; it must not erase nearby furniture as payload.
    Retain prior surface points so a partial view cannot shrink collision shape.
    """
    from scipy.spatial import cKDTree
    live=reconstruct_held_points(packet,geometry)
    prior=np.asarray(prior_points_tool)
    distances=cKDTree(prior).query(live)[0]
    agreement=float(np.mean(distances<.04))
    shift=np.zeros(3);aligned=prior
    if agreement<.9 or float(np.quantile(distances,.99))>.07:
        # Acquisition can shift the vessel within the fingers. Fit a bounded
        # translation using live depth only; never substitute scene object poses.
        tree=cKDTree(prior)
        for _ in range(30):
            _,indices=tree.query(live-shift)
            updated=np.median(live-prior[indices],axis=0)
            if np.linalg.norm(updated-shift)<1e-5:break
            shift=updated
        residual=tree.query(live-shift)[0]
        if (np.linalg.norm(shift)>.06 or np.mean(residual<.02)<.9
                or float(np.quantile(residual,.99))>.025):
            raise ValueError(f'Observed held surface disagrees with attachment: {agreement:.3f}')
        aligned=prior+shift
    # Union retains the original extent as well as the adjusted observed extent.
    # A partial live view must never shrink collision geometry.
    combined=np.concatenate([prior,aligned,live])
    _,ids=np.unique(np.floor(combined/.005).astype(int),axis=0,return_index=True)
    return combined[ids],dict(live_points=len(live),agreement=agreement,max_residual_m=float(distances.max()),attachment_translation_m=shift.tolist())
