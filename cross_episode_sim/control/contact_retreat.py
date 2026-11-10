"""Short coordinated contact retreat from onboard pose and robot calibration.

This component plans against supplied observed geometry. Caller supplies floor
support validation from its observed map; missing coverage must trigger sensing,
never a truth-map fallback. No simulator objects are accepted by this class.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def tool_pose(geometry,q):
    method=getattr(geometry,'tcp_pose',None)
    return method(q) if method is not None else geometry.fk(q)[geometry.tcp_body]@geometry.tcp_local


class ContactRetreatPlanner:
    def __init__(self, geometry, collision_clear, support_clear, pose_clear=None, support_pose_clear=None, analytic_jacobian=False, fast_ik=True):
        self.geometry=geometry
        self.fast_ik=fast_ik
        self.analytic_jacobian=analytic_jacobian
        self.collision_clear=collision_clear
        self.support_clear=support_clear
        self.pose_clear=pose_clear;self.support_pose_clear=support_pose_clear

    def plan_oracle_segment(self, joints, tcp, *, distance, tcp_end, base_offset_end=None, base_yaw_end=0.):
        """Oracle endpoint-IK/interpolated sweep with sensor-only validators.

        Rejected interpolation is a preflight failure, never executed blindly.
        """
        from .coordinated_contact import CoordinatedContactSegment
        from .local_ik import local_ik
        offset = np.array([-distance,0.,0.]) if base_offset_end is None else np.asarray(base_offset_end,float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or abs(offset[2])>1e-8 or not 0<np.linalg.norm(offset)<=.3:
            raise ValueError('Invalid planar base offset')
        if not np.isfinite(base_yaw_end) or abs(base_yaw_end)>.15:
            raise ValueError('Invalid local base yaw')
        if base_yaw_end and (self.pose_clear is None or self.support_pose_clear is None):
            raise ValueError('Base yaw requires pose sweep guards')
        goal = np.asarray(tcp_end).copy()
        inv = Rotation.from_euler('z',-base_yaw_end).as_matrix()
        goal[:3,3] = inv@(goal[:3,3]-offset);goal[:3,:3]=inv@goal[:3,:3]
        end = local_ik(self.geometry,joints,goal)
        if end is None:
            lo,hi=self.geometry.limits.T
            def residual(q):
                pose=tool_pose(self.geometry,q)
                return np.r_[pose[:3,3]-goal[:3,3],.2*Rotation.from_matrix(goal[:3,:3]@pose[:3,:3].T).as_rotvec()]
            end=least_squares(residual,np.clip(joints,lo+.001,hi-.001),bounds=(lo+.001,hi-.001),max_nfev=80).x
        segment=CoordinatedContactSegment(np.asarray(joints),end,offset,base_yaw_end)
        turn=Rotation.from_matrix(tcp_end[:3,:3]@tcp[:3,:3].T).as_rotvec()
        rows=[]
        for u in np.linspace(0,1,int(np.ceil(max(np.linalg.norm(offset),abs(base_yaw_end)*.8)/.002))+1):
            xy,q,yaw=segment.sample(u)
            support=self.support_pose_clear(xy,yaw) if base_yaw_end else self.support_clear(xy)
            if not support:raise RuntimeError('contact_retreat_requires_observed_floor')
            R=Rotation.from_euler('z',yaw).as_matrix();actual=tool_pose(self.geometry,q)
            expected=(1-u)*tcp[:3,3]+u*tcp_end[:3,3]
            expected_R=Rotation.from_rotvec(u*turn).as_matrix()@tcp[:3,:3]
            if np.linalg.norm(R@actual[:3,3]+xy-expected)>.002 or np.linalg.norm(Rotation.from_matrix(expected_R@(R@actual[:3,:3]).T).as_rotvec())>.025:
                raise RuntimeError('contact_interpolation_tracking_error')
            if not (self.pose_clear(q,xy,yaw) if base_yaw_end else self.collision_clear(q,xy)):
                raise RuntimeError('contact_retreat_sweep_blocked')
            rows.append((xy.copy(),q.copy()))
        return rows

    def plan(self, joints, tcp, *, distance=.12, spacing=.002, tcp_end=None, base_offset_end=None, base_yaw_end=0.):
        if not 0<abs(distance)<=.3 or not 0<spacing<=.005:
            raise ValueError('Invalid local contact retreat extent')
        if not np.isfinite(base_yaw_end) or abs(base_yaw_end)>.15:raise ValueError('Invalid local base yaw')
        if base_yaw_end and (self.pose_clear is None or self.support_pose_clear is None):raise ValueError('Base yaw requires pose sweep guards')
        q=np.asarray(joints).copy();lo,hi=self.geometry.limits.T
        endpoint=np.array([-distance,0.,0.]) if base_offset_end is None else np.asarray(base_offset_end,float)
        if endpoint.shape!=(3,) or not np.isfinite(endpoint).all() or abs(endpoint[2])>1e-8 or not 0<np.linalg.norm(endpoint)<=.3:raise ValueError('Invalid planar base offset')
        rows=[]
        start_tcp=np.asarray(tcp,float)
        end_tcp=start_tcp if tcp_end is None else np.asarray(tcp_end,float)
        rotvec=Rotation.from_matrix(end_tcp[:3,:3]@start_tcp[:3,:3].T).as_rotvec()
        for d in np.linspace(0,distance,int(np.ceil(max(np.linalg.norm(endpoint),abs(base_yaw_end)*.8)/spacing))+1):
            offset=(d/distance)*endpoint;goal=np.asarray(tcp).copy()
            yaw=(d/distance)*base_yaw_end
            if not (self.support_pose_clear(offset,yaw) if base_yaw_end else self.support_clear(offset)):
                raise RuntimeError('contact_retreat_requires_observed_floor: '+str(getattr(self.support_clear,'missing_cells',[])))
            fraction=float(d/distance)
            goal[:3,3]=(1-fraction)*start_tcp[:3,3]+fraction*end_tcp[:3,3]-offset
            goal[:3,:3]=Rotation.from_rotvec(fraction*rotvec).as_matrix()@start_tcp[:3,:3]
            inverse_rotation=Rotation.from_euler('z',-yaw).as_matrix()
            goal[:3,3]=inverse_rotation@goal[:3,3];goal[:3,:3]=inverse_rotation@goal[:3,:3]
            def residual(x):
                # Preserve the qualified finite-difference arithmetic on contact paths.
                # Even roundoff-scale differences can select another redundant
                # arm solution after repeated contact steps.
                T=self.geometry.fk(x)[self.geometry.tcp_body]@self.geometry.tcp_local
                return np.r_[T[:3,3]-goal[:3,3],.2*Rotation.from_matrix(goal[:3,:3]@T[:3,:3].T).as_rotvec()]
            jac='2-point'
            if self.analytic_jacobian and hasattr(self.geometry,'tcp_pose_jacobian'):
                last_x=None;last_value=None
                def evaluate(x):
                    nonlocal last_x,last_value
                    if last_x is None or not np.array_equal(x,last_x):
                        T,J=self.geometry.tcp_pose_jacobian(x)
                        rotation=Rotation.from_matrix(goal[:3,:3]@T[:3,:3].T).as_rotvec()
                        value=np.r_[T[:3,3]-goal[:3,3],.2*rotation]
                        derivative=np.vstack([J[:3],-.2*so3_right_jacobian_inverse(rotation)@J[3:]])
                        last_x=x.copy();last_value=(value,derivative)
                    return last_value
                residual=lambda x:evaluate(x)[0]
                jac=lambda x:evaluate(x)[1]
            from .local_ik import local_ik
            from types import SimpleNamespace
            candidate=local_ik(self.geometry,q,goal) if self.fast_ik else None
            fitted=(SimpleNamespace(x=candidate) if candidate is not None else
                    least_squares(residual,np.clip(q,lo+.001,hi-.001),jac=jac,bounds=(lo+.001,hi-.001),max_nfev=80))
            error=residual(fitted.x)
            if np.linalg.norm(error[:3])>.002 or np.linalg.norm(error[3:])>.005:
                raise RuntimeError('contact_retreat_ik_unreachable')
            if not (self.pose_clear(fitted.x,offset,yaw) if base_yaw_end else self.collision_clear(fitted.x,offset)):
                raise RuntimeError('contact_retreat_sweep_blocked: '+str(getattr(getattr(self.collision_clear,'__self__',None),'last_rejection',None)))
            q=fitted.x;rows.append((offset.copy(),q.copy()))
        return rows


def occupied_support_validator(base_boxes, resolution=.1, observed_cells=(), observed_support=None):
    """Rasterize currently occupied support, as ObservedShapeChecker does.

    This permits only sub-cell repositioning inside already occupied support
    cells. Longer retreat requires a caller-provided observed floor map.
    """
    from itertools import product
    footprints=[]
    for box in base_boxes:
        corners=np.array(list(product([-1.,1.],repeat=3)))*box.half
        corners=corners@box.axes.T+box.center
        lo,hi=corners[:,:2].min(0),corners[:,:2].max(0)
        x=np.linspace(lo[0],hi[0],int(np.ceil((hi[0]-lo[0])/.01))+1)
        y=np.linspace(lo[1],hi[1],int(np.ceil((hi[1]-lo[1])/.01))+1)
        xx,yy=np.meshgrid(x,y);footprints.append(np.c_[xx.ravel(),yy.ravel()])
    points=np.concatenate(footprints)
    initial=set(map(tuple,np.floor(points/resolution).astype(int)))
    initial.update(map(tuple,observed_cells))
    from shapely.geometry import box as polygon_box
    from shapely.ops import unary_union
    support_region=unary_union([polygon_box(x*resolution,y*resolution,(x+1)*resolution,(y+1)*resolution) for x,y in initial])
    if observed_support is not None:support_region=support_region.union(observed_support)
    from shapely.prepared import prep
    covered=prep(support_region.buffer(1e-9))
    def clear(offset,yaw=0.):
        co,si=np.cos(yaw),np.sin(yaw);rotation=np.array([[co,-si],[si,co]])
        cells=set(map(tuple,np.floor((points@rotation.T+np.asarray(offset)[:2])/resolution).astype(int)))
        clear.missing_cells=sorted((x,y) for x,y in cells-initial if not covered.covers(polygon_box(x*resolution,y*resolution,(x+1)*resolution,(y+1)*resolution)))
        return not clear.missing_cells
    clear.support_region=support_region
    return clear


def base_offset_for_tool_motion(start_tcp, goal_tcp, yaw):
    """Translation keeping the tool relative to a base that also rotates."""
    rotation=Rotation.from_euler('z',yaw).as_matrix()
    offset=np.asarray(goal_tcp)[:3,3]-rotation@np.asarray(start_tcp)[:3,3]
    if abs(offset[2])>.01:raise ValueError('Base cannot absorb vertical tool motion')
    offset[2]=0.
    return offset


def transform_support_region(region, T_new_old):
    """Carry observed floor coverage through estimated planar base motion."""
    from shapely.affinity import affine_transform
    T=np.asarray(T_new_old)
    return affine_transform(region,[T[0,0],T[0,1],T[1,0],T[1,1],T[0,3],T[1,3]])


def contact_motion_duration(offset, yaw=0.):
    """Quintic motion at 50 Hz, bounded by qualified translation/yaw rates."""
    distance=float(np.linalg.norm(offset));turn=abs(float(yaw))
    # The previously validated 12 cm / 2 s repark peaks at 0.1125 m/s.
    # Faster short-contact profiles failed full-cycle qualification; retain
    # the measured two-second contact profile until separately requalified.
    duration=max(2.,1.875*distance/.1125,np.sqrt(5.774*distance/.2),
                 1.875*turn/.12,np.sqrt(5.774*turn/.3))
    return np.ceil(duration/.02)*.02


def so3_right_jacobian_inverse(rotation):
    r=np.asarray(rotation,float);theta=np.linalg.norm(r)
    x,y,z=r;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    coefficient=1/12+theta**2/720 if theta<1e-4 else (1-.5*theta/np.tan(.5*theta))/theta**2
    return np.eye(3)+.5*skew+coefficient*(skew@skew)
