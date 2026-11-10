"""Vertical fixture docking, separate from the qualified tabletop-pick backend.

Only onboard RGB-D, SLAM and robot calibration enter the pose proposal. A
reachable pre-contact pose is a docking hint, not proof of a successful grasp.
"""
import math
import numpy as np
from scipy.optimize import least_squares

from .shaped_docking import ShapedDocking, rotation
from .local_scan import ScanConfig
from ..control.sensor_hinged import fit_panel


def observed_fixture_normal(packet, target_base):
    clouds=[]
    for camera in packet.cameras.values():
        valid=camera.valid_depth.copy()
        if camera.robot_mask is not None:valid &= ~camera.robot_mask
        v,u=np.where(valid);z=camera.depth_m[v,u]
        xyz=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],
                  (v-camera.K[1,2])*z/camera.K[1,1],z]
        T=camera.T_base_optical
        clouds.append(xyz@T[:3,:3].T+T[:3,3])
    # Handles and flush door edges both need a vertical face, not an inferred
    # horizontal countertop below the point.
    panel=fit_panel(np.concatenate(clouds),target_base,radius=.4,min_gap=-.015)
    normal=panel.normal.copy()
    if normal@(-panel.anchor)<0:normal=-normal
    return normal


def fixture_dock_candidates(target, normal):
    normal=np.asarray(normal,float)[:2];normal=normal/np.linalg.norm(normal)
    for distance in (.60,.55,.65):
        yield np.r_[np.asarray(target)[:2]+distance*normal,
                    math.atan2(-normal[1],-normal[0])]


def frontal_reach(geometry,target,normal,q):
    """Robot-only IK for frontal pre-contact, with either jaw orientation."""
    normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
    z=-normal;y=np.cross([0.,0.,1.],z);y/=np.linalg.norm(y)
    R=np.column_stack([np.cross(y,z),y,z]);point=np.asarray(target)+.08*normal
    lo,hi=geometry.limits[:,0]+1e-5,geometry.limits[:,1]-1e-5
    seeds=[np.asarray(q),np.asarray(ScanConfig().stowed_arm)]
    for jaw in (R,R@np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])):
        def residual(joints):
            actual=geometry.tcp_pose(joints)
            return np.r_[actual[:3,3]-point,.2*(actual[:3,:3]-jaw).ravel()]
        for seed in seeds:
            solved=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=100)
            actual=geometry.tcp_pose(solved.x)
            if np.linalg.norm(actual[:3,3]-point)<.015 and np.linalg.norm(actual[:3,:3]-jaw)<.15:
                return solved.x
    return None


class FixtureDocking(ShapedDocking):
    def step(self,packet,tracking,now):
        # Reuse the established follower and motion guards after proposing a
        # fixture-specific dock. Never enter its countertop fitting branch.
        valid=(not self.terminal and tracking.accepted and 0<=now-packet.timestamp<=.25
               and np.allclose(packet.arm.position,ScanConfig().stowed_arm,atol=.05))
        if valid and self.path is None:
            T=tracking.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
            if self.started is None:self.started=now
            if np.linalg.norm(self.target.position[:2]-pose[:2])>1.6:
                self.plan_advance(pose,packet.arm.position)
            if self.path is None:
                try:
                    local=T[:3,:3].T@(self.target.position-T[:3,3])
                    normal=T[:3,:3]@observed_fixture_normal(packet,local)
                    candidates=[];solutions={}
                    for dock in fixture_dock_candidates(self.target.position,normal):
                        target=self.target.position.copy();target[:2]=rotation(dock[2]).T@(target[:2]-dock[:2])
                        outward=normal.copy();outward[:2]=rotation(dock[2]).T@normal[:2]
                        solution=frontal_reach(self.geometry,target,outward,packet.arm.position)
                        if solution is not None:candidates.append(dock);solutions[tuple(dock)]=solution
                    accepted=self.plan_candidates(pose,packet.arm.position,candidates,source='observed_fixture_face')
                    if accepted:
                        self.reach_joints=solutions[tuple(self.goal)]
                        self.events.append(dict(event='fixture_dock',normal=normal.tolist(),goal=self.goal.tolist()))
                except ValueError as exc:
                    accepted=False;self.events.append(dict(event='fixture_face_unobserved',reason=str(exc)))
                if not accepted:
                    accepted=self.plan_observe(pose,packet.arm.position) or self.plan_advance(pose,packet.arm.position)
                if not accepted:self.terminal='shape_route_unverified'
        return super().step(packet,tracking,now)
