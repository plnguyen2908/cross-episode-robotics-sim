"""Fallback docking for new generic targets without a visible counter face."""
import math
import numpy as np
from .shaped_docking import ShapedDocking


class HouseholdObjectDocking(ShapedDocking):
    def reset(self,target):
        super().reset(target);self.generic_fallback_attempted=False;self.close_dock_attempted=False

    def step(self,packet,tracking,now):
        pose=np.r_[tracking.T_map_base[:2,3],math.atan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0])]
        # Parent diagnostics may inspect the checker after surface fitting
        # fails before planning. Initialize it for this added backend only.
        if not hasattr(self,'checker') and tracking.accepted:
            self.checker=self.make_checker(pose,packet.arm.position)
        if (not self.terminal and self.path is None and tracking.accepted
                and not self.close_dock_attempted
                and np.linalg.norm(self.target.position[:2]-pose[:2])<1.6):
            from .local_scan import ScanConfig
            if np.allclose(packet.arm.position,ScanConfig().stowed_arm,atol=.05):
                self.close_dock_attempted=True
                # Generic packaged objects need room for their top approach,
                # not just an IK solution at the object's centroid. All routes
                # still use the inherited observed swept-shape planner.
                outward=pose[:2]-self.target.position[:2]
                outward/=max(np.linalg.norm(outward),1e-8)
                candidates=[]
                for distance in ((.65,.60,.55) if 'cookware' in getattr(self.target,'label','') else (.50,.55,.60)):
                    dock=np.r_[self.target.position[:2]+distance*outward,
                               math.atan2(-outward[1],-outward[0])]
                    pre=np.array([distance,0.,self.target.position[2]+.06])
                    if self.geometry.reach(pre,packet.arm.position) is not None:candidates.append(dock)
                direct=False
                if 'cookware' in getattr(self.target,'label',''):
                    checker=self.make_checker(pose,packet.arm.position)
                    for goal in candidates:
                        angle=(goal[2]-pose[2]+np.pi)%(2*np.pi)-np.pi
                        if abs(angle)<.05 and checker.swept(pose,goal):
                            self.checker=checker;self.initial=pose.copy();self.path=[goal];self.goal=goal
                            self.events.append(dict(event='cookware_checked_standoff',goal=goal.tolist()))
                            direct=True;break
                if not direct:self.plan_candidates(pose,packet.arm.position,candidates,source='household_close_grasp')
        result=super().step(packet,tracking,now)
        if self.terminal=='shape_route_unverified' and not self.generic_fallback_attempted:
            self.generic_fallback_attempted=True
            # The inherited target-centred SE(2) planner requires a measured
            # target, observed swept base clearance and calibrated reach IK.
            # It does not require a solid vertical cabinet below a tabletop.
            if self.plan(pose,packet.arm.position):
                self.terminal=None;self.curve=None;self.approaching=False;self.surface_outward=None
                self.events.append(dict(event='generic_target_dock',reason='counter_face_unavailable'))
                return dict(status='generic_object_dock_replan',forward=0.,yaw=0.)
        return result
