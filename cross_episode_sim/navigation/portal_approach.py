"""Approach an observed door handle using the ordinary observed-map follower."""
import math
from types import SimpleNamespace
import numpy as np
from .semantic_navigation import TargetFollower
from .house_exploration import HouseExplorationConfig


class PortalApproach:
    def __init__(self,grid,target,config=None,*,align=True,geometry=None):
        self.follower=TargetFollower(grid,config or HouseExplorationConfig(),preferred_distance=.65)
        self.follower.detour_turn_in_place=True
        self.follower.target=SimpleNamespace(position=np.asarray(target),track_id=0)
        self.align=False;self.align_at_goal=align
        self.geometry=geometry;self.undock=None;self.undock_attempted=False
        self._alignment_packet=None;self._alignment_checker=None
        self.rejected_view_headings=[]

    def retry_view(self,tracking):
        """Try a different local viewing angle after failed fresh perception."""
        if len(self.rejected_view_headings)>=4:return False
        T=tracking.T_map_base
        self.rejected_view_headings.append(math.atan2(T[1,0],T[0,0]))
        self.align=True;self.follower.terminal=None
        return True

    def step(self,packet,tracking,now):
        follower=self.follower
        if not tracking.accepted:return follower._stop('tracking_lost')
        follower.camera=packet.cameras['navigation'];follower.base_pose=tracking.T_map_base
        xy=tracking.T_map_base[:2,3];yaw=math.atan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0])
        if self.undock is not None:
            follower.layers(xy)
            result=self.undock.step(packet,tracking,follower.free,now)
            if result['status']=='undock_complete':
                self.undock=None;follower.terminal=None;follower.path=[]
                follower.velocity[:]=0;follower.last_progress=xy.copy();follower.progress_time=now
                return follower._stop('door_approach_departure_ready')
            if self.undock.error:follower.terminal=result['status']
            return result
        if self.align:
            from .object_view import visible_object_heading
            heading=visible_object_heading(tracking.T_map_base,follower.target.position,follower.camera,xy,robot_margin_fraction=.10,rejected_headings=self.rejected_view_headings)
            if heading is None:
                follower.stop_detail=dict(reason='handle_outside_calibrated_camera_view')
                return follower._stop('handle_alignment_blocked')
            error=math.atan2(math.sin(heading-yaw),math.cos(heading-yaw))
            if abs(error)<.015:return follower._stop('handle_view_ready')
            follower.layers(xy);follower._depth(packet.cameras['navigation'])
            command=np.array([0.,np.clip(error*1.5,-.3,.3)])
            if self.geometry is None:
                clear=follower._command_clear(xy,yaw,command)
            else:
                # Close fixture approaches already require robot-shape planning.
                # The room-transit circle would reject valid rotations here.
                from .shaped_docking import ShapedDocking
                pose=np.r_[xy,yaw]
                if packet is not self._alignment_packet:
                    dock=ShapedDocking(follower.grid,self.geometry)
                    self._alignment_checker=dock.make_checker(pose,packet.arm.position)
                    self._alignment_packet=packet
                checker=self._alignment_checker
                end=pose.copy();end[2]+=np.sign(error)*min(abs(error),abs(command[1])*.25+command[1]**2/(2*.8))
                clear=checker.swept(pose,end)
                if not clear:follower.stop_detail=checker.detail
            if not clear:return follower._stop('handle_alignment_blocked')
            return dict(status='align_handle',forward=0.,yaw=float(command[1]))
        result=follower.step(packet,tracking,now)
        if result['status']=='target_not_reachable_yet' and follower.max_distance < 1.5:
            # Observation stand-off is not a grasp pose; opening verifies reach.
            # Widen the view ring without changing occupancy or swept checks.
            follower.max_distance=1.5
            follower.terminal=None;follower.path=[]
            follower.events.append(dict(time=now,event='portal_view_ring_expanded',maximum_distance=1.5))
            return follower._stop('door_approach_wider_view')
        if result['status']=='departure_blocked' and self.geometry is not None and not self.undock_attempted:
            from .shaped_docking import ShapedDocking
            from .observed_undock import ObservedUndock
            self.undock=ObservedUndock(ShapedDocking(follower.grid,self.geometry))
            self.undock_attempted=True;follower.terminal=None
            return follower._stop('door_approach_checked_departure')
        if result['status']=='arrived_frontier':
            if not self.align_at_goal:return follower._stop('portal_crossed')
            self.align=True
        return result
