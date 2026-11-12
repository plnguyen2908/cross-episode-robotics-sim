"""One continuous sensor tracker with explicit stage-coordinate views.

Only the coordinate representation changes at a legacy stage boundary. The
underlying tracker, keyframes and global pose are never reset or restored.
"""
from dataclasses import replace
import json
import numpy as np
from pathlib import Path
from cross_episode_sim.runtime import artifacts


class TrackingSession:
    def __init__(self,mapper,*,global_frame=False):
        self.mapper=mapper;self.last_result=None;self.global_frame=global_frame
        # Stage views own only their local diagnostics. Route memory needs the
        # continuous sensor trajectory, including nested door-opening stages.
        self.trajectory=[];self.timestamps=[]
        self.map_alignment=np.eye(4);self.map_localizer=None;self.map_localization_timestamp=None
        self.planar_motion=None
        self.map_localization_events=[]
    def configure_frozen_localization(self,grid):
        """Use observed walls for SE(2) correction; retain the same RGB-D tracker."""
        if not self.global_frame:raise ValueError('Frozen-map localization requires the shared map frame')
        from cross_episode_sim.navigation.frozen_map_localization import FrozenMapLocalizer
        self.map_localizer=FrozenMapLocalizer(grid)
        self.map_localization_timestamp=None
        if self.planar_motion is None:
            from .planar_motion import PlanarVisualMotion
            self.planar_motion=PlanarVisualMotion()
    def record_control(self,forward,yaw,dt,lateral=0.):
        if self.planar_motion is not None:self.planar_motion.command(forward,yaw,dt,lateral)
        if hasattr(self.mapper,'record_control'):self.mapper.record_control(forward,yaw,dt,lateral)
    def update(self,camera,timestamp):
        if self.last_result is not None and timestamp==self.last_result.timestamp:
            return self.last_result
        result=self.mapper.update(camera,timestamp)
        if result.accepted:
            pose=(self.planar_motion.observe(result.T_map_base) if self.planar_motion is not None
                  else self.map_alignment@result.T_map_base)
            if self.map_localizer is not None and (self.map_localization_timestamp is None or timestamp-self.map_localization_timestamp>=.5-1e-8):
                from cross_episode_sim.navigation.success_graph_transit import se2
                corrected,detail=self.map_localizer.correct(camera,se2(pose))
                if detail['accepted']:
                    adjusted=np.eye(4);c,s=np.cos(corrected[2]),np.sin(corrected[2])
                    adjusted[:2,:2]=[[c,-s],[s,c]];adjusted[:2,3]=corrected[:2]
                    self.map_alignment=adjusted@np.linalg.inv(result.T_map_base);pose=adjusted
                    if self.planar_motion is not None:self.planar_motion.correct(adjusted)
                self.map_localization_timestamp=timestamp
                self.map_localization_events.append(dict(timestamp=timestamp,**detail))
            result=replace(result,T_map_base=pose)
        self.last_result=result
        if result.accepted and (not self.timestamps or timestamp>self.timestamps[-1]):
            self.timestamps.append(timestamp);self.trajectory.append(result.T_map_base.copy())
        return result
    def fixed_reference(self, **kwargs):return self.mapper.fixed_reference(**kwargs)
    def matched_keyframes(self):return self.mapper.matched_keyframes()
    def view(self):return StageTrackingView(self)


class StageTrackingView:
    def __init__(self,session):
        self.session=session;self.trajectory=[];self.timestamps=[]
        last=session.last_result
        if last is not None and not last.accepted:
            raise RuntimeError('Tracker lost: relocalize before a stage handoff')
        self.anchor=np.eye(4) if last is None or session.global_frame else last.T_map_base.copy()
        self.inverse=np.linalg.inv(self.anchor)
    def fixed_reference(self, **kwargs):return self.session.fixed_reference(**kwargs)
    def matched_keyframes(self):return self.session.matched_keyframes()
    @property
    def last_timestamp(self):
        last=self.session.last_result
        return None if last is None else last.timestamp
    def update(self,camera,timestamp):
        result=self.session.update(camera,timestamp)
        pose=None if result.T_map_base is None else self.inverse@result.T_map_base
        if result.accepted and (not self.timestamps or timestamp>self.timestamps[-1]):
            self.timestamps.append(timestamp);self.trajectory.append(pose.copy())
        return replace(result,T_map_base=pose)
    def save(self,directory):
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        artifacts.save_array(out/'estimated_base_poses.npy',np.asarray(self.trajectory))
        artifacts.save_array(out/'estimated_timestamps.npy',np.asarray(self.timestamps))
        artifacts.write_text(out/'mapper.json',json.dumps(dict(backend='continuous_rgbd',fuse_dense=False,
            global_anchor=self.anchor.tolist(),tracker_reinitialized=False),indent=2))
