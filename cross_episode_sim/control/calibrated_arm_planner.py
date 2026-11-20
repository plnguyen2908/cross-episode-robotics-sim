"""Resolve arm covering-sphere overlap using calibrated collision-shape boxes.

Boxes enclose the original robot collision geometry. This changes neither the
observed world nor clearance margins; self collision stays in the native check.
"""
import numpy as np
from .observed_hinged_planner import ObservedHingedPlanner
from .observed_grasp_planner import ObservedGraspPlanner


class CalibratedArmPlanner(ObservedHingedPlanner):
    def __init__(self,*args,arm_body_ids,box_link_names=None,**kwargs):
        super().__init__(*args,**kwargs)
        cfg=self.arm.planner.kinematics.config.kinematics_config
        self.box_geometry=self.finger_geometry or self.geometry
        self.arm_box_names={shape[0] for body in arm_body_ids for shape in self.box_geometry.geoms[body]}
        if not self.arm_box_names:raise ValueError('Missing calibrated arm collision boxes')
        names=box_link_names or [f'fr3_link{i}' for i in range(8)]
        indices=[cfg.get_sphere_index_from_link_name(name).cpu().numpy()
                 for name in names if name in cfg.link_name_to_idx_map]
        self.arm_box_spheres=np.concatenate(indices).astype(int)

    def world_clear(self,spheres,margin=.005,base_offset=None,base_yaw=0.):
        if getattr(self,'ik_only',False):return True      # user rule: IK found is enough (2026-11-18)
        if self._checked_q is None:
            return super().world_clear(spheres,margin,base_offset,base_yaw)
        checked=np.asarray(spheres).copy()
        checked[:,self.arm_box_spheres,3]=-1
        if not super().world_clear(checked,margin,base_offset,base_yaw):return False
        from dataclasses import replace
        offset=np.zeros(3) if base_offset is None else np.asarray(base_offset)
        c,s=np.cos(base_yaw),np.sin(base_yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        for i,q in enumerate(np.atleast_2d(self._checked_q)):
            arm=np.asarray(spheres)[i:i+1,self.arm_box_spheres,:]
            if ObservedGraspPlanner.world_clear(self,arm,margin):continue
            for box in getattr(self,'box_geometry',self.geometry).boxes(q):
                if box.name not in self.arm_box_names:continue
                moved=replace(box,center=R@box.center+offset,axes=R@box.axes)
                if not self.world_clear_box(moved,margin):
                    self.last_rejection=dict(self.last_rejection,robot_part=box.name,
                                             narrow_phase='calibrated_arm_box')
                    return False
        return True
