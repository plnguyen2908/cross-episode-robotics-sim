"""Validate a basket placement approach at a proposed base pose before moving.

Inputs are calibrated robot geometry, observed scene/payload, estimated poses
and measured-footprint proposals. No scene truth or simulator motion is used.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .observed_grasp_planner import ObservedGraspPlanner, merge_voxels


def in_dock_frame(pose, map_from_source):
    dock = np.eye(4)
    dock[:3, :3] = Rotation.from_euler('z', pose[2]).as_matrix()
    dock[:2, 3] = pose[:2]
    return np.linalg.inv(dock) @ map_from_source


class BasketDockProbe:
    def __init__(self, model, data, geometry, points, tool_points, rows, output):
        from .observed_arm import robot_calibration
        from .planner_cache import sensor_arm_planner
        self.model, self.data, self.geometry = model, data, geometry
        self.points, self.tool, self.rows = points, tool_points, rows
        self.calibration = robot_calibration(model, data, output/'probe_robot')
        self.arm = sensor_arm_planner(self.calibration, collision_cache={'cuboid':8192, 'mesh':2})
        self.fixed = [[*s['center'],s['radius']] for s in self.calibration['collision_spheres'].get('base',[])]

    def evaluate(self, pose, map_from_source, target, q):
        from curobo._src.geom.types import Cuboid, SceneCfg
        from .fixture_approach import calibrated_finger_planner
        started = time.monotonic()
        transform = in_dock_frame(pose, map_from_source)
        points = self.points @ transform[:3,:3].T + transform[:3,3]
        # Floor support belongs to the base-map planner. Rasterizing the floor
        # below a hypothetical base as a raised arm obstacle falsely collides
        # with the base's calibrated ground-contact box.
        points = points[(np.linalg.norm(points,axis=1)<1.6)&(points[:,2]>.03)]
        vox = np.unique(np.floor(points/.025).astype(int),axis=0)
        boxes = [Cuboid(name=f'probe_{i}',pose=[*((lo+hi+1)*.0125),1,0,0,0],
                        dims=((hi-lo+1)*.025+.003).tolist())
                 for i,(lo,hi) in enumerate(merge_voxels(vox))]
        self.arm.planner.update_world(SceneCfg(cuboid=boxes))
        planner = calibrated_finger_planner(self.arm,self.geometry,boxes,self.model,self.data,
            fixed_floor_support_spheres=self.fixed,allow_curobo_fallback=True)
        # One contact per orientation near this proposed dock's target.
        seen=set(); rows=[]
        ranked=sorted(self.rows,key=lambda r:np.linalg.norm(
            map_from_source[:3,:3]@np.asarray(r['contact'])+map_from_source[:3,3]-target))
        for row in ranked:
            if np.linalg.norm(map_from_source[:3,:3]@np.asarray(row['contact'])+map_from_source[:3,3]-target)>.035:
                continue
            if row['group'] not in seen:
                seen.add(row['group']);rows.append(row)
        attempts=0;errors=[]
        for clearance in (.05,.075,.10):
            for row in rows:
                goal=transform @ np.asarray(row['tcp']);goal[2,3]+=clearance
                attempts+=1
                try:
                    approach=planner.plan(q,goal[:3,3]+[0,0,.08],goal[:3,:3])
                    lower=planner.plan(approach[-1],goal[:3,3],goal[:3,:3])
                    for joints in np.concatenate([approach,lower]):
                        tcp=self.geometry.tcp_pose(joints)
                        xyz=self.tool[::max(1,len(self.tool)//150)]@tcp[:3,:3].T+tcp[:3,3]
                        if not ObservedGraspPlanner.world_clear(planner,np.c_[xyz,np.full(len(xyz),.006)][None],margin=.002):
                            raise RuntimeError('held surface blocks approach')
                    return dict(success=True,attempts=attempts,wall_seconds=time.monotonic()-started,
                        contact_source=row['contact'],tcp_source=row['tcp'],release_clearance_m=clearance,
                        reason='native arm and held-surface approach/lower planned at proposed dock')
                except RuntimeError as exc:
                    errors.append(str(exc))
        return dict(success=False,attempts=attempts,wall_seconds=time.monotonic()-started,
                    reason='no_placement_approach_at_dock',errors=errors[-3:])
