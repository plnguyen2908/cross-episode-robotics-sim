"""Robot-only FK/Jacobian and local IK, without scene-aware motion planning."""
from __future__ import annotations

import numpy as np


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite numbers')
    return result


class RobotKinematics:
    """Scratch-state queries; never modify the live robot or expose object state."""

    def __init__(self, model, data, profile):
        import mujoco
        self.model, self.data = model, data
        self.probe = mujoco.MjData(model)
        ns = profile.namespace
        arm = [model.joint(ns+n).id for n in profile.arm_joints]
        base = [model.joint(ns+n).id for n in profile.base_joints]
        self.arm_qpos = model.jnt_qposadr[arm]
        self.base_qpos = model.jnt_qposadr[base]
        self.arm_dofs = model.jnt_dofadr[arm]
        self.limits = model.jnt_range[arm].copy()
        self.site = model.site(ns+profile.tcp_site).id
        self.names = list(profile.arm_joints)

    def forward(self, joints=None, base=None):
        import mujoco
        joints = self.data.qpos[self.arm_qpos] if joints is None else vector(joints, 7, 'joints')
        base = self.data.qpos[self.base_qpos] if base is None else vector(base, 3, 'base')
        self.probe.qpos[:] = self.model.qpos0
        self.probe.qpos[self.arm_qpos] = joints
        self.probe.qpos[self.base_qpos] = base
        mujoco.mj_kinematics(self.model, self.probe)
        mujoco.mj_comPos(self.model, self.probe)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.probe, jp, jr, self.site)
        pose = np.eye(4)
        pose[:3, 3] = self.probe.site_xpos[self.site]
        pose[:3, :3] = self.probe.site_xmat[self.site].reshape(3, 3)
        return dict(tcp_pose_world=pose.tolist(), jacobian_world=np.vstack((jp, jr))[:, self.arm_dofs].tolist(),
                    joint_limits_rad=self.limits.tolist(), joint_names=self.names,
                    joints_rad=np.asarray(joints).tolist(), base_world=np.asarray(base).tolist(),
                    jacobian_rows=['vx', 'vy', 'vz', 'wx', 'wy', 'wz'])

    def solve(self, position, quaternion_wxyz):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        position = vector(position, 3, 'position')
        quat = vector(quaternion_wxyz, 4, 'quaternion_wxyz')
        if abs(np.linalg.norm(quat)-1.) > .001:
            raise ValueError('quaternion_wxyz must have unit norm')
        target = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        seed = self.data.qpos[self.arm_qpos].copy()

        def residual(q):
            pose = np.asarray(self.forward(q)['tcp_pose_world'])
            rotation = Rotation.from_matrix(target @ pose[:3, :3].T).as_rotvec()
            return np.r_[pose[:3, 3]-position, .3*rotation]

        low, high = self.limits[:, 0]+1e-6, self.limits[:, 1]-1e-6
        result = least_squares(residual, np.clip(seed, low, high), bounds=(low, high), max_nfev=100)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > .002 or np.linalg.norm(error[3:])/.3 > np.radians(1.):
            raise ValueError('Target has no converged local IK solution (2 mm / 1 degree)')
        return result.x


def camera_calibration(model, data, cameras, image_size):
    """Pinhole calibration: optical x right, y down, z forward; world meters."""
    height, width = image_size
    result = {}
    for name in cameras:
        cid = model.camera(name).id
        fy = .5*height/np.tan(np.radians(model.cam_fovy[cid])*.5)
        transform = np.eye(4)
        transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = data.cam_xpos[cid]
        result[name] = dict(intrinsics=[[float(fy), 0., width/2], [0., float(fy), height/2], [0., 0., 1.]],
                            world_from_optical=transform.tolist(), image_size=[height, width])
    return result
