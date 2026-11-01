"""Existing onboard cameras and encoder-like readings; no environment truth output.

Only the simulator bridge owns model/data. Downstream code receives SensorPacket.
Camera extrinsics use static robot geometry and allowed joint readings, not
world-frame body/camera transforms. Optical axes are right, down, forward.
"""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass(frozen=True)
class JointObservation:
    names: tuple
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class RGBDCamera:
    frame_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_depth: np.ndarray
    K: np.ndarray
    distortion: np.ndarray
    T_base_optical: np.ndarray

    def camera_info(self):
        """ROS CameraInfo-compatible values, without a ROS dependency."""
        h,w=self.depth_m.shape
        P=np.zeros((3,4));P[:3,:3]=self.K
        return dict(frame_id=self.frame_id,width=w,height=h,distortion_model='plumb_bob',
                    d=self.distortion.tolist(),k=self.K.ravel().tolist(),
                    r=np.eye(3).ravel().tolist(),p=P.ravel().tolist())


@dataclass(frozen=True)
class SensorPacket:
    timestamp: float
    cameras: dict
    arm: JointObservation
    gripper: JointObservation
    base_frame: str = 'robot_0/base'


def transform(position, quaternion):
    R=np.empty(9);mujoco.mju_quat2Mat(R,np.asarray(quaternion))
    T=np.eye(4);T[:3,:3]=R.reshape(3,3);T[:3,3]=position
    return T


class OnboardSensorInterface:
    """Own this bridge in the simulator; pass only capture() results to policies.

    No simulator stepping, controller creation, mount changes or SLAM is done here.
    Gripper state is two driver-joint angles, not an invented aperture measurement.
    """
    CAMERA_NAMES={'navigation':'robot_0/navigation_cam','wrist':'robot_0/wrist_cam'}
    ARM_NAMES=tuple(f'robot_0/fr3_joint{i}' for i in range(1,8))
    GRIPPER_NAMES=('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')

    def __init__(self,model,data,width=640,height=480,max_depth_m=8.):
        if width<=0 or height<=0 or max_depth_m<=0:raise ValueError('Invalid sensor dimensions/range')
        self._model,self._data=model,data
        self.width,self.height,self.max_depth_m=width,height,max_depth_m
        self._base=model.body('robot_0/base').id
        self._allowed={model.joint(n).id for n in self.ARM_NAMES+self.GRIPPER_NAMES}
        self._cameras={k:model.camera(n).id for k,n in self.CAMERA_NAMES.items()}
        self._K={}
        for key,cid in self._cameras.items():
            if model.cam_mode[cid]!=mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
                raise ValueError('Only robot-mounted fixed cameras are supported')
            if np.any(model.cam_sensorsize[cid]):
                raise ValueError('Physical-sensor intrinsics need explicit support; refusing fovy approximation')
            f=.5*height/np.tan(np.deg2rad(model.cam_fovy[cid]/2))
            self._K[key]=np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
        # Rendering-only buffer sizes; leave geometry, camera settings and physics untouched.
        model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
        model.vis.global_.offheight=max(height,model.vis.global_.offheight)
        self._renderer=mujoco.Renderer(model,height=height,width=width)
        self._option=mujoco.MjvOption();self._option.geomgroup[0]=0;self._option.sitegroup[:]=0

    def _joints(self,names):
        ids=[self._model.joint(n).id for n in names]
        return JointObservation(names,self._data.qpos[self._model.jnt_qposadr[ids]].copy(),
                               self._data.qvel[self._model.jnt_dofadr[ids]].copy())

    def _extrinsic(self,cid):
        m,d=self._model,self._data
        chain=[];bid=int(m.cam_bodyid[cid])
        while bid!=self._base:
            if bid==0:raise ValueError('Camera is not attached to the robot base')
            chain.append(bid);bid=int(m.body_parentid[bid])
        T=np.eye(4)
        for bid in reversed(chain):
            T=T@transform(m.body_pos[bid],m.body_quat[bid])
            for jid in range(m.body_jntadr[bid],m.body_jntadr[bid]+m.body_jntnum[bid]):
                if jid not in self._allowed:raise ValueError('Non-proprioceptive joint in camera chain')
                angle=float(d.qpos[m.jnt_qposadr[jid]]-m.qpos0[m.jnt_qposadr[jid]])
                if m.jnt_type[jid]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('Unsupported camera-chain joint')
                q=np.r_[np.cos(angle/2),m.jnt_axis[jid]*np.sin(angle/2)]
                A=transform(np.zeros(3),q);pivot=m.jnt_pos[jid]
                A[:3,3]=pivot-A[:3,:3]@pivot;T=T@A
        T=T@transform(m.cam_pos[cid],m.cam_quat[cid])
        return T@np.diag([1.,-1.,-1.,1.])

    def capture(self):
        # mj_step may leave position-dependent render caches at the previous step.
        # Refresh kinematics only: no stepping, forces, control or qpos changes.
        mujoco.mj_kinematics(self._model,self._data)
        mujoco.mj_camlight(self._model,self._data)
        r=self._renderer;frames={}
        for key,cid in self._cameras.items():
            r.disable_depth_rendering();r.update_scene(self._data,camera=cid,scene_option=self._option)
            rgb=r.render().copy()
            try:
                r.enable_depth_rendering();depth=r.render().copy().astype(np.float32)
            finally:r.disable_depth_rendering()
            valid=np.isfinite(depth)&(depth>0)&(depth<=self.max_depth_m)
            depth[~valid]=np.nan
            frames[key]=RGBDCamera(self.CAMERA_NAMES[key]+'_optical',rgb,depth,valid,
                                  self._K[key].copy(),np.zeros(5),self._extrinsic(cid))
        return SensorPacket(float(self._data.time),frames,self._joints(self.ARM_NAMES),self._joints(self.GRIPPER_NAMES))

    def close(self):self._renderer.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
