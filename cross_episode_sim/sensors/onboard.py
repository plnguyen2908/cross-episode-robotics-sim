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
    robot_mask: np.ndarray | None = None
    raw_depth_m: np.ndarray | None = None

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


def depth_validity(depth, max_depth_m=8.):
    """Reject isolated inconsistent returns without inventing replacement depth.

    One agreeing 8-connected neighbor preserves thin edges/poles. Isolated
    returns become unknown, not free space. Keep raw_depth_m for audit/export.
    """
    depth=np.asarray(depth)
    valid=np.isfinite(depth)&(depth>0)&(depth<=max_depth_m)
    if min(depth.shape)<3:return valid
    supported=np.zeros(depth.shape,bool)
    tolerance=np.maximum(.02,.02*depth)
    h,w=depth.shape
    for dr,dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        target=(slice(max(0,-dr),min(h,h-dr)),slice(max(0,-dc),min(w,w-dc)))
        neighbor=(slice(max(0,dr),min(h,h+dr)),slice(max(0,dc),min(w,w+dc)))
        with np.errstate(invalid='ignore'):
            supported[target] |= valid[neighbor] & (np.abs(depth[target]-depth[neighbor])<=tolerance[target])
    return valid & supported


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
        # Multisample resolve blends foreground/background depth at robot
        # silhouettes, producing phantom geometry outside calibrated bounds.
        # Render synchronized RGB/depth without MSAA; this changes rasterization
        # only, not camera pose, scene geometry, collision, or physical state.
        model.vis.quality.offsamples=0
        self._renderer=mujoco.Renderer(model,height=height,width=width)
        self._option=mujoco.MjvOption();self._option.geomgroup[0]=0;self._option.sitegroup[:]=0

    def _joints(self,names):
        ids=[self._model.joint(n).id for n in names]
        return JointObservation(names,self._data.qpos[self._model.jnt_qposadr[ids]].copy(),
                               self._data.qvel[self._model.jnt_dofadr[ids]].copy())

    def _body_transform(self,bid):
        m,d=self._model,self._data
        chain=[];bid=int(bid)
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
        return T

    def _extrinsic(self,cid):
        m=self._model
        T=self._body_transform(m.cam_bodyid[cid])@transform(m.cam_pos[cid],m.cam_quat[cid])
        return T@np.diag([1.,-1.,-1.,1.])

    def _robot_mask(self,cid,K,depth=None):
        # Project known robot-only visual bounds via encoder FK. No world poses,
        # scene labels or simulator segmentation are consulted.
        from PIL import Image,ImageDraw,ImageFilter
        import itertools
        canvas=Image.new('L',(self.width,self.height));draw=ImageDraw.Draw(canvas)
        optical=np.linalg.inv(self._extrinsic(cid));m=self._model
        signs=np.array(list(itertools.product([-1,1],repeat=3)))
        depth_mask=np.zeros((self.height,self.width),bool)
        if depth is not None:
            v,u=np.indices(depth.shape)
            measured=np.stack([(u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth],axis=-1)
        def box(T,center,half):
            corners=center+signs*half
            xyz=corners@T[:3,:3].T+T[:3,3]
            if depth is not None:
                # Restrict the exact 3-D depth test to the projected box bounds.
                # Boxes crossing the camera plane retain a full-image test.
                if xyz[:,2].max()<=0:return
                x0,y0,x1,y1=0,0,self.width,self.height
                if xyz[:,2].min()>.02:
                    projected=xyz@K.T;uv=projected[:,:2]/projected[:,2:3]
                    x0,y0=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0])
                    x1,y1=np.minimum(np.ceil(uv.max(0)).astype(int)+2,[self.width,self.height])
                    if x1<=x0 or y1<=y0:return
                local=(measured[y0:y1,x0:x1]-T[:3,3])@T[:3,:3]
                depth_mask[y0:y1,x0:x1] |= np.all(np.abs(local-center)<=half,axis=-1)
                return
            if np.any(xyz[:,2]<=.02):return
            uv=xyz@K.T;uv=uv[:,:2]/uv[:,2:3]
            # Project the convex hull of the box, not an image-sized rectangle.
            from scipy.spatial import ConvexHull,QhullError
            try:order=ConvexHull(uv).vertices
            except QhullError:return
            draw.polygon([tuple(p) for p in uv[order]],fill=255)
        # All calibrated robot visuals, including base and camera mast. Passive
        # finger links remain covered by the conservative gripper envelope.
        for bid in range(m.nbody):
            name=m.body(bid).name or ''
            if not name.startswith('robot_0/') or ('/gripper/' in name and name!='robot_0/gripper/base'):continue
            T=self._body_transform(bid)
            for gid in range(m.body_geomadr[bid],m.body_geomadr[bid]+m.body_geomnum[bid]):
                if m.geom_group[gid]==0:continue
                G=T@transform(m.geom_pos[gid],m.geom_quat[gid])
                box(optical@G,m.geom_aabb[gid,:3],m.geom_aabb[gid,3:]+.02)
        # Conservative known gripper envelope covers passive finger linkages.
        T=self._body_transform(m.body('robot_0/gripper/base').id)
        box(optical@T,np.array([0.,0.,.08]),np.array([.10,.10,.13]))
        return depth_mask if depth is not None else np.asarray(canvas.filter(ImageFilter.MaxFilter(5)))>0


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
            raw_depth=depth.copy()
            valid=depth_validity(depth,self.max_depth_m)
            depth[~valid]=np.nan
            frames[key]=RGBDCamera(self.CAMERA_NAMES[key]+'_optical',rgb,depth,valid,
                                  self._K[key].copy(),np.zeros(5),self._extrinsic(cid),
                                  self._robot_mask(cid,self._K[key],depth),raw_depth)
        return SensorPacket(float(self._data.time),frames,self._joints(self.ARM_NAMES),self._joints(self.GRIPPER_NAMES))

    def close(self):self._renderer.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
