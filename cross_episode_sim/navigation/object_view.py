"""Plan a forward object view from estimated pose and own camera calibration."""
import math
import numpy as np
from scipy.ndimage import binary_dilation


def robot_blocks_view(camera,point_base,margin=.02):
    """Sparse camera-to-target ray test against calibrated robot bounds.

    Works when full image masking is disabled. Only robot geometry and encoders
    are involved; the navigation map and scene points are not modified.
    """
    origin=np.asarray(camera.T_base_optical[:3,3]);delta=np.asarray(point_base)-origin
    length=np.linalg.norm(delta)
    if length<1e-8:return False
    for T,center,half in getattr(camera,'robot_bounds_base',()):
        start=(origin-T[:3,3])@T[:3,:3]-center
        end=delta@T[:3,:3];half=np.asarray(half)+margin
        # The camera housing may contain the optical origin.
        if np.all(np.abs(start)<=half):continue
        low=.03/length;high=1.-.01/length
        for axis in range(3):
            if abs(end[axis])<1e-10:
                if abs(start[axis])>half[axis]:low=2.;break
            else:
                a=(-half[axis]-start[axis])/end[axis];b=(half[axis]-start[axis])/end[axis]
                low=max(low,min(a,b));high=min(high,max(a,b))
        if low<=high:return True
    return False


def visible_object_heading(base_pose,target,camera,xy=None,*,robot_margin_fraction=.03,rejected_headings=()):
    """Return the closest forward-facing heading with no robot self-occlusion.

    The arm must remain in the checked stowed posture. Its onboard silhouette
    is then rigid relative to the mast camera. Environment occlusion still
    requires fresh RGB-D confirmation after arrival; this is not a grasp pose.
    """
    point=np.asarray(target);pose=np.asarray(base_pose).copy()
    if xy is not None:pose[:2,3]=xy
    bearing=math.atan2(point[1]-pose[1,3],point[0]-pose[0,3])
    old_yaw=math.atan2(pose[1,0],pose[0,0])
    h,w=camera.depth_m.shape;margin=max(4,int(min(h,w)*robot_margin_fraction))
    robot=np.zeros((h,w),bool) if camera.robot_mask is None else binary_dilation(camera.robot_mask,iterations=margin)
    for offset in sorted(np.linspace(-.75,.75,31),key=abs):
        heading=bearing+offset;change=heading-old_yaw;c,s=math.cos(change),math.sin(change)
        if any(abs(math.atan2(math.sin(heading-old),math.cos(heading-old)))<.09 for old in rejected_headings):continue
        pose[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])@base_pose[:3,:3]
        T=np.linalg.inv(pose@camera.T_base_optical)
        q=T[:3,:3]@point+T[:3,3]
        if q[2]<=.1:continue
        pixel=camera.K@q;u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
        if margin<=u<w-margin and margin<=v<h-margin and not robot[v,u]:
            point_base=camera.T_base_optical[:3,:3]@q+camera.T_base_optical[:3,3]
            if robot_blocks_view(camera,point_base):continue
            return float(math.atan2(math.sin(heading),math.cos(heading)))
    return None
