"""Two onboard observations: wrist and forward navigation mast."""
import mujoco
import numpy as np

# Default Franka tuck reaches approximately 1.53 m above the floor.
NAVIGATION_CAMERA_HEIGHT_M = 1.70  # base-relative; about 17 cm clearance
NAVIGATION_MAST_X_M = -.40  # rear extension plate, clear of the Franka elbow sweep

ONBOARD_CAMERAS = ('robot_0/navigation_cam', 'robot_0/wrist_cam')


def install_onboard_cameras(spec, profile, namespace='robot_0/'):
    """Keep the existing wrist optics; add a rigid, collidable mast on the base."""
    joint_name = namespace + profile.base_joints[-1]
    joint = next((j for j in spec.joints if j.name == joint_name), None)
    if joint is None:
        raise ValueError(f'Missing camera base joint {joint_name}; robot joints: '+str([j.name for j in spec.joints if namespace in j.name]))
    base = joint.parent
    if spec.camera(namespace + 'navigation_cam') is None:
        # Centered behind the arm, above its default tucked envelope. All poses
        # are local to the yawing base, never aligned to a task object.
        mast = base.add_body(name=namespace+'navigation_mast', pos=[NAVIGATION_MAST_X_M, 0., 0.])
        # 245 x 80 x 6 mm plate overlaps the rear of the 481 mm base and
        # extends to x=-.425 m. The pole sits 25 mm inside its rear edge.
        mast.add_geom(name=namespace+'navigation_mast_plate', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[.0975,0,.317], size=[.1225,.04,.003], mass=.318,
                      rgba=[.18,.18,.20,1], contype=1, conaffinity=1, group=1)
        mast.add_geom(name=namespace+'navigation_mast_pole', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0,0,.32, 0,0,NAVIGATION_CAMERA_HEIGHT_M], size=[.012,0,0], mass=.25,
                      rgba=[.18,.18,.20,1], contype=1, conaffinity=1, group=1)
        mast.add_geom(name=namespace+'navigation_camera_housing', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0,0,NAVIGATION_CAMERA_HEIGHT_M], size=[.025,.045,.025], mass=.08,
                      rgba=[.08,.08,.09,1], contype=1, conaffinity=1, group=1)
        # MuJoCo cameras look down local -Z. Base +X is forward.
        angle=np.radians(30.)
        forward=np.array([np.cos(angle),0,-np.sin(angle)])
        right=np.array([0.,-1.,0.]); up=np.cross(right,forward)
        from scipy.spatial.transform import Rotation
        quat=Rotation.from_matrix(np.column_stack((right,up,-forward))).as_quat(scalar_first=True)
        mast.add_camera(name=namespace+'navigation_cam',pos=[.03,0,NAVIGATION_CAMERA_HEIGHT_M],quat=quat,fovy=75)
    if spec.camera(namespace+'wrist_cam') is None:
        # Embodiments without an authored wrist camera get a hand-fixed mount.
        # Franka retains its existing authored wrist camera unchanged.
        site=spec.site(namespace+profile.tcp_site)
        site.parent.add_camera(name=namespace+'wrist_cam',pos=site.pos,quat=site.quat,fovy=75)


def onboard_camera_names(profile):
    return (profile.namespace+'navigation_cam', profile.namespace+'wrist_cam')


def validate_onboard_cameras(cameras):
    unknown = set(cameras)-set(ONBOARD_CAMERAS)
    if unknown:
        raise ValueError('Robot observations support only navigation_cam and wrist_cam; got '+str(sorted(unknown)))
