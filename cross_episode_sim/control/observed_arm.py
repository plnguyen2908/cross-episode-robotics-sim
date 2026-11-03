"""cuRobo arm planning from robot calibration and observed occupancy only."""
import numpy as np
from scipy.spatial.transform import Rotation


def robot_calibration(model,data,output):
    """Simulator boundary: export only robot geometry and robot joint readings.

    Robot base is deliberately zeroed. No scene transforms enter the export.
    Passive gripper linkage angles are robot proprioception, not object state.
    """
    import mujoco
    from cross_episode_sim.robot.arm_model import export_arm_model
    from cross_episode_sim.robot.profile import MOBILE_FRANKA
    local=mujoco.MjData(model)
    for j in range(model.njnt):
        name=model.joint(j).name or ''
        if not name.startswith('robot_0/') or name in ('robot_0/base_x','robot_0/base_y','robot_0/base_theta'):continue
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):raise ValueError('Unexpected robot joint')
        adr=model.jnt_qposadr[j];local.qpos[adr]=data.qpos[adr]
    mujoco.mj_kinematics(model,local)
    kin=export_arm_model(model,local,MOBILE_FRANKA,output,collision_resolution=.025)
    # The URDF root is a calibrated base frame, not a simulator world frame.
    import xml.etree.ElementTree as ET
    tree=ET.parse(kin['urdf_path'])
    for joint in tree.getroot().findall('joint'):
        if joint.find('child').get('link')=='base':
            joint.find('origin').set('xyz','0 0 0');joint.find('origin').set('rpy','0 0 0')
    tree.write(kin['urdf_path'])
    return kin


class ObservedArmPlanner:
    """No simulator handle: inputs are calibration, map, SLAM and encoders."""
    def __init__(self,kinematics,grid,T_map_base,extra_obstacle_capacity=0):
        from curobo._src.geom.types import Cuboid,SceneCfg
        from cross_episode_sim.robot.arm_planner import RobotArmPlanner
        ij=np.argwhere(grid.evidence>=2)
        points=np.column_stack((grid.origin[0]+(ij[:,1]+.5)*grid.config.resolution,grid.origin[1]+(ij[:,0]+.5)*grid.config.resolution,grid.config.min_obstacle_height+(ij[:,2]+.5)*grid.config.height_resolution))
        R=T_map_base[:3,:3];points=(points-T_map_base[:3,3])@R
        points=points[np.linalg.norm(points[:,:2],axis=1)<1.4]
        quat=Rotation.from_matrix(R.T).as_quat(scalar_first=True).tolist()
        dims=[grid.config.resolution+.01,grid.config.resolution+.01,grid.config.height_resolution+.01]
        self.boxes=[Cuboid(name=f'observed_{i}',pose=p.tolist()+quat,dims=dims) for i,p in enumerate(points)]
        self.arm=RobotArmPlanner(None,{},kinematics=kinematics,collision_cache={'cuboid':max(16,len(self.boxes)+1+extra_obstacle_capacity),'mesh':2})
        self.arm.planner.update_world(SceneCfg(cuboid=self.boxes))
        self.dt=self.arm.dt
    def plan(self,measured_joints,goal_position,goal_rotation):
        quat=Rotation.from_matrix(goal_rotation).as_quat(scalar_first=True)
        return self.arm.plan(measured_joints,np.r_[goal_position,quat].tolist())
