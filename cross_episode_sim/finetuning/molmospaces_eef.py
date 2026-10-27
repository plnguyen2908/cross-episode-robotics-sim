"""Base-frame TCP pose input solved with upstream MolmoSpaces IK."""
import numpy as np
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController


class MolmoSpacesPoseController:
    def __init__(self,model,data,**kwargs):
        from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig
        from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
        kwargs.pop('action_space',None)
        self.joints=MolmoSpacesController(model,data,action_space='absolute_joint_position',**kwargs)
        self.groups=self.joints.groups
        self.changes=self.joints.changes
        cfg=MobileFrankaRobotConfig(base_size=[.481,.481,.335])
        self.kinematics=MlSpacesKinematics(cfg)
        # Upstream standalone IK spawns at z=0, while saved scenes spawn
        # the base 5 mm above the floor. Its planar base frame has z=0.
        # Match the saved mounting height in the private FK/IK model only.
        self.kinematics._mj_model.body('robot_0/base').pos[2] = model.body('robot_0/base').pos[2]
        self.seed={g:q.copy() for g,q in self.joints.robot_view.get_qpos_dict().items()}
        self.seed['base']=np.zeros(3)
        expected=self.joints.robot_view.get_move_group('gripper').leaf_frame_to_robot
        actual=self.kinematics.fk(self.seed,np.eye(4),rel_to_base=True)['gripper']
        if not np.allclose(actual,expected,atol=1e-8,rtol=0):
            raise ValueError('Native IK geometry does not match the saved robot TCP frame')
        self.last_pose=None
        self.ik_calls=0
        self.max_pose_error=0.
        self.ik_failure=None
        self.ik_retries=0
        self.max_joint_step=0.

    def step(self,action):
        if set(action)!= {'base','eef_pose','gripper'}:
            raise ValueError('Expected base, eef_pose and gripper')
        pose=np.asarray(action['eef_pose'],dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():
            raise ValueError('Expected finite 4x4 base-frame TCP pose')
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(pose[:3,:3])<.999:
            raise ValueError('Invalid rigid pose')
        if self.last_pose is None or not np.array_equal(pose,self.last_pose):
            self.ik_calls+=1
            self.seed={g:q.copy() for g,q in self.joints.robot_view.get_qpos_dict().items()}
            self.seed['base']=np.zeros(3)
            solution=self.kinematics.ik('gripper',pose,['arm'],self.seed,np.eye(4),
                                        rel_to_base=True)
            if solution is None:
                # Resolve the redundant elbow posture with local null-space
                # seeds, using only the previous solution and native Jacobian.
                self.kinematics.fk(self.seed,np.eye(4),rel_to_base=True)
                jac=self.kinematics._robot_view.get_jacobian('gripper',['arm'])
                direction=np.linalg.svd(jac,full_matrices=True)[2][-1]
                limits=self.kinematics._robot_view.get_move_group('arm').joint_pos_limits
                for distance in (.01,-.01,.02,-.02,.04,-.04,.08,-.08,.16,-.16):
                    candidate={g:q.copy() for g,q in self.seed.items()}
                    candidate['arm']=np.clip(candidate['arm']+distance*direction,limits[:,0],limits[:,1])
                    self.ik_retries+=1
                    trial=self.kinematics.ik('gripper',pose,['arm'],candidate,np.eye(4),
                                              rel_to_base=True,max_iter=200)
                    if trial is not None and np.max(np.abs(trial['arm']-self.seed['arm']))<=.05:
                        solution=trial
                        break
            if solution is None:
                self.ik_failure=dict(call=self.ik_calls,pose=pose.tolist(),seed={k:v.tolist() for k,v in self.seed.items()},last_iterate={k:v.tolist() for k,v in self.kinematics._robot_view.get_qpos_dict().items()})
                raise RuntimeError(f'Native MolmoSpaces IK failed at solve {self.ik_calls}')
            actual=self.kinematics.fk(solution,np.eye(4),rel_to_base=True)['gripper']
            self.max_pose_error=max(self.max_pose_error,float(np.max(np.abs(actual-pose))))
            self.max_joint_step=max(self.max_joint_step,float(np.max(np.abs(solution['arm']-self.seed['arm']))))
            self.seed={g:q.copy() for g,q in solution.items()}
            self.last_pose=pose.copy()
        self.joints.step({'base':action['base'],'arm':self.seed['arm'],'gripper':action['gripper']})


def encode_eef_targets(controller,commands):
    """Offline FK only; joint labels are never passed to the IK executor."""
    seed={g:q.copy() for g,q in controller.seed.items()}
    poses=np.empty((len(commands),4,4),dtype=np.float64)
    previous=None
    for i,command in enumerate(commands):
        arm=command[controller.groups['arm']]
        if previous is None or not np.array_equal(arm,previous):
            seed['arm']=arm.copy()
            pose=controller.kinematics.fk(seed,np.eye(4),rel_to_base=True)['gripper'].copy()
            previous=arm.copy()
        poses[i]=pose
    return poses
