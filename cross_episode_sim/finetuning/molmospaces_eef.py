"""Base-frame TCP pose input solved with upstream MolmoSpaces IK."""
import numpy as np
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController


class MolmoSpacesPoseController:
    backend = 'molmospaces'
    supported_action_spaces = ('eef_pose',)

    def __init__(self,model,data,**kwargs):
        from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig
        from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
        if kwargs.pop('action_space','eef_pose') != 'eef_pose':
            raise ValueError('MolmoSpacesPoseController requires eef_pose actions')
        self.joints=MolmoSpacesController(model,data,action_space='absolute_joint_position',**kwargs)
        self.groups=self.joints.groups
        self.changes=self.joints.changes
        cfg=MobileFrankaRobotConfig(base_size=[.481,.481,.335])
        self.nominal_arm=np.asarray(cfg.init_qpos['arm'],dtype=float)
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
        self.max_measured_target_step=0.
        self.seed_reference='measured'
        self._retry_rng=np.random.default_rng(0)  # Does not perturb native action-noise RNG.

    def _interior_seed(self, seed):
        """Bias only IK's redundant posture away from a nearby joint limit."""
        import mujoco
        self.kinematics.fk(seed,np.eye(4),rel_to_base=True)
        mujoco.mj_fwdPosition(self.kinematics._mj_model,self.kinematics._mj_data)
        view=self.kinematics._robot_view
        limits=view.get_move_group('arm').joint_pos_limits
        q=seed['arm']
        low=q-limits[:,0];high=limits[:,1]-q
        gradient=np.maximum(.3-low,0)-np.maximum(.3-high,0)+.1*(self.nominal_arm-q)
        jac=view.get_jacobian('gripper',['arm'])
        null=np.linalg.svd(jac,full_matrices=True)[2][-1]
        shift=null*np.dot(null,gradient)
        norm=np.max(np.abs(shift))
        if norm>0.02:shift*=.02/norm
        candidate={g:v.copy() for g,v in seed.items()}
        candidate['arm']=np.clip(q+shift,limits[:,0],limits[:,1])
        return candidate

    def step(self,action):
        if set(action)!= {'base','eef_pose','gripper'}:
            raise ValueError('Expected base, eef_pose and gripper')
        for group,size in (('base',3),('gripper',1)):
            value=np.asarray(action[group],dtype=float)
            if value.shape!=(size,) or not np.isfinite(value).all():
                raise ValueError('Invalid pose action for '+group)
        pose=np.asarray(action['eef_pose'],dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():
            raise ValueError('Expected finite 4x4 base-frame TCP pose')
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(pose[:3,:3])<.999:
            raise ValueError('Invalid rigid pose')
        if self.last_pose is None or not np.array_equal(pose,self.last_pose):
            self.ik_calls+=1
            previous_target=self.seed['arm'].copy()
            # Match upstream manipulation policies: tracking/contact can make
            # the previous command differ materially from the actual posture.
            self.seed={g:q.copy() for g,q in self.joints.robot_view.get_qpos_dict().items()}
            self.seed['base']=np.zeros(3)
            candidate=self._interior_seed(self.seed)
            solution=self.kinematics.ik('gripper',pose,['arm'],candidate,np.eye(4),
                                        rel_to_base=True,eps=1e-7)
            if solution is None:
                # Resolve the redundant elbow posture with local null-space
                # seeds, using only the measured state and native Jacobian.
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
                # The almost-straight initial tuck has two local elbow branches.
                # Retry a bent-elbow seed rather than staying pinned at the
                # wrist's lower limit. This uses no recorded joint targets.
                for bend in (.2,.4,.6):
                    candidate={g:q.copy() for g,q in self.seed.items()}
                    candidate['arm'][1]-=bend/2
                    candidate['arm'][3]-=bend
                    candidate['arm'][5]+=bend/2
                    candidate['arm']=np.clip(candidate['arm'],limits[:,0],limits[:,1])
                    self.ik_retries+=1
                    solution=self.kinematics.ik('gripper',pose,['arm'],candidate,np.eye(4),
                        rel_to_base=True,eps=1e-7,damping=1e-4,dt=.5,max_iter=500)
                    if solution is not None:
                        break
            if solution is None:
                # Other joints can also pin a local branch. Try bounded seeds
                # and select the nearest solved posture, not the first branch.
                best_distance=np.inf
                for attempt in range(24):
                    candidate={g:q.copy() for g,q in self.seed.items()}
                    scale=.2 if attempt<8 else .5 if attempt<16 else 1.
                    candidate['arm']=np.clip(candidate['arm']+self._retry_rng.normal(0,scale,7),
                                              limits[:,0],limits[:,1])
                    self.ik_retries+=1
                    trial=self.kinematics.ik('gripper',pose,['arm'],candidate,np.eye(4),
                        rel_to_base=True,eps=1e-7,max_iter=300)
                    if trial is not None:
                        distance=np.linalg.norm(trial['arm']-self.seed['arm'])
                        if distance<best_distance:
                            solution=trial
                            best_distance=distance
                        if np.max(np.abs(trial['arm']-self.seed['arm']))<.1:
                            break
            if solution is None:
                candidate={g:q.copy() for g,q in self.seed.items()}
                candidate['arm']=self.nominal_arm.copy()
                self.ik_retries+=1
                solution=self.kinematics.ik('gripper',pose,['arm'],candidate,np.eye(4),
                    rel_to_base=True,eps=1e-7)
            if solution is None:
                self.ik_failure=dict(call=self.ik_calls,pose=pose.tolist(),seed={k:v.tolist() for k,v in self.seed.items()},last_iterate={k:v.tolist() for k,v in self.kinematics._robot_view.get_qpos_dict().items()})
                raise RuntimeError(f'Native MolmoSpaces IK failed at solve {self.ik_calls}')
            actual=self.kinematics.fk(solution,np.eye(4),rel_to_base=True)['gripper']
            self.max_pose_error=max(self.max_pose_error,float(np.max(np.abs(actual-pose))))
            self.max_joint_step=max(self.max_joint_step,float(np.max(np.abs(solution['arm']-previous_target))))
            self.max_measured_target_step=max(self.max_measured_target_step,float(np.max(np.abs(solution['arm']-self.seed['arm']))))
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
