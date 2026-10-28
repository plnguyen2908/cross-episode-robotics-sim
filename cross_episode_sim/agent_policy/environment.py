"""Saved-scene physics with native MolmoSpaces control, no oracle execution."""
import json
from pathlib import Path
import numpy as np

class Environment:
    def __init__(self,source,output,controller_config):
        import mujoco
        import imageio.v2 as imageio
        from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
        from cross_episode_sim.finetuning.replay_scoring import CompositeReplayScorer
        from cross_episode_sim.control.replay_fixtures import ReplayFixtures
        self.mj=mujoco; self.output=Path(output); self.source=Path(source)
        self.model=mujoco.MjModel.from_binary_path(str(self.source/'action_recording/model.mjb'))
        self.data=mujoco.MjData(self.model)
        with np.load(self.source/'action_recording/initial_state.npz') as saved:
            mujoco.mj_setState(self.model,self.data,saved['state'],int(saved['state_type']))
        mujoco.mj_forward(self.model,self.data)
        self.config=controller_config
        self.control=MolmoSpacesController(self.model,self.data,configuration=controller_config,native_action_noise=False)
        self.scorer=CompositeReplayScorer(self.model,self.data,self.source)
        self.fixtures=ReplayFixtures(self.model,self.data)
        self.initial=json.loads((self.source/'initial_snapshot.json').read_text())
        self.prompt=(self.source/'instruction.txt').read_text().strip()
        self.task=json.loads((self.source/'config.json').read_text())['task']
        self.renderer=mujoco.Renderer(self.model,height=256,width=256)
        self.writer=imageio.get_writer(str(self.output/'rollout.mp4'),fps=25,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart'])
        self.physics=round(.02/self.model.opt.timestep)
        if not np.isclose(self.physics*self.model.opt.timestep,.02):raise ValueError('Invalid physics rate')
        self.start=float(self.data.time);self.steps=0;self.frames=0;self.pose=None;self.stable=0
        self.command={g:self.data.ctrl[ids].copy() for g,ids in self.control.groups.items()}
        self.trace=(self.output/'commands.jsonl').open('w')
        self.snapshot_video()

    def state(self):
        q=self.control.robot_view.get_qpos_dict()
        tcp=self.control.robot_view.get_move_group('gripper').leaf_frame_to_robot
        grip=float(self.command['gripper'][0]/255.)
        return {'base_xy_yaw':q['base'].tolist(),'arm_joints':q['arm'].tolist(),
                'tcp_base_matrix':tcp.tolist(),'gripper_command_fraction_closed':grip,
                'gripper_measured_qpos':q.get('gripper',np.array([])).tolist(),
                'simulation_seconds':float(self.data.time)-self.start,
                'arm_joint_limits':self.control.robot_view.get_move_group('arm').joint_pos_limits.tolist()}

    def images(self,depth=False):
        from cross_episode_sim.robot.observations import onboard_scene_option,depth_rgb
        rows=[];rgbs=[]
        for camera in ('robot_0/navigation_cam','robot_0/wrist_cam'):
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data,camera=camera,scene_option=onboard_scene_option())
            rgb=self.renderer.render().copy();rgbs.append(rgb)
            if depth:
                self.renderer.enable_depth_rendering()
                metric=self.renderer.render().copy()
                rows.append(np.concatenate([rgb,depth_rgb(metric)],axis=1))
        self.renderer.disable_depth_rendering()
        return np.concatenate(rows,axis=0) if depth else rgbs

    def snapshot_video(self):
        self.writer.append_data(self.images(depth=True));self.frames+=1

    def observe(self,turn):
        from PIL import Image
        paths=[]
        for name,rgb in zip(('navigation','wrist'),self.images()):
            p=self.output/f'{turn:04d}_{name}.png';Image.fromarray(rgb).save(p);paths.append(p)
        return self.state(),paths

    def tick(self,command):
        self.control.step(command);self.command={k:v.copy() for k,v in command.items()}
        self.trace.write(json.dumps({'time':float(self.data.time),'command':{k:v.tolist() for k,v in command.items()}})+'\n')
        before=float(self.data.time)
        for _ in range(self.physics):
            self.fixtures.step();self.scorer.observe(self.data);self.mj.mj_step(self.model,self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isclose(self.data.time-before,.02):raise RuntimeError('Physics diverged')
        self.steps+=1
        if self.steps%2==0:self.snapshot_video()
        if self.steps%25==0:
            self.scorer.observe(self.data)
            self.stable=self.stable+1 if all(self.goals().values()) else 0

    def goals(self):return self.scorer.score(self.initial,self.data)

    def act(self,name,args):
        if name=='observe':return {'ok':True,'executed_seconds':0}
        seconds=args['seconds']; n=max(1,round(seconds*50))
        original={k:v.copy() for k,v in self.command.items()}
        target={k:v.copy() for k,v in original.items()}
        q=self.control.robot_view.get_qpos_dict()
        if name=='move_base':
            yaw=q['base'][2]; f,l=args['forward'],args['left']
            target['base']=q['base']+np.array([np.cos(yaw)*f-np.sin(yaw)*l,np.sin(yaw)*f+np.cos(yaw)*l,args['yaw']])
            if np.linalg.norm([f,l])*1.875/seconds>.5 or abs(args['yaw'])*1.875/seconds>.8:raise ValueError('Increase seconds: base limits 0.5m/s and 0.8rad/s')
        elif name in ('move_joints','move_ee'):
            target['gripper']=np.array([args['gripper']*255])
            if name=='move_joints':target['arm']=np.array(args['arm'],dtype=float)
            else:
                from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController
                from scipy.spatial.transform import Rotation
                quat=np.array(args['quaternion_wxyz'])
                if abs(np.linalg.norm(quat)-1)>.01:raise ValueError('Quaternion must be unit length')
                if self.pose is None:self.pose=MolmoSpacesPoseController(self.model,self.data,configuration=self.config,native_action_noise=False)
                pose=np.eye(4);pose[:3,3]=args['position'];pose[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                saved=self.data.ctrl.copy()
                try:
                    self.pose.step({'base':original['base'],'eef_pose':pose,'gripper':target['gripper']})
                    target['arm']=self.data.ctrl[self.control.groups['arm']].copy()
                finally:self.data.ctrl[:]=saved
            limits=self.control.robot_view.get_move_group('arm').joint_pos_limits
            if np.any(target['arm']<limits[:,0]) or np.any(target['arm']>limits[:,1]):raise ValueError('Joint limit violation')
            if np.max(abs(target['arm']-original['arm']))*1.875/seconds>.8:raise ValueError('Increase seconds: arm limit 0.8rad/s')
        elif name!='wait':raise ValueError('Unknown physical command')
        for group,values in target.items():
            bounds=self.control.controllers[group].ctrl_range
            if np.any(values<bounds[:,0]) or np.any(values>bounds[:,1]):
                raise ValueError('Actuator target outside limits: '+group)
        for i in range(n):
            u=(i+1)/n;blend=10*u**3-15*u**4+6*u**5
            command={k:original[k]+blend*(target[k]-original[k]) for k in original}
            self.tick(command)
            if self.stable>=2:break
        return {'ok':True,'executed_seconds':(i+1)*.02,'note':'Motion executed; task success is assessed separately.'}

    def close(self):
        self.trace.close();self.writer.close();self.renderer.close()
