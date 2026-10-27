"""Replay successful legacy commands through upstream MobileFranka controllers.
This is a controller compatibility test in the saved scene, not full migration.
"""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from tqdm import tqdm
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController, DEFAULT_CONFIG, configure_native_actuators
from cross_episode_sim.finetuning.replay_scoring import CompositeReplayScorer
from cross_episode_sim.control.replay_fixtures import ReplayFixtures
from cross_episode_sim.data.replay_angles import continuous_yaw_commands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controller-config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--native-action-noise',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--reference-trace',type=Path,help='Successful native absolute replay control_trace.npz for offline deltas')
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--action-space',choices=['absolute_joint_position','joint_delta','joint_target_delta','joint_target_velocity','eef_pose'],default='absolute_joint_position')
    p.add_argument('--seconds',type=float,help='Truncated wiring check only; not task qualification')
    p.add_argument('--render-only',action='store_true')
    args=p.parse_args()
    np.random.seed(args.seed)
    if args.render_only:
        m=mujoco.MjModel.from_binary_path(str(args.output/'model.mjb'))
        d=mujoco.MjData(m)
        render_video(m,d,np.load(args.output/'executed.npz')['qpos'],args.output)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    recording=args.source/'action_recording'
    meta=json.loads((recording/'metadata.json').read_text())
    if not meta['source_success']:raise ValueError('Source must be successful')
    m=mujoco.MjModel.from_binary_path(str(recording/'model.mjb'))
    actuator_changes=configure_native_actuators(m,args.controller_config)
    d=mujoco.MjData(m)
    with np.load(recording/'initial_state.npz') as initial:
        mujoco.mj_setState(m,d,initial['state'],int(initial['state_type']))
    mujoco.mj_forward(m,d)
    controller_class=MolmoSpacesController
    if args.action_space=='eef_pose':
        from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController
        controller_class=MolmoSpacesPoseController
    controller=controller_class(m,d,configuration=args.controller_config,action_space=args.action_space,native_action_noise=args.native_action_noise,dt=float(m.opt.timestep))
    scorer=CompositeReplayScorer(m,d,args.source);fixtures=ReplayFixtures(m,d)
    with np.load(recording/'actions.npz') as raw:
        times=raw['time'];commands=raw['ctrl'];expected=raw['qpos'];stages=raw['stage']
    commands,expected,wraps=continuous_yaw_commands(expected,commands,
        yaw_qpos=int(m.joint('robot_0/base_theta').qposadr[0]),
        yaw_actuator=m.actuator('robot_0/base_theta_act').id)
    if not np.allclose(np.diff(times),m.opt.timestep,atol=1e-8):raise ValueError('Timing mismatch')
    # Fixed offline conversion: reference is the recorded pre-step measured
    # state, never the live replay state or an online absolute-target correction.
    if args.reference_trace:
        reference=np.load(args.reference_trace)
        if not np.array_equal(reference['time'],times):raise ValueError('Reference clock differs')
        if not bool(reference['success']):raise ValueError('Reference replay did not pass')
        expected=reference['qpos']
    if args.action_space == 'joint_delta':
        converted=commands.copy()
        for group in ('base', 'arm'):
            for actuator,joint in zip(controller.groups[group],controller.robot_view.get_move_group(group)._joint_ids):
                address=int(m.jnt_qposadr[joint])
                converted[:,actuator]-=expected[:,address]
        np.savez_compressed(args.output/'converted_actions.npz',
                            commands=converted,time=times)
        commands=converted
    if args.action_space in ('joint_target_delta','joint_target_velocity'):
        converted=commands.astype('S256' if args.action_space=='joint_target_velocity' else np.longdouble)
        for group in ('base','arm'):
            ids=controller.groups[group]
            from cross_episode_sim.data.target_deltas import encode_target_deltas, encode_target_velocities
            converted[:,ids]=(encode_target_velocities(controller.reference[group],commands[:,ids],float(m.opt.timestep))
                if args.action_space=='joint_target_velocity' else encode_target_deltas(controller.reference[group],commands[:,ids]))
        np.savez_compressed(args.output/'converted_actions.npz',commands=converted,time=times)
        commands=converted
    eef_poses=None
    if args.action_space=='eef_pose':
        from cross_episode_sim.finetuning.molmospaces_eef import encode_eef_targets
        eef_poses=encode_eef_targets(controller,commands)
        np.savez_compressed(args.output/'eef_actions.npz',eef_pose=eef_poses,
            base=commands[:,controller.groups['base']],gripper=commands[:,controller.groups['gripper']],time=times)
    full_steps=len(times)
    if args.seconds is not None:
        if args.seconds<=0:raise ValueError('seconds must be positive')
        times=times[:max(1,round(args.seconds/m.opt.timestep))]
    groups=controller.groups
    report=dict(source=str(args.source),backend="molmospaces",controller_config=str(args.controller_config),
        controller=('molmo_spaces.controllers.joint_rel_pos.JointRelPosController' if args.action_space=='joint_delta' else 'molmo_spaces.controllers.joint_pos.JointPosController'),
        action_space=args.action_space,storage_dtype=str(commands.dtype),integration_dt=float(m.opt.timestep),native_action_noise=args.native_action_noise,seed=args.seed,delta_reference='previous_command_target' if args.action_space in ('joint_target_delta','joint_target_velocity') else str(args.reference_trace) if args.reference_trace else 'legacy_pre_step_measured_state' if args.action_space=='joint_delta' else None,
        actuator_changes=actuator_changes,retiming=False,state_resets_during_execution=0,
        full_native_environment=False)
    if eef_poses is not None:
        report.update(ik_seed_reference=controller.seed_reference,
                      ik_uses_recorded_joint_targets=False)
    (args.output/'configuration.json').write_text(json.dumps(report,indent=2))
    control_states=np.empty((len(times),m.nq));
    execution_error=None;completed_steps=0
    poses=[];max_error=0.;start=float(d.time);last_stage=None
    stride=round(.04/m.opt.timestep)
    for i in tqdm(range(len(times)),desc='Native MolmoSpaces controller replay'):
        stage=meta['stages'][int(stages[i])]
        if stage!=last_stage:
            tqdm.write(f'{d.time-start:.2f}s — {stage}')
            last_stage=stage
        control_states[i]=d.qpos
        if eef_poses is not None:
            try:
                controller.step({'eef_pose':eef_poses[i],'base':commands[i,controller.groups['base']],
                                 'gripper':commands[i,controller.groups['gripper']]})
            except RuntimeError as error:
                (args.output/'ik_failure.json').write_text(json.dumps(dict(step=i,time=float(d.time),error=str(error),detail=controller.ik_failure),indent=2))
                execution_error=str(error)
                break
        else:
            controller.apply_recorded_command(commands[i])
        fixtures.step();scorer.observe(d)
        mujoco.mj_step(m,d)
        completed_steps=i+1
        if not np.isfinite(d.qpos).all():raise RuntimeError('Physics diverged')
        if i%stride==0:poses.append(d.qpos.copy())
        max_error=max(max_error,float(np.max(np.abs(d.qpos-expected[i]))))
    scorer.observe(d)
    goal=scorer.score(json.loads((args.source/'initial_snapshot.json').read_text()),d)
    result=dict(backend="molmospaces",action_space=args.action_space,native_action_noise=args.native_action_noise,seed=args.seed,success=all(goal.values()) and completed_steps==full_steps and execution_error is None,truncated=completed_steps!=full_steps,execution_error=execution_error,goal_evidence=goal,
                simulated_seconds=float(d.time)-start,max_qpos_error=max_error,
                full_native_environment=False)
    if eef_poses is not None:
        result.update(ik_backend='molmo_spaces.kinematics.mujoco_kinematics.MlSpacesKinematics',
                      ik_seed_reference=controller.seed_reference,
                      ik_calls=controller.ik_calls,ik_retries=controller.ik_retries,
                      max_joint_target_step=controller.max_joint_step,
                      max_target_from_measured=controller.max_measured_target_step,
                      max_pose_matrix_error=controller.max_pose_error)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output/'executed.npz',qpos=poses,final_qpos=d.qpos)
    np.savez_compressed(args.output/'control_trace.npz',qpos=control_states[:completed_steps],time=times[:completed_steps],success=result['success'])
    mujoco.mj_saveModel(m,str(args.output/'model.mjb'),None)
    print(json.dumps(result),flush=True)
    if not args.no_render:render_video(m,d,poses,args.output)


def render_video(m,d,poses,output):
    import imageio.v2 as imageio
    from cross_episode_sim.robot.observations import depth_rgb
    option=mujoco.MjvOption();option.geomgroup[0]=0;option.sitegroup[:]=0
    with mujoco.Renderer(m,height=256,width=256) as renderer, imageio.get_writer(
        str(output/'rollout.mp4'),fps=25,codec='libx264',pixelformat='yuv420p',
        output_params=['-movflags','+faststart']) as writer:
        for q in tqdm(poses,desc='Render executed states'):
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            panels=[]
            for cam in ('robot_0/navigation_cam','robot_0/wrist_cam'):
                renderer.disable_depth_rendering();renderer.update_scene(d,cam,scene_option=option)
                rgb=renderer.render().copy()
                renderer.enable_depth_rendering();renderer.update_scene(d,cam,scene_option=option)
                panels.append(np.concatenate([rgb,depth_rgb(renderer.render())],axis=1))
            writer.append_data(np.concatenate(panels,axis=0))


if __name__=='__main__':main()
