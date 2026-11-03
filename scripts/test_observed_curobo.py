"""Continue a private navigation checkpoint; plan solely from its observed map."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.control.observed_arm import ObservedArmPlanner,robot_calibration
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import ContactAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(a.source/'private_checkpoint/model.mjb'));d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,np.load(a.source/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    r=json.loads((a.source/'report.json').read_text())
    if not r.get('docking_success'):raise RuntimeError('Requires successful sensor docking')
    g=ObservedNavigationMap(MapConfig(**json.loads((a.source/'summary.json').read_text())['config']));saved=np.load(a.source/'navigation_map.npz');g.origin=saved['origin_xy'];g.evidence=saved['evidence'];g.floor_views=saved['floor_views']
    T=np.load(a.source/'slam/estimated_base_poses.npy')[-1];target=T[:3,:3].T@(np.array(r['selected_object']['position'])-T[:3,3])
    with OnboardSensorInterface(m,d) as sensors:
        packet=sensors.capture();q=packet.arm.position.copy();geometry=RobotGeometry(m)
        planner=ObservedArmPlanner(robot_calibration(m,d,a.output/'robot'),g,T)
        attempts=[];trajectory=None;goal=None;wall=time.monotonic()
        from scipy.spatial.transform import Rotation
        for height in (.18,.15,.12):
            candidate=target+np.array([0.,0.,height])
            for tilt in (0.,30.,-30.,60.,-60.):
                orientation=Rotation.from_euler('y',tilt,degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
                try:
                    # Let cuRobo's multi-seed IK decide reachability, rather
                    # than rejecting from a separate local IK optimizer.
                    trajectory=planner.plan(q,candidate,orientation);goal=candidate
                    attempts.append(dict(height=height,tilt_degrees=tilt,status='planned',samples=len(trajectory)));break
                except RuntimeError as e:attempts.append(dict(height=height,tilt_degrees=tilt,status='planning_failed',error=str(e)))
            if trajectory is not None:break
        report=dict(source=str(a.source),attempts=attempts,obstacle_voxels=len(planner.boxes),planning_wall_seconds=time.monotonic()-wall,target_base=target.tolist(),success=False,scope='Non-contact arm pregrasp only; no grasp or lift. Unobserved geometry is not certified free.')
        if trajectory is not None:
            controller=MolmoSpacesController(m,d);hold={name:controller.robot_view.get_move_group(name).noop_ctrl.copy() for name in ('base','arm','gripper')};audit=ContactAudit(m)
            # Interpolate the planned timed path onto the existing 50 Hz controller.
            times=np.arange(len(trajectory))*planner.dt;sample_times=np.arange(0,times[-1]+.019,.02)
            commands=np.column_stack([np.interp(sample_times,times,trajectory[:,j]) for j in range(7)])
            np.save(a.output/'commands.npy',commands);start=d.time;max_error=0.;stable=0;executed=0
            with imageio.get_writer(str(a.output/'approach.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
                for i in tqdm(range(len(commands)+100),desc='cuRobo observed pregrasp'):
                    qcmd=commands[min(i,len(commands)-1)];action={k:v.copy() for k,v in hold.items()};action['arm']=qcmd;controller.step(action)
                    for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d);audit.observe(d)
                    packet=sensors.capture() if i%5==0 else packet
                    measured=controller.robot_view.get_move_group('arm').joint_pos;error=float(np.max(np.abs(measured-qcmd)));max_error=max(max_error,error);executed+=1
                    if i%5==0:
                        panel=Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0));draw=ImageDraw.Draw(panel);draw.rectangle((0,0,panel.width,24),fill='white');draw.text((4,5),f'cuRobo observed pregrasp | {d.time-start:.2f}s | tracking {error:.3f} rad',fill='black');video.append_data(np.array(panel));panel.save(a.output/'latest_frame.png')
                    if error>.25:report['stop_reason']='joint_tracking_error';break
                    if i>=len(commands):
                        stable=stable+1 if error<.025 and np.max(np.abs([d.joint(n).qvel[0] for n in packet.arm.names]))<.04 else 0
                        if stable>=5:break
            final=geometry.fk(measured)[geometry.tcp_body]@geometry.tcp_local;position_error=float(np.linalg.norm(final[:3,3]-goal))
            report.update(simulated_seconds=float(d.time-start),max_joint_tracking_error=max_error,goal_base=goal.tolist(),final_tcp_base=final[:3,3].tolist(),position_error_m=position_error,external_contacts=list(audit.events.values()),success=bool(stable>=5 and position_error<.025 and not audit.events),executed_steps=executed)
        (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
