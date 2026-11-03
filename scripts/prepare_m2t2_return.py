"""Physically return an empty arm to navigation posture using observed obstacles."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.navigation.local_scan import ScanConfig
from cross_episode_sim.control.observed_arm import ObservedArmPlanner,robot_calibration
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner,smooth_joint_path
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--place',type=Path,required=True)
    p.add_argument('--observed-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_binary_path(str(a.place/'private_checkpoint/model.mjb'));d=mujoco.MjData(m)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(m,d,np.load(a.place/'private_checkpoint/state.npy'),spec);mujoco.mj_forward(m,d)
    c=MolmoSpacesController(m,d);hold={k:c.robot_view.get_move_group(k).noop_ctrl.copy() for k in ('base','arm','gripper')}
    if np.max(hold['gripper'])>5:raise RuntimeError('Return requires released gripper command')
    grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12))
    restore_observed_map(grid,a.observed_map);slam=RGBDMapper();geo=RobotGeometry(m)
    start=d.time;success=False;error=None;tick=0
    with OnboardSensorInterface(m,d) as sensors,imageio.get_writer(a.output/'rollout.mp4',fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video:
        def observe():
            packet=sensors.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
            grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
            video.append_data(np.concatenate([np.concatenate([v.rgb,depth_rgb(v.depth_m)],axis=1) for v in packet.cameras.values()]))
            return packet,tracking
        def step(q):
            nonlocal tick
            c.step(dict(base=hold['base'],arm=q,gripper=hold['gripper']))
            for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
            tick+=1
            if tick%5==0:observe()
        try:
            packet,tracking=observe()
            if not tracking.accepted:raise RuntimeError('No accepted sensor pose for return')
            calibration=robot_calibration(m,d,a.output/'robot')
            observed=ObservedArmPlanner(calibration,grid,tracking.T_map_base)
            fixed=[[*s['center'],s['radius']] for s in calibration['collision_spheres'].get('base',[])]
            planner=ObservedGraspPlanner(observed.arm,geo,observed.boxes,fixed_floor_support_spheres=fixed)
            q=packet.arm.position;goal=np.asarray(ScanConfig().stowed_arm)
            path=smooth_joint_path(q,goal)
            dense=np.linspace(q,goal,max(2,int(np.ceil(np.sum(abs(q-goal))/.001))+1))
            if not planner.clear(path) or not planner.clear(dense):raise RuntimeError('Return posture sweep is not collision-clear')
            for joints in path:step(joints)
            stable=0
            for _ in range(100):
                step(goal)
                error_q=np.max(abs(c.robot_view.get_move_group('arm').joint_pos-goal))
                velocity=max(abs(float(d.joint(n).qvel[0])) for n in geo.joint_names)
                stable=stable+1 if error_q<.015 and velocity<.04 else 0
                if stable>=5:break
            if stable<5:raise RuntimeError('Return posture did not settle')
            success=True
        except Exception as exc:error=str(exc)
    checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None)
    state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state)
    grid.save(a.output);slam.save(a.output/'slam')
    from cross_episode_sim.mapping.continuity import load_memory,save_memory
    save_memory(a.output,load_memory(a.observed_map) if a.observed_map else dict(rooms=[],tables=[]))
    result=dict(success=success,error=error,simulated_seconds=float(d.time-start),reason='Checked return to currently required navigation posture; posture retention not yet supported by pickup navigation')
    (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
