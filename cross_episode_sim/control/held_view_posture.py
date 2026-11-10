"""Checked arm clearance for a held object beside a higher receptacle.

Uses the observed payload and local RGB-D only. Never releases or snaps the
object; native arm control follows a collision-checked trajectory at 50 Hz.
"""
import json
import mujoco
import numpy as np
from curobo._src.geom.types import Cuboid,SceneCfg
from .observed_payload import ObservedPayload
from .observed_grasp_planner import ObservedGraspPlanner,merge_voxels
from .observed_arm import robot_calibration
from .planner_cache import sensor_arm_planner
from .sensor_appliance_backend import camera_points
from ..navigation.shaped_docking import RobotGeometry
from ..finetuning.molmospaces_execution import MolmoSpacesController
from ..recording.deferred import StateVideoRecorder


def raise_held_view(runtime,tool_points,output,target_height=1.25):
    output.mkdir(parents=True,exist_ok=False)
    m,d=runtime.model,runtime.data;geometry=RobotGeometry(m);payload=ObservedPayload(geometry,tool_points)
    raw=runtime.sensors.capture();packet=payload.filter_packet(raw);q=packet.arm.position.copy()
    tcp=geometry.tcp_pose(q);start=float(d.time)
    if tcp[2,3]>=target_height-.02:return dict(success=True,simulated_seconds=0.,reason='already_clear')
    points=np.concatenate([camera_points(camera)[0] for camera in packet.cameras.values()])
    points=points[(points[:,2]>.08)&(np.linalg.norm(points,axis=1)<1.8)]
    for box in geometry.boxes(q):points=points[~np.all(np.abs((points-box.center)@box.axes)<=box.half+.01,axis=1)]
    vox=np.unique(np.floor(points/.015).astype(int),axis=0)
    boxes=[Cuboid(name=f'view_{i}',pose=((lo+hi+1)*.0075).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.015+.003).tolist()) for i,(lo,hi) in enumerate(merge_voxels(vox))]
    arm=sensor_arm_planner(robot_calibration(m,d,output/'robot'),collision_cache={'cuboid':max(16,len(boxes)),'mesh':2})
    arm.planner.update_world(SceneCfg(cuboid=boxes))
    planner=ObservedGraspPlanner(arm,geometry,boxes,allow_curobo_fallback=True)
    target=tcp[:3,3].copy();target[2]=target_height
    strategy='vertical_clearance'
    try:
        path=planner.plan(q,target,tcp[:3,:3])
    except RuntimeError:
        from ..robot.profile import MOBILE_FRANKA
        path=planner.plan_joint_goal(q,np.asarray(MOBILE_FRANKA.travel_posture))
        strategy='compact_retained_grasp'
    for joints in path:
        T=geometry.tcp_pose(joints);xyz=tool_points[::max(1,len(tool_points)//200)]@T[:3,:3].T+T[:3,3]
        if not planner.world_clear(np.c_[xyz,np.full(len(xyz),.006)][None],margin=.002):
            raise RuntimeError('Observed held surface blocks clearance posture')
    controller=MolmoSpacesController(m,d)
    hold={g:controller.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')}
    hold['gripper']=controller.robot_view.get_move_group('gripper').ctrl.copy()
    with StateVideoRecorder(m,d,output/'state_recording') as recorder:
        def command(joints):
            hold['arm']=joints;controller.step(hold)
            for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
            recorder.record('held_object_clearance')
        for joints in path:command(joints)
        steady=0
        for _ in range(100):
            command(path[-1]);measured=controller.robot_view.get_move_group('arm').joint_pos
            velocity=np.array([d.joint(n).qvel[0] for n in geometry.joint_names])
            steady=steady+1 if np.max(abs(measured-path[-1]))<.015 and np.max(abs(velocity))<.04 else 0
            if steady>=5:break
        if steady<5:raise RuntimeError('Clearance posture did not settle')
    result=dict(success=True,simulated_seconds=float(d.time)-start,
                reason='clearance for loaded local docking',strategy=strategy,requested_height=target_height,
                actual_tcp=geometry.tcp_pose(controller.robot_view.get_move_group('arm').joint_pos)[:3,3].tolist())
    (output/'report.json').write_text(json.dumps(result,indent=2));return result
