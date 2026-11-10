"""Live observed handle acquisition -> shared sensor door opening, no resets."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from cross_episode_sim.control.m2t2_policy import M2T2Policy,grasp_to_tcp
from cross_episode_sim.perception.handle_surface import observed_handle_surface


class LiveDoorOpener:
    def __init__(self,runtime,pointer,output,*,handle_then_push=False):
        self.runtime=runtime;self.pointer=pointer;self.output=Path(output);self.calls=0
        self.handle_then_push=handle_then_push
        self.model=M2T2Policy('/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth','external/M2T2')

    def open(self,sensors,tracking,target_map):
        root=self.output/f'door_{self.calls:03d}';self.calls+=1;root.mkdir(parents=True)
        packet=sensors.capture();cloud=[];colors=[];targets=[]
        inv=np.linalg.inv(tracking.T_map_base);expected=inv[:3,:3]@target_map+inv[:3,3]
        for camera in packet.cameras.values():
            valid=camera.valid_depth.copy()
            if camera.robot_mask is not None:valid &= ~camera.robot_mask
            v,u=np.where(valid);z=camera.depth_m[v,u]
            points=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z]@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
            cloud.append(points);colors.append(camera.rgb[v,u])
        points=np.concatenate(cloud);rgb=np.concatenate(colors)
        from cross_episode_sim.perception.handle_acquisition import observed_handle_near
        acquisition=dict(method='tracked_handle_current_depth',failures=[])
        try:
            target,info=observed_handle_near(points,expected)
            targets.append(target)
        except ValueError as exc:acquisition['failures'].append(str(exc))
        # Existing semantic identity + current measured handle geometry avoids
        # asking a VLM to independently rediscover the same small target.
        for camera in (() if targets else packet.cameras.values()):
            valid=camera.valid_depth.copy()
            if camera.robot_mask is not None:valid &= ~camera.robot_mask
            result=self.pointer.point(Image.fromarray(camera.rgb),'room door handles')
            for x,y in result['points']:
                u,v=int(x),int(y)
                if not(0<=v<valid.shape[0] and 0<=u<valid.shape[1] and valid[v,u]):continue
                z=camera.depth_m[v,u]
                target=camera.T_base_optical[:3,:3]@np.array([(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z])+camera.T_base_optical[:3,3]
                world=tracking.T_map_base[:3,:3]@target+tracking.T_map_base[:3,3]
                if np.linalg.norm(world-target_map)<.2:
                    try:target,_=observed_handle_near(points,target)
                    except ValueError as exc:
                        acquisition['failures'].append(str(exc));continue
                    targets.append(target);acquisition['method']='fresh_pointing_current_depth'
        if not targets:
            print(json.dumps(dict(stage='handle_reacquisition',success=False,**acquisition)),flush=True)
            return dict(success=False,reason='fresh_handle_not_visible',recoverable=True,output=str(root))
        target=min(targets,key=lambda p:np.linalg.norm(p-expected))
        print(json.dumps(dict(stage='handle_reacquisition',success=True,
            target_residual=float(np.linalg.norm(target-expected)),**acquisition)),flush=True)
        crop=np.linalg.norm(points-target,axis=1)<.18
        try:observed_handle_surface(points[crop],target)
        except ValueError as e:return dict(success=False,reason=str(e))
        np.savez(root/'observed_world.npz',points=points,rgb=rgb)
        np.savez(root/'input_0.npz',points=points[crop],rgb=rgb[crop],target=target)
        candidates=[]
        for _ in range(3):
            out=self.model.predict(points[crop],rgb[crop])
            for grasps,confidence,contacts in zip(out['grasps'],out['grasp_confidence'],out['grasp_contacts']):
                for grasp,score,contact in zip(grasps,confidence,contacts):
                    if np.linalg.norm(contact-target)<.16:candidates.append(dict(confidence=float(score),contact=contact.tolist(),tcp=grasp_to_tcp(grasp).tolist()))
        candidates.sort(key=lambda r:-r['confidence']);(root/'grasps_0.json').write_text(json.dumps(dict(candidates=candidates)))
        grasp=root/'grasp'
        with (root/'grasp.log').open('w') as log:
            result=self.runtime.run([sys.executable,'scripts/test_m2t2_handle_execution.py','--source',str(root),'--output',str(grasp),'--target','0','--surface-filter','--align-panel','--center-handle','--max-candidates',str(len(candidates))],log)
        if result.returncode:return dict(success=False,reason='grasp_stage_error',output=str(root))
        # Do not consume private success/contact scoring as policy feedback.
        report=json.loads((grasp/'report.json').read_text())
        print(json.dumps(dict(stage='handle_grasp_execution',output=str(grasp),
            physical_grasp_tested=report.get('physical_grasp_tested',False),
            planned_pose_attempts=report.get('tested_candidates'),
            contact_backoff=report.get('chosen',{}).get('contact_backoff'),
            rejected_plans=len(report.get('planning_rejections',[])))),flush=True)
        if not report.get('physical_grasp_tested'):return dict(success=False,reason='no_grasp_plan',output=str(root))
        opened=root/'opening'
        grid=getattr(self.runtime,'observed_navigation_map',None)
        if grid is not None:
            from shapely.geometry import box
            from shapely.ops import unary_union
            from .contact_retreat import transform_support_region
            rr,cc=np.where(grid.floor_views>=grid.config.minimum_floor_views)
            r=grid.config.resolution;xy=grid.origin+np.c_[cc,rr]*r
            floor=unary_union([box(x,y,x+r,y+r) for x,y in xy])
            self.runtime.contact_support_region=transform_support_region(floor,np.linalg.inv(tracking.T_map_base))
        with (root/'open.log').open('w') as log:
            from .door_open_command import door_open_command
            command=door_open_command(grasp,opened,contact_motion_profile='fast',handle_then_push=self.handle_then_push)
            # The full pipeline and saved-grasp qualification use this exact
            # backend. Tracking warnings remain diagnostics inside that backend.
            backend='handle_then_panel_push' if self.handle_then_push else 'original_handle_pull'
            print(json.dumps(dict(stage='door_execution',backend=backend,
                contact_motion_profile='fast',tracking_deviation='diagnostic_only',
                command=command)),flush=True)
            result=self.runtime.run(command,log)
        if result.returncode:return dict(success=False,reason='opening_stage_error',output=str(root))
        report=json.loads((opened/'report.json').read_text())
        opened_handle=None;doorway_center=None;door_update=None
        if report.get('observed_open_handle_base') is not None:
            packet=sensors.capture(camera_keys=('navigation',))
            final_tracking=self.runtime.mapper.update(packet.cameras['navigation'],packet.timestamp)
            if final_tracking.accepted:
                point=np.asarray(report['observed_open_handle_base'])
                opened_handle=(final_tracking.T_map_base[:3,:3]@point+final_tracking.T_map_base[:3,3]).tolist()
                if grid is not None and report.get('error') is None and (report.get('observed_open_angle_degrees') or 0)>=80. and report.get('observed_hinge_base') is not None and report.get('observed_closed_handle_base') is not None:
                    T=final_tracking.T_map_base
                    hinge=T[:3,:3]@np.asarray(report['observed_hinge_base'])+T[:3,3]
                    closed=T[:3,:3]@np.asarray(report['observed_closed_handle_base'])+T[:3,3]
                    doorway_center=((hinge+closed)/2).tolist()
                    # Sensor-derived leaf reach for an outward-first crossing
                    # approach. Do not cut around its partly occluded free tip.
                    grid.opened_leaf_reach=float(np.linalg.norm(closed[:2]-hinge[:2]))+.15
                    door_update=grid.refresh_opened_door(hinge[:2],closed[:2],np.asarray(opened_handle)[:2],timestamp=packet.timestamp)

        # Error is observed execution status. Private score is deliberately ignored.
        if grid is not None and door_update is None and report.get('error') is None and (report.get('observed_open_angle_degrees') or 0)>=80.:
            # Invalidate stale panel evidence only when subsequent sensor rays
            # observe free space. Use post-motion time, not the pre-open map time.
            grid.refresh_opened_portal(np.asarray(target_map)[:2],timestamp=sensors.capture(camera_keys=('navigation',)).timestamp)
        return dict(doorway_center_map=doorway_center,door_map_update=door_update,success=report.get('error') is None and (report.get('observed_open_angle_degrees') or 0)>=80.,reason=report.get('error'),observed_angle=report.get('observed_open_angle_degrees'),output=str(root),opened_handle_map=opened_handle,portal_normal_map=(tracking.T_map_base[:3,:3]@np.asarray(json.loads((grasp/'report.json').read_text())['surface_info']['panel_normal'])).tolist(),
                    opening_backend=backend,panel_push_requested=self.handle_then_push,
                    panel_contact_selected=any(e.get('phase')=='push_contact_selected' for e in report.get('events',[])),
                    simulated_seconds=report.get('simulated_seconds'),wall_timings=report.get('wall_timings'))

    def stow(self,sensors,recorder,posture):
        from curobo._src.geom.types import Cuboid,SceneCfg
        from .observed_grasp_planner import ObservedGraspPlanner,smooth_joint_path
        from .observed_arm import robot_calibration
        from .planner_cache import sensor_arm_planner
        from .sensor_drawer_geometry import unique_voxels,merge_occupied
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry
        from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
        import mujoco
        m,d=self.runtime.model,self.runtime.data;geo=RobotGeometry(m);cloud=[]
        for cam in sensors.capture().cameras.values():
            valid=cam.valid_depth.copy()
            if cam.robot_mask is not None:valid &= ~cam.robot_mask
            v,u=np.where(valid);z=cam.depth_m[v,u]
            pts=np.c_[(u-cam.K[0,2])*z/cam.K[0,0],(v-cam.K[1,2])*z/cam.K[1,1],z]@cam.T_base_optical[:3,:3].T+cam.T_base_optical[:3,3]
            cloud.append(pts[(pts[:,2]>.08)&(np.linalg.norm(pts,axis=1)<2)])
        vox=unique_voxels(np.floor(np.concatenate(cloud)/.01).astype(int))
        boxes=[Cuboid(name=f'stow_{i}',pose=((lo+hi+1)*.005).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.01).tolist()) for i,(lo,hi) in enumerate(merge_occupied(vox))]
        arm=sensor_arm_planner(robot_calibration(m,d,self.output/'stow_robot'),collision_cache={'cuboid':12000,'mesh':2})
        planner=ObservedGraspPlanner(arm,geo,boxes)
        control=MolmoSpacesController(m,d);q=control.robot_view.get_move_group('arm').joint_pos.copy();target=np.asarray(posture)
        path=smooth_joint_path(q,target)
        if not planner.clear(path):
            arm.planner.update_world(SceneCfg(cuboid=boxes));path=arm.plan_joints(q.tolist(),target.tolist())
            if not planner.clear(path):raise RuntimeError('Observed stow sweep blocked')
        hold={g:control.robot_view.get_move_group(g).noop_ctrl.copy() for g in ('base','arm','gripper')}
        for row in path:
            hold['arm']=row;hold['gripper']=np.array([0.]);control.step(hold)
            for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
            recorder.record('stow_after_open')
        if np.max(abs(control.robot_view.get_move_group('arm').joint_pos-target))>.05:raise RuntimeError('Stow tracking failed')
