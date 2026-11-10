"""Live fixture skill adapters using the qualified sensor contact primitives.

No reset or fixture-state feedback is permitted. Remembered hinge geometry is
RGB-D-derived and reanchored using the live SLAM pose before a later close call.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from ..skills.m2t2_interface import SkillOutcome
from ..control.sensor_appliance_backend import camera_points,pixel_point
from ..perception.handle_surface import observed_handle_surface


class LiveFixtureControls:
    def __init__(self,runtime,output):
        from ..perception.molmo_pointing import MolmoPointer
        from ..control.m2t2_policy import M2T2Policy
        self.runtime=runtime;self.output=Path(output);self.index=0;self.hinges={};self.target_observation=None
        self.pointer=MolmoPointer()
        self.model=M2T2Policy('/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth','external/M2T2')

    def handlers(self):
        return {name:(lambda _name=name,**kw:self.execute(_name,**kw))
                for name in ('open_door','close_door','open_drawer','close_drawer')}

    def packet(self):
        from .episode import episode_sensors
        with episode_sensors(self.runtime.model,self.runtime.data,**(self.runtime.sensor_config or {})) as sensor:
            return sensor.capture()

    def estimated_pose(self):
        packet=self.packet()
        if self.runtime.mapper is None:raise RuntimeError('Fixture motion requires an observed map/localization')
        result=self.runtime.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not result.accepted:raise RuntimeError('Fixture localization unavailable')
        return result.T_map_base.copy()

    def run_script(self,command,root):
        logfile=root/(Path(command[1]).stem+'.log')
        with logfile.open('w') as log:
            from .logging import TimestampedStream
            result=self.runtime.run(command,TimestampedStream(log))
        if result.returncode:raise RuntimeError('Fixture stage failed: '+str(logfile))

    def acquire_grasp(self,target,root,drawer=False,opening=False):
        from ..control.m2t2_policy import grasp_to_tcp
        packet=self.packet();cloud=[];colors=[];views=[]
        for name in ('wrist','navigation'):
            cam=packet.cameras[name];points,rgb,valid=camera_points(cam)
            cloud.append(points);colors.append(rgb);views.append((cam,valid))
        points=np.concatenate(cloud);rgb=np.concatenate(colors);found=None
        if self.target_observation is not None:
            observed_name,expected=self.target_observation
            self.target_observation=None
            if observed_name==target:
                from ..perception.handle_acquisition import observed_handle_near
                try:
                    # Microwave pointing can land on the backing panel beside
                    # its narrow edge. Reacquire measured protruding geometry,
                    # then retain the ordinary local grasp/path validation.
                    target_point,info=observed_handle_near(
                        points,expected,max_distance=.10 if target=='microwave' else .07)
                    surface,_=observed_handle_surface(points,target_point)
                    found=(target_point,surface,info)
                except ValueError:pass
        for camera,valid in ([] if found is not None else views):
            description=('middle of the vertical right edge of the movable microwave door, immediately left of the keypad; not the top edge or keypad'
                         if target=='microwave' else 'graspable handle on '+target)
            pointer=self.pointer
            if target=='microwave':
                from ..perception.fixture_contact import FixtureContactPointer
                pointer=FixtureContactPointer(pointer,target)
            elif any(word in target for word in ('cabinet','drawer','fridge')):
                from ..perception.fixture_pointing import ApplianceAssociatedPointer
                pointer=ApplianceAssociatedPointer(pointer,target)
            answer=pointer.point(Image.fromarray(camera.rgb),description)
            for pixel in answer['points']:
                try:
                    target_point=pixel_point(camera,valid,pixel)
                    surface,info=observed_handle_surface(points,target_point)
                    found=(target_point,surface,info);break
                except ValueError:continue
            if found is not None:break
        if found is None:raise RuntimeError('Named handle not visible: '+target)
        target_point,surface,info=found
        crop=np.linalg.norm(points-target_point,axis=1)<.18
        root.mkdir(parents=True);np.savez(root/'observed_world.npz',points=points,rgb=rgb)
        np.savez(root/'input_0.npz',points=points[crop],rgb=rgb[crop],target=target_point)
        rows=[]
        # The small handle may be missed when scene-point sampling favors its
        # backing panel. Resample the same observed crop before moving.
        import torch
        for sample in range(3):
            torch.manual_seed(sample)
            prediction=self.model.predict(points[crop],rgb[crop])
            for grasps,scores,contacts in zip(prediction['grasps'],prediction['grasp_confidence'],prediction['grasp_contacts']):
                rows.extend(dict(tcp=grasp_to_tcp(g).tolist(),confidence=float(s),contact=c.tolist())
                            for g,s,c in zip(grasps,scores,contacts))
            if any(np.linalg.norm(np.asarray(r['contact'])-target_point)<.03 for r in rows):break
        rows.sort(key=lambda row:-row['confidence'])
        (root/'grasps_0.json').write_text(json.dumps(dict(candidates=rows)))
        grasp=root/'grasp'
        command=[sys.executable,'scripts/test_m2t2_handle_execution.py','--source',str(root),'--output',str(grasp),
            '--target','0','--surface-filter','--align-panel','--center-handle','--max-candidates','30','--curobo-fallback',
            '--voxel-size','.01','--approach-stand-off','.12']
        if 'cabinet' in target:command+=['--prefer-deep-contact']
        if drawer:command+=['--grasp-ranking','horizontal']
        if target=='microwave':command+=['--center-length']
        else:command+=['--clearance-approach']
        if drawer and opening:command+=['--open-to-stop']
        self.run_script(command,root)
        report=json.loads((grasp/'report.json').read_text())
        if not report.get('physical_grasp_tested'):raise RuntimeError('No reachable handle grasp')
        # Robot encoder evidence only; private contact scores are not consumed.
        fingers=[float(self.runtime.data.joint(n).qpos[0]) for n in
            ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')]
        if min(fingers)>.79:raise RuntimeError('Handle grasp is empty by gripper encoders')
        if drawer and opening and report.get('stop_reason')!='resisted_motion':
            raise RuntimeError('Drawer did not reach an observed motion stop')
        return grasp,info

    def navigation_posture(self,root,progress):
        """Finish fixture contact with the posture required by pickup navigation."""
        from .episode import episode_sensors
        from ..navigation.local_scan import ScanConfig
        from ..control.fixture_posture import stow_after_fixture
        from ..recording.deferred import StateVideoRecorder
        with episode_sensors(self.runtime.model,self.runtime.data) as sensors:
            if np.allclose(sensors.capture().arm.position,ScanConfig().stowed_arm,atol=.05):return
            progress('checked_navigation_posture','running')
            with StateVideoRecorder(self.runtime.model,self.runtime.data,root/'posture_recording') as rec:
                # Exact calibrated base/finger geometry and a checked short
                # withdrawal avoid the opened panel during folding. Keep the
                # room-door / set-table posture backend unchanged.
                report=stow_after_fixture(self.runtime,sensors,rec,root/'posture',ScanConfig().stowed_arm)
                (root/'posture_report.json').write_text(json.dumps(report,indent=2))
            progress('checked_navigation_posture','completed')

    def execute(self,skill,*,target,progress):
        opening=skill.startswith('open');drawer=skill.endswith('drawer')
        if not drawer and not opening and target not in self.hinges:
            return SkillOutcome(False,'closed_panel_reference_unobserved')
        root=self.output/f'{self.index:03d}_{skill}';self.index+=1
        progress('observe_and_grasp','running')
        grasp,info=self.acquire_grasp(target,root,drawer,opening)
        progress('observe_and_grasp','completed')
        progress('articulate_release_withdraw','running')
        if drawer:
            self.drawer_contact(root,info,opening)
        else:
            output=root/'motion'
            command=[sys.executable,'scripts/test_sensor_hinged_cycle.py','--grasp-run',str(grasp),
                '--output',str(output),'--contact-ik','analytic','--contact-motion-profile','fast',
                '--base-follow-handle','--planar-push','--lateral-base-fraction','1',
                '--contact-retreat','--absolute-open-angle','90','--orientation-gain','.5']
            if target=='microwave':command+=['--base-yaw-follow']
            if opening:command+=['--open-only','--auto-probe-direction']
            else:
                T=np.linalg.inv(self.estimated_pose());memory=self.hinges[target]
                local=dict(closed_anchor=(T[:3,:3]@memory['closed_anchor']+T[:3,3]).tolist(),
                    closed_normal=(T[:3,:3]@memory['closed_normal']).tolist(),
                    hinge_xy=(T[:3,:3]@memory['hinge']+T[:3,3])[:2].tolist())
                path=root/'observed_close_reference.json';path.write_text(json.dumps(local))
                command+=['--close-observation',str(path)]
            self.run_script(command,root)
            result=json.loads((output/'report.json').read_text())
            if result.get('error'):raise RuntimeError(result['error'])
            if opening:
                T=self.estimated_pose()
                self.hinges[target]=dict(
                    hinge=T[:3,:3]@np.asarray(result['observed_hinge_base'])+T[:3,3],
                    closed_handle=T[:3,:3]@np.asarray(result['observed_closed_handle_base'])+T[:3,3],
                    closed_anchor=T[:3,:3]@np.asarray(result['observed_closed_panel_anchor_base'])+T[:3,3],
                    closed_normal=T[:3,:3]@np.asarray(result['observed_closed_panel_normal_base']))
                if result.get('observed_open_handle_base') is not None:
                    self.hinges[target]['open_handle']=T[:3,:3]@np.asarray(result['observed_open_handle_base'])+T[:3,3]
        progress('articulate_release_withdraw','completed')
        self.navigation_posture(root,progress)
        return SkillOutcome(True,'commands_completed_fixture_goal_not_scored')

    def aperture_observation(self,target,*,map_frame=False):
        """Return remembered opened geometry reanchored by current SLAM."""
        memory=self.hinges.get(target)
        if memory is None or 'closed_handle' not in memory:return None
        keys=('closed_handle','hinge','closed_normal')+(('open_handle',) if 'open_handle' in memory else ())
        if map_frame:return {key:np.asarray(memory[key]).tolist() for key in keys}
        T=np.linalg.inv(self.estimated_pose())
        return {key:(T[:3,:3]@np.asarray(memory[key])+
                     (0 if key=='closed_normal' else T[:3,3])).tolist()
                for key in keys}

    def drawer_contact(self,root,info,opening):
        from .appliance_controls import build_appliance_backend
        from ..control.sensor_drawer import SensorDrawerMotion
        from ..control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
        from curobo._src.geom.types import Cuboid
        from ..recording.deferred import StateVideoRecorder
        from ..control.contact_retreat import tool_pose
        phase='drawer_release'
        with StateVideoRecorder(self.runtime.model,self.runtime.data,root/'release_recording') as recorder:
            b=build_appliance_backend(self.runtime,output=root/'backend',command_observer=lambda:recorder.record(phase))
            def planner(current,goal,contact_patch):
                pts=np.concatenate([camera_points(cam)[0] for cam in self.packet().cameras.values()])
                pts=pts[(np.linalg.norm(pts,axis=1)<1.8)&(pts[:,2]>.08)]
                if contact_patch:
                    delta=goal[:3,3]-current[:3,3]
                    t=np.clip((pts-current[:3,3])@delta/max(delta@delta,1e-12),0,1)
                    pts=pts[np.linalg.norm(pts-current[:3,3]-t[:,None]*delta,axis=1)>.075]
                vox=np.unique(np.floor(pts/.015).astype(int),axis=0)
                boxes=[Cuboid(name=f'drawer_{i}',pose=((lo+hi+1)*.0075).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.015).tolist()) for i,(lo,hi) in enumerate(merge_voxels(vox))]
                return ObservedGraspPlanner(b.prepare_arm(),b.geometry,boxes)
            motion=SensorDrawerMotion(joint_position=b.joints,
                tcp_pose=lambda q=None:tool_pose(b.geometry,b.joints() if q is None else q),
                fingers=b.fingers,command=b.command,planner_factory=planner,outward_normal=info['panel_normal'])
            if not opening:
                phase='drawer_close';result=motion.move_until_resistance(opening=False)
                if not result.completed:raise RuntimeError(result.reason)
            phase='drawer_release';result=motion.release_and_withdraw()
            if not result.completed:raise RuntimeError(result.reason)
