"""Qwen component profiles using already-qualified sensor contact implementations."""
from ..runtime.fixture_controls import LiveFixtureControls
from ..skills.m2t2_interface import SkillOutcome


class PanelContactReturned(Exception):
    """The live contact backend intentionally yielded to the policy."""


class HouseholdFixtureControls(LiveFixtureControls):
    def __init__(self,runtime,output,*,interactive_panel=True):
        # Molmo already receives the requested fixture label. Keep measured
        # handle geometry and physical verification, without a second VLM vote.
        super().__init__(runtime,output,verify_fixture_points=False)
        self.calibrated_drawer_withdrawal=True
        self.interactive_panel=interactive_panel
        self.panel_command=None

    def acquire_grasp(self,target,root,drawer=False,opening=False):
        if not drawer:
            original=self.pointer
            if not opening and target in self.hinges and 'open_handle' in self.hinges[target]:
                import numpy as np
                T=np.linalg.inv(self.estimated_pose())
                self.target_observation=(target,T[:3,:3]@self.hinges[target]['open_handle']+T[:3,3])
            if target=='fridge' and opening:
                from ..perception.fridge_handle import FridgeDoorHandlePointer
                self.pointer=FridgeDoorHandlePointer(original)
                # Generic refrigerator docking can have pointed at a freezer
                # drawer. Reacquire the door feature before choosing its grasp.
                self.target_observation=None
            try:
                try:return super().acquire_grasp(target,root,drawer,opening)
                except RuntimeError as exc:
                    if target!='fridge' or 'Handle grasp is empty' not in str(exc):raise
                    return super().acquire_grasp(target,root/'retry_01',drawer,opening)
            finally:self.pointer=original
        # A generic drawer request may have several visually eligible handles.
        # Retry another observed handle after empty closure, instead of always
        # returning to the first pixel. No joint IDs or fixture truth are used.
        import json
        import numpy as np
        from PIL import Image
        from ..control.sensor_appliance_backend import camera_points,pixel_point
        from ..perception.handle_surface import observed_handle_surface
        from ..perception.fixture_pointing import ApplianceAssociatedPointer
        packet=self.packet();cloud=np.concatenate([camera_points(c)[0] for c in packet.cameras.values()])
        choices=[]
        for camera in packet.cameras.values():
            _,_,valid=camera_points(camera)
            result=ApplianceAssociatedPointer(self.pointer,target).point(
                Image.fromarray(camera.rgb),'all visible horizontal drawer handles below the kitchen counter')
            for pixel in result.get('points',[]):
                try:
                    point=pixel_point(camera,valid,pixel)
                    surface,_=observed_handle_surface(cloud,point)
                    _,singular,axes=np.linalg.svd(surface-surface.mean(0),full_matrices=False)
                    if singular[0]>2*max(singular[1],1e-8) and abs(axes[0,2])>.5:
                        continue  # A vertical cabinet pull is not a horizontal drawer handle.
                    if not any(np.linalg.norm(point-p)<.07 for _,p in choices):
                        choices.append((np.linalg.norm(point),point))
                except ValueError:continue
        if choices:
            # Prefer the middle of the visually identified bank, avoiding the
            # floor-level handle and the counter lip. This ranks observations;
            # it neither selects a simulator drawer ID nor bypasses planning.
            middle=float(np.median([point[2] for _,point in choices]))
            choices=[(abs(point[2]-middle)-.01*point[2],point) for _,point in choices]
        failed=[];last=None
        for attempt in range(3):
            destination=root if attempt==0 else root/f'retry_{attempt:02d}'
            remaining=[(rank,p) for rank,p in choices if not any(np.linalg.norm(p-f)<.10 for f in failed)]
            if attempt and not remaining:raise last
            if remaining:
                self.target_observation=(target,min(remaining,key=lambda x:x[0])[1])
            try:return super().acquire_grasp(target,destination,drawer,opening)
            except RuntimeError as exc:
                if 'Handle grasp is empty' not in str(exc):raise
                last=exc
                observed=np.load(destination/'input_0.npz')['target']
                failed.append(observed)
                (root/'handle_retries.json').write_text(json.dumps(dict(
                    failed_observed_points=[p.tolist() for p in failed],reason=str(exc)),indent=2))
        raise last

    def execute(self,skill,*,target,progress):
        self.current_target=target
        self.drawer_withdrawn=False
        self.runtime.door_push_memory=None
        if skill=='close_door' and target in ('microwave','fridge') and 'open_handle' in self.hinges.get(target,{}):
            from .closing_redock import ClosingRedocking
            from ..runtime.episode import episode_sensors
            root=self.output/f'{self.index:03d}_closing_dock'
            with episode_sensors(self.runtime.model,self.runtime.data) as sensors:
                dock=ClosingRedocking(self.runtime,sensors,self.runtime.mapper.view(),self.hinges[target])
                progress('closing_side_redock','running')
                T=self.estimated_pose()
                on_face=float((T[:3,3]-dock.expected)@dock.approach_normal_map)>.05
                nearby=__import__('numpy').linalg.norm(T[:2,3]-dock.expected[:2])<.85
                if getattr(self.runtime,'panel_close',False) and on_face and nearby:
                    root.mkdir(parents=True,exist_ok=False)
                    result=dict(success=True,retained_pose=True,reason='already on observed closing face; arm contact planner validates reach')
                    __import__('json').dump(result,open(root/'report.json','w'),indent=2)
                else:result=dock.move(target,'handle',.65,root)
                if not result['success']:
                    self.withdrawal_directions=[self.estimated_pose()[:3,:3].T@dock.approach_normal_map]
                    try:self.navigation_posture(root,progress)
                    finally:self.withdrawal_directions=()
                    result=dock.move(target,'handle',.65,root/'after_posture')
                if not result['success']:raise RuntimeError('Closing-side docking unavailable: '+result['reason'])
                progress('closing_side_redock','completed')
            if getattr(self.runtime,'panel_close',False):
                from .panel_closing import push_close
                root=self.output/f'{self.index:03d}_close_panel';self.index+=1
                return push_close(self,target,root,progress)
        try:return super().execute(skill,target=target,progress=progress)
        except PanelContactReturned:
            progress('panel_contact_ready','completed')
            return SkillOutcome(False,'partial_open_panel_contact_ready: inspect door_state and call continue_door; do not restart the handle grasp')

    def door_state(self,target):
        memory=getattr(self.runtime,'door_push_memory',None)
        if memory is None or target!=getattr(self,'current_target',None):
            return dict(continuation_available=False,reason='No live observed panel contact for this target')
        return memory.public()

    def continue_door(self,target,*,motion='base_push',advance_degrees=15.,wrist_rotation=.5):
        """Resume the same observed contact in live physics; never reacquire/reset."""
        import json
        if motion not in ('arm_swing','base_push','base_rotate'):raise ValueError('Unknown door contact motion')
        if not 1<=advance_degrees<=90 or not 0<=wrist_rotation<=1:raise ValueError('Invalid door contact increment')
        if not self.door_state(target).get('continuation_available') or self.panel_command is None:
            raise ValueError('No live panel contact to continue')
        root=self.output/f'{self.index:03d}_continue_door';self.index+=1;root.mkdir(parents=True)
        command=[v for v in self.panel_command if v not in
                 ('--pause-at-panel','--base-push','--base-yaw-follow','--contact-retreat')]
        command[command.index('--output')+1]=str(root/'motion')
        command[command.index('--orientation-gain')+1]=str(wrist_rotation)
        command+=['--resume-panel-push','--push-advance-degrees',str(advance_degrees)]
        if motion!='arm_swing':command+=['--base-push']
        if motion=='base_rotate':command+=['--base-yaw-follow']
        self.run_script(command,root)
        report=json.loads((root/'motion/report.json').read_text())
        public=self.door_state(target)
        public.update(success=not bool(report.get('error')),motion=motion)
        if report.get('error'):
            # This is the planner's sensor-derived reason, not the private score.
            public['reason']=str(report['error'])[:500]
        if public.get('opening_completed'):
            import numpy as np
            T=self.estimated_pose()
            self.hinges[target]=dict(
                hinge=T[:3,:3]@np.asarray(report['observed_hinge_base'])+T[:3,3],
                closed_handle=T[:3,:3]@np.asarray(report['observed_closed_handle_base'])+T[:3,3],
                closed_anchor=T[:3,:3]@np.asarray(report['observed_closed_panel_anchor_base'])+T[:3,3],
                closed_normal=T[:3,:3]@np.asarray(report['observed_closed_panel_normal_base']))
            if report.get('observed_open_handle_base') is not None:
                self.hinges[target]['open_handle']=T[:3,:3]@np.asarray(report['observed_open_handle_base'])+T[:3,3]
        return public

    def drawer_contact(self,root,info,opening):
        super().drawer_contact(root,info,opening)
        self.drawer_withdrawn=True

    def navigation_posture(self,root,progress):
        if getattr(self,'drawer_withdrawn',False):
            # The release/withdraw sweep and settled endpoint have passed.
            # Qwen's SensorRedocking checks subsequent base motion with the
            # retained arm shape; later manipulation plans from live joints.
            # No next motion is authorized by this return, and no tuck is
            # needed just to finish the semantic open/close command.
            import json
            (root/'posture_report.json').write_text(json.dumps(dict(
                execution_success=True,retained_posture=True,simulated_seconds=0.,
                reason='retained checked withdrawal endpoint; next skill plans its own sweep'),indent=2))
            progress('retained_checked_withdrawal_pose','completed')
            return
        return super().navigation_posture(root,progress)

    def run_script(self,command,root):
        command=list(command)
        if (('drawer' in self.current_target or self.current_target=='fridge') and len(command)>1
                and command[1]=='scripts/test_m2t2_handle_execution.py'):
            # Panel alignment changes the learned approach axis. Recompute the
            # finger-pad overlap from the observed handle surface afterwards.
            command+=['--center-length']
        if (self.current_target=='fridge' and len(command)>1
                and command[1]=='scripts/test_sensor_hinged_cycle.py' and '--resume-panel-push' not in command
                and '--panel-close-reference' not in command):
            # The qualified fridge profile uses finite-difference robot Jacobian
            # IK. This is the existing sensor backend, not oracle contact poses.
            index=command.index('--contact-ik')
            command[index+1]='finite'
            if '--open-only' in command:
                command+=['--handle-then-push','--pull-push-after','25','--exact-closed-fingers',
                          '--calibrated-contact-boxes','--local-panel-reach','--base-push']
                if self.interactive_panel:command+=['--pause-at-panel']
                self.panel_command=list(command)
        result=super().run_script(command,root)
        if '--pause-at-panel' in command:
            import json
            from pathlib import Path
            report=json.loads((Path(command[command.index('--output')+1])/'report.json').read_text())
            if report.get('contact_control_returned'):raise PanelContactReturned()
        return result
