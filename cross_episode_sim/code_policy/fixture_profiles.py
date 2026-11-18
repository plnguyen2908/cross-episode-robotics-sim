"""Qwen component profiles using already-qualified sensor contact implementations."""
from ..runtime.fixture_controls import LiveFixtureControls
from ..skills.m2t2_interface import SkillOutcome


class PanelContactReturned(Exception):
    """The live contact backend intentionally yielded to the policy."""


DRAWER_HANDLE_MAX_HEIGHT=.08
# SLAM drift between opening and returning was 11 cm in fix80, and the open
# drawer itself moved 6 cm in fix84. The other drawers' handles are an opening
# distance (38 cm) deeper, so 18 cm (the handle finder's limit) cannot reach them.
REMEMBERED_HANDLE_TOLERANCE=.18
# Appliance doors pushed shut from the door memory, like cabinet doors.
PUSH_CLOSED_APPLIANCES=('microwave','fridge')
# Fridge door angle (deg) at which the opening lets go of the handle and pushes the
# leaf from inside. 18 was tried after the handle slid out of the hand at 22 deg
# (fix107): at 20 deg the gap was too small for the hand to get behind the edge.
FRIDGE_PULL_PUSH_AFTER=25
# How the fridge door is opened: 'handle_then_push' (pull to FRIDGE_PULL_PUSH_AFTER,
# let go, push the leaf from inside) or 'pull' (hold the handle all the way with
# the base following, as for the cabinet and microwave doors).
FRIDGE_OPENING='handle_then_push'


def fridge_opening():
    import os
    # Diagnostic override for isolated comparisons only; unset in runs.
    return os.environ.get('FRIDGE_OPENING_OVERRIDE',FRIDGE_OPENING)


# Times the fridge opening's inside push is resumed from its live contact after stopping early.
FRIDGE_PUSH_RESUMES=2
# Most pushes (stand, push, re-stand) spent on closing one door.
CLOSE_STROKES=4
# Doors that get one far-side push when the closing push cannot start, and the
# least rotation (deg) it must give for the close to go on.
FAR_SIDE_PUSH_DOORS=('fridge',)
FAR_SIDE_PUSH_MIN_DEGREES=8.
# Doors pressed against their frame after a push-close, and how far the hand may move in (m).
# Empty since tree -fixdr: the press missed the door by 5 cm in two replays and
# once left the arm unable to fold (fix130); the push-close itself now aims
# slightly past closed instead. The press stays available for a named door.
CLOSED_PRESS_DOORS=()
CLOSED_PRESS_TRAVEL=.18
CLOSED_PRESS_WITHDRAW=.20
# Straight back-off (m) before a last search for a closing-side stand spot.
CLOSING_BACK_OFF=.40
# Times a push-close that could not start is tried again from a new stand pose (doors without a far-side push).
PUSH_START_RETRIES=1


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
                # fix63: when the remembered handle was not reacquired, a fresh
                # pointing grabbed the dishwasher door. Closing uses only it.
                self.require_target_observation=True
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
                    if float(np.ptp(surface[:,2]))>DRAWER_HANDLE_MAX_HEIGHT:
                        # A horizontal drawer pull is a thin bar. An 18 x 17 cm
                        # blob between two handles was chosen first in fix75
                        # and the gripper closed on nothing.
                        continue
                    if not any(np.linalg.norm(point-p)<.07 for _,p in choices):
                        choices.append((np.linalg.norm(point),point))
                except ValueError:continue
        remembered=None if opening else self.drawer_observation(target)
        if remembered is not None:
            # Closing: the drawer this robot opened, at the handle it remembers
            # (reanchored by SLAM). The other drawers' handles sit an opening
            # distance deeper, so the nearest protruding handle is this one.
            choices=[(0.,np.asarray(remembered['open_handle'],float))]
            self.target_observation_tolerance=REMEMBERED_HANDLE_TOLERANCE
            print('DRAWER_REMEMBERED_HANDLE',json.dumps(dict(base_point=np.round(choices[0][1],3).tolist())),flush=True)
        elif choices:
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

    def _closing_stroke(self,target,progress):
        """Stand on the closing side of the remembered leaf and push it; None when the push could not start."""
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
            else:
                try:result=dock.move(target,'handle',.65,root)
                except RuntimeError as exc:
                    # No stand spot could even be looked for: the push and the
                    # other fallbacks still run from where the robot is.
                    result=dict(success=False,reason=str(exc)[:160])
            if not result['success']:
                self.withdrawal_directions=[self.estimated_pose()[:3,:3].T@dock.approach_normal_map]
                try:
                    self.navigation_posture(root,progress)
                    result=dock.move(target,'handle',.65,root/'after_posture')
                except RuntimeError as exc:
                    # A failed fold or second dock must not end the close
                    # before the push and handle fallbacks have run.
                    print('CLOSING_SIDE_POSTURE_RETRY_FAILED',str(exc)[:160],flush=True)
                finally:self.withdrawal_directions=()
            if not result['success']:
                # Still no route to a stand spot: the robot can be boxed in next
                # to the open door after placing the food (fix140: 0.34 m from
                # the open handle, both searches without a route, microwave
                # left open). Back straight off into the room and search again.
                try:
                    dock.back_off(CLOSING_BACK_OFF,root/'back_off')
                    result=dock.move(target,'handle',.65,root/'after_back_off')
                except RuntimeError as exc:print('CLOSING_SIDE_BACK_OFF_FAILED',str(exc)[:160],flush=True)
            if not result['success']:
                # The push planner validates reach from the live pose itself;
                # a failed stand-spot search should not end the close before
                # the push and handle fallbacks run.
                print('CLOSING_SIDE_DOCK_FAILED_PUSH_FROM_HERE',result['reason'],flush=True)
            progress('closing_side_redock','completed')
        from .panel_closing import push_close,PartialPanelClose
        root=self.output/f'{self.index:03d}_close_panel';self.index+=1
        # Push the open leaf shut from its closing side with the closed hand
        # (no handle grasp). A cabinet's opened handle faces sideways and was
        # not re-observable for a grasp (fix65).
        try:return push_close(self,target,root,progress)
        except PartialPanelClose:raise
        except RuntimeError as exc:
            print('PUSH_CLOSE_FAILED_TRY_HANDLE',str(exc)[:200],flush=True)
            return None

    def _press_closed(self,target,progress):
        """Best effort: press a just-closed door against its frame until it resists.

        A push-close ends where the robot measures the leaf at its remembered
        closed position; from a poor stand pose that left the microwave door
        2-2.5 deg ajar in two isolated replays of fix125's second close (first
        closes ended shut). The closed hand goes a stand-off in front of the
        remembered closed handle and moves in until resistance, at most
        CLOSED_PRESS_TRAVEL. Any failure leaves the close as it was.
        """
        import json,numpy as np
        from .closing_redock import ClosingRedocking
        from .panel_closing import advance_open_memory
        from ..runtime.episode import episode_sensors
        kept=self.hinges[target];memory={k:np.array(v,float) for k,v in kept.items()}
        advance_open_memory(memory,180.)          # remembered leaf at its closed position
        root=self.output/f'{self.index:03d}_closed_press';self.index+=1;root.mkdir(parents=True,exist_ok=True)
        self.hinges[target]=memory
        try:
            progress('closed_press','running')
            with episode_sensors(self.runtime.model,self.runtime.data) as sensors:
                dock=ClosingRedocking(self.runtime,sensors,self.runtime.mapper.view(),memory)
                try:dock.move(target,'handle',.65,root/'dock')
                except (RuntimeError,ValueError) as exc:print('CLOSED_PRESS_DOCK_FAILED',str(exc)[:160],flush=True)
            # Withdraw well clear of the door afterwards: left 5 cm from the
            # handle bar (press blocked before it moved, fix130) the arm had no
            # collision-checked path to fold, and the next call failed on it.
            _,reason=self.door_hook_push(target,root/'press',travel_budget=CLOSED_PRESS_TRAVEL,withdraw=CLOSED_PRESS_WITHDRAW)
            print('CLOSED_PRESS',json.dumps(dict(target=target,stop=str(reason)[:60])),flush=True)
            progress('closed_press','completed')
        except (RuntimeError,ValueError) as exc:
            print('CLOSED_PRESS_SKIPPED',str(exc)[:200],flush=True)
        finally:
            self.hinges[target]=kept
            try:
                # Fold by first drawing the hand straight out from the door.
                import numpy as np
                outward=np.array(memory['closed_normal'],float);outward[2]=0.
                self.withdrawal_directions=[self.estimated_pose()[:3,:3].T@outward]
                self.navigation_posture(root,progress)
            except RuntimeError as exc:print('CLOSED_PRESS_FOLD_FAILED',str(exc)[:160],flush=True)
            finally:self.withdrawal_directions=()

    def execute(self,skill,*,target,progress):
        outcome=self._execute(skill,target=target,progress=progress)
        if (skill=='close_door' and target in CLOSED_PRESS_DOORS and getattr(outcome,'success',False)
                and str(getattr(outcome,'reason','')).startswith('observed_panel_push_completed')
                and 'open_handle' in self.hinges.get(target,{})):
            self._press_closed(target,progress)
        return outcome

    def _execute(self,skill,*,target,progress):
        self.current_target=target
        self.drawer_withdrawn=False
        self.runtime.door_push_memory=None
        # Every door this robot opened and remembers (cabinet, microwave,
        # fridge) is closed the same way: stand on the remembered leaf's
        # closing side and push it shut; the handle grasp is the fallback.
        push_door=(skill=='close_door' and ('cabinet' in target or target in PUSH_CLOSED_APPLIANCES)
            and 'open_handle' in self.hinges.get(target,{}))
        if push_door:
            # In strokes, like the datagen oracle: when a push turns the leaf
            # part of the way and stops, stand again for the leaf as it is now.
            from .panel_closing import PartialPanelClose,ShortPanelClose,advance_open_memory
            far_side_tried=False;pushed_short=None;not_started=0
            for stroke in range(CLOSE_STROKES):
                try:
                    outcome=self._closing_stroke(target,progress)
                    if outcome is not None:return outcome
                    # A finishing stroke that could not start does not undo a
                    # push that already ran to its end.
                    if pushed_short is not None:return pushed_short
                    if target not in FAR_SIDE_PUSH_DOORS:
                        # The push could not start from this stand pose (fix125
                        # replays: no panel fit, or the hand's line blocked).
                        # Stand again once before giving the push up.
                        not_started+=1
                        if not_started<=PUSH_START_RETRIES:
                            print('PUSH_CLOSE_START_RETRY',__import__('json').dumps(dict(target=target,attempt=not_started)),flush=True)
                            continue
                        break
                    if far_side_tried:break
                    # The push could not start: no stand spot on the closing
                    # side lets it see and reach the far face (fridge door
                    # 0.45 m from the wall, fix113). Reach round the free edge
                    # and push from the far side to make room, as the datagen
                    # oracle does, then stand again for the turned leaf.
                    far_side_tried=True
                    root=self.output/f'{self.index:03d}_far_side_push';self.index+=1
                    progress('far_side_push','running')
                    try:turned,_=self.door_hook_push(target,root)
                    except RuntimeError as far_error:
                        print('FAR_SIDE_PUSH_FAILED',str(far_error)[:200],flush=True);break
                    progress('far_side_push','completed')
                    try:self.navigation_posture(root,progress)
                    except RuntimeError as fold_error:print('FAR_SIDE_PUSH_FOLD_FAILED',str(fold_error)[:160],flush=True)
                    if turned<FAR_SIDE_PUSH_MIN_DEGREES:break
                    raise PartialPanelClose(turned,'far-side push')
                except PartialPanelClose as exc:
                    if isinstance(exc,ShortPanelClose):
                        pushed_short=SkillOutcome(True,'observed_panel_push_completed; final goal scored separately')
                    remaining=advance_open_memory(self.hinges[target],exc.degrees)
                    print('PUSH_CLOSE_STROKE',__import__('json').dumps(dict(target=target,stroke=stroke+1,
                        turned_degrees=round(exc.degrees,1),remembered_remaining_degrees=round(remaining,1),reason=str(exc)[:120])),flush=True)
            else:
                if pushed_short is not None:return pushed_short
                print('PUSH_CLOSE_FAILED_TRY_HANDLE',f'door still open after {CLOSE_STROKES} strokes',flush=True)
        try:return super().execute(skill,target=target,progress=progress)
        except PanelContactReturned:
            progress('panel_contact_ready','completed')
            return SkillOutcome(False,'partial_open_panel_contact_ready: inspect door_state and call continue_door; do not restart the handle grasp')
        except RuntimeError as exc:
            if not (skill=='open_door' and target=='fridge'):raise
            outcome=self._continue_fridge_opening(target,progress,exc)
            if outcome is None:raise
            return outcome

    def _continue_fridge_opening(self,target,progress,error):
        """After the push phase stopped early with the hand on the leaf, push on from the same contact.

        The fridge opening lets go of the handle at 25 deg and pushes the leaf
        from inside; that push ended at 32 deg in an isolated run with no error
        of its own. The live contact is still remembered, so the same backend
        resumes it (as the interactive continue_door does) before the call is
        given up. Returns None when there is nothing to resume.
        """
        import json
        for attempt in range(FRIDGE_PUSH_RESUMES):
            state=self.door_state(target)
            if not state.get('continuation_available') or self.panel_command is None:return None
            print('FRIDGE_OPEN_PUSH_RESUMED',json.dumps(dict(attempt=attempt+1,observed_open_degrees=round(float(state.get('observed_open_angle_degrees',0.)),1),
                after=str(error)[:120])),flush=True)
            try:state=self.continue_door(target,motion='base_push',advance_degrees=90.)
            except (RuntimeError,ValueError) as exc:
                print('FRIDGE_OPEN_PUSH_RESUME_FAILED',str(exc)[:160],flush=True);return None
            if state.get('opening_completed') and 'open_handle' in self.hinges.get(target,{}):
                root=self.output/f'{self.index:03d}_resumed_open_posture';self.index+=1;root.mkdir(parents=True,exist_ok=True)
                self.runtime.door_push_memory=None
                self.navigation_posture(root,progress)
                return SkillOutcome(True,'commands_completed_fixture_goal_not_scored')
        return None

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

    def drawer_contact(self,root,info,opening,**options):
        super().drawer_contact(root,info,opening,**options)
        self.drawer_withdrawn=True

    def navigation_posture(self,root,progress):
        # Live fix58: the drawer's retained arm-out withdrawal pose went into a
        # cross-room drive to the dining table; the robot never moved and the
        # pick redock stopped 0.96 m from the cups. Fold to the driving posture
        # after every fixture call, as door calls already do.
        self.drawer_withdrawn=False
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
            if '--open-only' in command and fridge_opening()=='handle_then_push':
                command+=['--handle-then-push','--pull-push-after',str(FRIDGE_PULL_PUSH_AFTER),'--exact-closed-fingers',
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
