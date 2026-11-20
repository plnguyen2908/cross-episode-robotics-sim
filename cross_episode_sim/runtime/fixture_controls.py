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


def handle_description(target, opening):
    if target == 'microwave':
        if opening:
            return 'middle of the vertical right edge of the movable microwave door, immediately left of the keypad; not the top edge or keypad'
        return 'vertical free edge or handle of the OPEN microwave door swung away from the appliance; not the keypad, cavity rim, shelf or fixed appliance body'
    if target == 'fridge' and not opening:
        return 'vertical handle on the OPEN refrigerator door swung away from the fridge; not the freezer drawer or interior shelf'
    return 'graspable handle on '+target


# Height (m) at which the fridge door's long vertical handle is grasped.
FRIDGE_HANDLE_GRASP_HEIGHT=1.05
# A handle found farther than this from the base (m) is grasped again after standing
# CLOSER_GRASP_DISTANCE in front of it (fix162: 96 grasps unreachable from 1.0 m).
CLOSER_GRASP_MIN_REACH=.80
CLOSER_GRASP_DISTANCE=.65
MICROWAVE_HANDLE_GRASP_HEIGHT=1.10


def slides_toward(height,before,after):
    """Whether moving a grasp point from height `before` to `after` brings it nearer the requested height."""
    return abs(float(after)-height)<abs(float(before)-height)
# The close-up re-acquisition of the microwave handle may move the grasp at most this far
# (m) up or down from the height the stand-spot search measured on the bar.
MICROWAVE_GRASP_HEIGHT_DRIFT=.03
# Far-side door push (door_hook_push): the closed hand starts this far outside the
# remembered opened-handle position, at the handle or a little toward the free
# edge, and sweeps at most DOOR_HOOK_TRAVEL toward the closed side. The leaf
# starts to turn after about DOOR_HOOK_CONTACT_AFTER of that sweep (isolated on
# the fix113 fridge state: 0.44 m of hand travel turned the door 24.7 deg at a
# 0.67 m handle radius, i.e. 0.28 m of leaf travel).
DOOR_HOOK_STANDOFF=.10
DOOR_HOOK_PAST_HANDLE=(.04,0.,.08)
DOOR_HOOK_TRAVEL=.44
DOOR_HOOK_CONTACT_AFTER=.16
# Observed points taken as the pushed leaf itself: within this of the remembered leaf
# line on the far side, up to the sweep depth on the near side, this far past the
# handle along the leaf, and within this height band around the handle.
DOOR_HOOK_LEAF_HALF_THICKNESS=.13
# Pushing a part-opened leaf further open from inside: the closed hand starts this
# far inside the remembered handle position (handle bar, leaf and any inner rim lie
# within DOOR_PUSH_OPEN_LEAF_DEPTH of it) and the leaf starts to turn after about
# DOOR_PUSH_OPEN_CONTACT_AFTER of the sweep.
DOOR_PUSH_OPEN_STANDOFF=.22
# Start positions along the leaf from the handle (m, negative toward the hinge). Nearer the
# hinge the gap between leaf and appliance was too narrow to reach (isolated, fix115 state).
DOOR_PUSH_OPEN_PAST_HANDLE=(0.,-.04,.04)
# Hand travel per opening sweep (m). Short, so the straight sweep stays on the turning
# leaf (0.44 m slid off the microwave door's edge after 10 deg); each sweep is followed
# by a look at the leaf and a new start beside it.
DOOR_PUSH_OPEN_TRAVEL=.32
DOOR_PUSH_OPEN_LEAF_DEPTH=.17
DOOR_PUSH_OPEN_CONTACT_AFTER=.10
DOOR_HOOK_LEAF_PAST_HANDLE=.12
DOOR_HOOK_LEAF_BAND=.30
# A handle pull that stopped with at least this opening (deg) measured is finished
# by pushing from inside: up to PUSH_OPEN_ATTEMPTS sweeps, each followed by a look
# at the leaf, ended when it is seen within PUSH_OPEN_DONE_WITHIN_DEGREES of the
# requested opening or a sweep turns it less than PUSH_OPEN_MIN_PROGRESS_DEGREES.
PARTIAL_OPENING_MIN_DEGREES=25.
PUSH_OPEN_ATTEMPTS=7
PUSH_OPEN_DONE_WITHIN_DEGREES=6.
PUSH_OPEN_MIN_PROGRESS_DEGREES=4.
# After an opening finished from beside the free edge, the robot returns to this distance (m)
# in front of the opening.
OPENED_FRONT_DISTANCE=1.0
FRONT_STAND_AFTER_OPENING=('fridge',)
# A push-open sweep that ends against resistance with the leaf seen within this
# many degrees of the requested opening has reached the door's stop.
PUSH_OPEN_STOP_WITHIN_DEGREES=12.


def opened_to_its_stop(seen,requested,stop):
    """True when a push-open sweep ended on resistance with the leaf seen near the requested opening."""
    return bool(seen is not None and str(stop).startswith('resisted_motion')
                and 0.<=float(requested)-float(seen)<=PUSH_OPEN_STOP_WITHIN_DEGREES)
# Door openings whose base-follow steps may stand on floor the episode map has already observed.
FLOOR_MEMORY_OPENINGS=('fridge',)
MIN_DRAWER_OPENING=.15
DRAWER_OPEN_TARGET=.38
DRAWER_ARM_PULL=.28
DRAWER_CLOSE_OVERTRAVEL=.12
# Closed hand this far in front of the remembered handle before a grasp-free push.
DRAWER_PUSH_STANDOFF=.04


class LiveFixtureControls:
    def __init__(self,runtime,output,*,verify_fixture_points=True):
        from ..perception.molmo_pointing import MolmoPointer
        from ..control.m2t2_policy import M2T2Policy
        self.runtime=runtime;self.output=Path(output);self.index=0;self.hinges={};self.target_observation=None;self.drawers={}
        self.verify_fixture_points=verify_fixture_points
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
                    tolerance=getattr(self,'target_observation_tolerance',None);self.target_observation_tolerance=None
                    target_point,info=observed_handle_near(
                        points,expected,max_distance=tolerance or (.10 if target=='microwave' else .07))
                    if target=='microwave' and opening:
                        # From the grasp pose the arm can hide most of the bar;
                        # the nearest visible piece was then its top end, 6 cm
                        # above the height measured from the stand-spot view,
                        # and the fingers closed on nothing (fix135). Keep the
                        # grasp near the measured height.
                        drift=float(target_point[2]-expected[2])
                        if abs(drift)>MICROWAVE_GRASP_HEIGHT_DRIFT:
                            kept=np.r_[target_point[:2],expected[2]+np.sign(drift)*MICROWAVE_GRASP_HEIGHT_DRIFT]
                            print('MICROWAVE_HANDLE_HEIGHT_KEPT',json.dumps(dict(measured_z=round(float(expected[2]),3),
                                reacquired_z=round(float(target_point[2]),3),grasp_z=round(float(kept[2]),3))),flush=True)
                            target_point=kept
                    surface,_=observed_handle_surface(points,target_point)
                    found=(target_point,surface,info)
                except ValueError:pass
        if found is None and getattr(self,'require_target_observation',False):
            self.require_target_observation=False
            raise RuntimeError('Remembered opened handle not observed near its remembered position')
        self.require_target_observation=False
        for camera,valid in ([] if found is not None else views):
            description=handle_description(target,opening)
            pointer=self.pointer
            if target=='microwave' and opening:
                from ..perception.fixture_contact import FixtureContactPointer
                pointer=FixtureContactPointer(pointer,target)
            elif self.verify_fixture_points and any(word in target for word in ('cabinet','drawer','fridge')):
                from ..perception.fixture_pointing import ApplianceAssociatedPointer
                pointer=ApplianceAssociatedPointer(pointer,target)
            answer=pointer.point(Image.fromarray(camera.rgb),description)
            bodies=[]
            if target=='microwave' and opening:
                # Same rule as the redock: only a handle on the microwave body
                # this view shows (not the cabinet door under it, fix102).
                from ..perception.fixture_pointing import appliance_body_points
                bodies=appliance_body_points(self.pointer,camera,valid,'microwave')
            for pixel in answer['points']:
                try:
                    target_point=pixel_point(camera,valid,pixel)
                    if target=='fridge' and opening:
                        # Only the door's vertical bar; never the freezer pull.
                        from ..perception.handle_surface import vertical_bar_handle
                        if not vertical_bar_handle(points,target_point):
                            print('HANDLE_NOT_A_VERTICAL_BAR',json.dumps(dict(target=target,point=np.round(target_point,3).tolist())),flush=True)
                            continue
                    if bodies:
                        from ..perception.fixture_pointing import on_appliance_body
                        if not on_appliance_body(target_point,bodies):
                            print('HANDLE_NOT_ON_APPLIANCE',json.dumps(dict(target=target,point=np.round(target_point,3).tolist())),flush=True)
                            continue
                    surface,info=observed_handle_surface(points,target_point)
                    found=(target_point,surface,info);break
                except ValueError:continue
            if found is not None:break
        if found is None:raise RuntimeError('Named handle not visible: '+target)
        target_point,surface,info=found
        if target=='fridge' and opening:
            # Grasp the long door handle at a height the arm reaches, not where
            # the pointer happened to mark it (1.40 m: no reachable grasp).
            from ..perception.handle_surface import slide_along_vertical_bar
            moved=slide_along_vertical_bar(points,target_point,FRIDGE_HANDLE_GRASP_HEIGHT)
            if np.linalg.norm(moved-target_point)>.01:
                try:
                    surface,info=observed_handle_surface(points,moved)
                    print('FRIDGE_HANDLE_GRASP_HEIGHT',json.dumps(dict(pointed_z=round(float(target_point[2]),3),grasp_z=round(float(moved[2]),3))),flush=True)
                    target_point=moved
                except ValueError:pass
        if target=='microwave' and opening:
            # Grasp the bar at a comfortable height, as for the fridge. Every
            # grasp that closed on nothing was at the bar's top end (1.28-1.29 m:
            # fix115, fix135, fix141; from the grasp pose only that end was one
            # connected patch); every grasp that held was at 1.05-1.16 m. On all
            # nine saved views the bar is measured at 1.10 m.
            from ..perception.handle_surface import slide_along_vertical_bar
            lower=slide_along_vertical_bar(points,target_point,MICROWAVE_HANDLE_GRASP_HEIGHT)
            if np.linalg.norm(lower-target_point)>.01 and not slides_toward(MICROWAVE_HANDLE_GRASP_HEIGHT,target_point[2],lower[2]):
                # The followed column led away from the requested height: off
                # the bar's bottom end, down the door edge to the cabinet handle
                # under the microwave (fix148 call 3: pointed 0.97 m, slid to
                # 0.84 m, then moved to the cabinet handle bar at 0.72 m, 'No
                # reachable handle grasp'). Keep the pointed point.
                print('MICROWAVE_HANDLE_SLIDE_REFUSED',json.dumps(dict(pointed_z=round(float(target_point[2]),3),
                    slid_z=round(float(lower[2]),3))),flush=True)
                lower=target_point
            if np.linalg.norm(lower-target_point)>.01:
                try:
                    surface,info=observed_handle_surface(points,lower)
                    print('MICROWAVE_HANDLE_GRASP_HEIGHT',json.dumps(dict(pointed_z=round(float(target_point[2]),3),grasp_z=round(float(lower[2]),3))),flush=True)
                    target_point=lower
                except ValueError:pass
        if target=='microwave' and opening:
            # The microwave handle is a vertical bar when it has one. A point
            # that is not on a bar (countertop lip under the door, fix123; the
            # bar's top end, fix115) moves to the middle of the bar beside it.
            from ..perception.handle_surface import vertical_bar_handle,nearby_vertical_bar,vertical_bar_middle,bar_matches_point
            if not vertical_bar_handle(points,target_point):
                # Same door only: the bar must reach the point's height
                # (cabinet handles below and above are bars too).
                bar=nearby_vertical_bar(points,target_point,
                    accept=lambda seed:bar_matches_point(points,seed,float(target_point[2])))
                if bar is not None:
                    moved=vertical_bar_middle(points,bar)
                    try:
                        surface,info=observed_handle_surface(points,moved)
                        print('MICROWAVE_HANDLE_MOVED_TO_BAR',json.dumps(dict(pointed=np.round(target_point,3).tolist(),
                            grasp=np.round(moved,3).tolist())),flush=True)
                        target_point=moved
                    except ValueError:pass
        crop=np.linalg.norm(points-target_point,axis=1)<.18
        root.mkdir(parents=True);np.savez(root/'observed_world.npz',points=points,rgb=rgb)
        np.savez(root/'input_0.npz',points=points[crop],rgb=rgb[crop],target=target_point)
        rows=[];proposal_model='m2t2'
        from ..perception.rpc import configured_client
        graspgenx=configured_client('graspgenx')
        if graspgenx is not None:
            # Opt-in by endpoint, like object picks: propose on the separated
            # handle surface. M2T2 below remains the fallback when GraspGen-X
            # returns no grasp facing the panel.
            from ..control.graspgenx_policy import handle_proposals
            rows=handle_proposals(graspgenx,surface,info['panel_normal'],target=target_point)
            if rows:proposal_model='graspgenx'
        # The small handle may be missed when scene-point sampling favors its
        # backing panel. Resample the same observed crop before moving.
        import torch
        for sample in ([] if rows else range(3)):
            torch.manual_seed(sample)
            prediction=self.model.predict(points[crop],rgb[crop])
            for grasps,scores,contacts in zip(prediction['grasps'],prediction['grasp_confidence'],prediction['grasp_contacts']):
                rows.extend(dict(tcp=grasp_to_tcp(g).tolist(),confidence=float(s),contact=c.tolist())
                            for g,s,c in zip(grasps,scores,contacts))
            if any(np.linalg.norm(np.asarray(r['contact'])-target_point)<.03 for r in rows):break
        rows.sort(key=lambda row:-row['confidence'])
        (root/'grasps_0.json').write_text(json.dumps(dict(candidates=rows,proposal_model=proposal_model)))
        grasp=root/'grasp'
        command=[sys.executable,'scripts/test_m2t2_handle_execution.py','--source',str(root),'--output',str(grasp),
            '--target','0','--surface-filter','--align-panel','--center-handle','--max-candidates','30','--curobo-fallback',
            '--voxel-size','.01','--approach-stand-off','.12']
        if 'cabinet' in target:command+=['--prefer-deep-contact']
        if drawer:command+=['--grasp-ranking','horizontal']
        if target=='microwave':command+=['--center-length']
        else:command+=['--clearance-approach']
        if drawer and opening:
            # 38 cm: with 28 cm only ~14 cm of the bottom is open to the sky past the
            # counter edge and no cup pose fits from above (fix80, isolated). The
            # arm pulls the 28 cm that has withdrawn cleanly; the base backs
            # away for the rest.
            command+=['--open-to-stop','--stop-at-target','--open-target',f'{DRAWER_OPEN_TARGET}','--arm-pull-limit',f'{DRAWER_ARM_PULL}',
                      '--base-assist-pull',f'{self.checked_retreat(info["panel_normal"]):.3f}']
        self.run_script(command,root)
        report=json.loads((grasp/'report.json').read_text())
        if not report.get('physical_grasp_tested'):raise RuntimeError('No reachable handle grasp')
        # Robot encoder evidence only; private contact scores are not consumed.
        fingers=[float(self.runtime.data.joint(n).qpos[0]) for n in
            ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')]
        self.drawer_push_only=False
        if min(fingers)>.79:
            if drawer and not opening and target in self.drawers:
                # Closing a drawer this robot opened does not need the grasp:
                # the closed hand at the remembered handle pushes the front
                # panel just as well. The push must still cover the distance
                # the drawer was opened (checked from encoders afterwards).
                self.drawer_push_only=True
                print('DRAWER_PUSH_WITHOUT_GRASP',json.dumps(dict(target=target,fingers=[round(f,3) for f in fingers])),flush=True)
            else:raise RuntimeError('Handle grasp is empty by gripper encoders')
        # Open = pulled to a measured stop or to the opening target, or far
        # enough by encoders before the arm ran out of reach (fix76 failed a
        # 12 cm opening only because the arm, not the drawer, stopped).
        opened=float(report.get('opening_distance') or 0.)
        if drawer and opening and not (report.get('stop_reason') in ('resisted_motion','opened_to_target') or
                (report.get('stop_reason') in ('arm_path_blocked','arm_limit_reached') and opened>=MIN_DRAWER_OPENING)):
            raise RuntimeError(f'Drawer did not reach an observed motion stop (stopped: {report.get("stop_reason")}, opened {opened:.2f} m)')
        if drawer and opening:print('DRAWER_OPENED',json.dumps(dict(stop=report.get('stop_reason'),opening_m=round(opened,3))),flush=True)
        return grasp,info

    def checked_retreat(self,panel_normal,limit=.30,step=.02):
        """How far the base may back away along the handle's outward normal, by its own map.

        Base footprint only (the arm stays over the drawer it is holding), swept
        in 2 cm steps on the observed navigation map from the estimated pose.
        Returns 0 when there is no map or pose, which disables the base assist.
        """
        try:
            from ..navigation.shaped_docking import RobotGeometry,ObservedShapeChecker
            grid=getattr(self.runtime,'observed_navigation_map',None)
            if grid is None:return 0.
            T=self.estimated_pose();pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
            n=np.asarray(panel_normal,float)[:2];n=T[:2,:2]@(n/np.linalg.norm(n))
            q=self.packet().arm.position
            boxes=[b for b in RobotGeometry(self.runtime.model).boxes(q) if b.name.startswith('robot_0/base/')]
            checker=ObservedShapeChecker(grid,boxes,pose,support_poses=list(getattr(grid,'inherited_support_poses',[]))+[pose])
            clear=0.
            for s in np.arange(step,limit+1e-9,step):
                if not checker.clear(np.r_[pose[:2]+s*n,pose[2]]):break
                clear=float(s)
            print('DRAWER_RETREAT_CHECK',json.dumps(dict(clear_m=round(clear,3),limit_m=limit,detail=None if clear>=limit-1e-9 else str(checker.detail)[:200])),flush=True)
            return clear
        except Exception as exc:
            print('DRAWER_RETREAT_CHECK',json.dumps(dict(clear_m=0.,error=str(exc)[:200])),flush=True)
            return 0.

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
                report=stow_after_fixture(self.runtime,sensors,rec,root/'posture',ScanConfig().stowed_arm,
                    arm_boxes=getattr(self,'calibrated_drawer_withdrawal',False),
                    withdrawal_directions=getattr(self,'withdrawal_directions',()))
                (root/'posture_report.json').write_text(json.dumps(report,indent=2))
            progress('checked_navigation_posture','completed')

    def execute(self,skill,*,target,progress):
        opening=skill.startswith('open');drawer=skill.endswith('drawer')
        if not drawer and not opening and target not in self.hinges:
            return SkillOutcome(False,'closed_panel_reference_unobserved')
        root=self.output/f'{self.index:03d}_{skill}';self.index+=1
        progress('observe_and_grasp','running')
        prepose=None
        try:grasp,info=self.acquire_grasp(target,root,drawer,opening)
        except RuntimeError as exc:
            retried=None
            if target=='fridge' and opening and 'No reachable handle grasp' in str(exc):
                retried=self._grasp_from_closer(target,root,drawer,opening)
            if retried is not None:(grasp,info),root=retried
            else:
                # Closing a drawer this robot opened: no grasp is needed to push it
                # shut. With no reachable or successful handle grasp, bring the
                # closed hand in front of the remembered handle, in the orientation
                # it held the handle with, and push (fix83/fix84: "No reachable
                # handle grasp"; fix80: "Handle grasp is empty").
                memory=self.drawers.get(target) if (drawer and not opening) else None
                if memory is None or 'tool_rotation' not in memory or not any(text in str(exc) for text in
                        ('No reachable handle grasp','Handle grasp is empty','Named handle not visible')):raise
                T=np.linalg.inv(self.estimated_pose())
                normal=T[:3,:3]@np.asarray(memory['normal'],float)
                prepose=np.eye(4);prepose[:3,:3]=T[:3,:3]@np.asarray(memory['tool_rotation'],float)
                prepose[:3,3]=T[:3,:3]@np.asarray(memory['open_handle'],float)+T[:3,3]+DRAWER_PUSH_STANDOFF*normal
                info=dict(panel_normal=normal.tolist());grasp=None;self.drawer_push_only=True
                root.mkdir(parents=True,exist_ok=True)
                print('DRAWER_PUSH_FALLBACK',json.dumps(dict(target=target,reason=str(exc)[:80],hand_base=np.round(prepose[:3,3],3).tolist())),flush=True)
        progress('observe_and_grasp','completed')
        progress('articulate_release_withdraw','running')
        if drawer:
            if opening:self.remember_drawer(target,info,grasp)
            if prepose is None:self.drawer_contact(root,info,opening)
            else:self.drawer_contact(root,info,opening,prepose=prepose)
            if not opening:self.drawers.pop(target,None)
        else:
            output=root/'motion'
            command=[sys.executable,'scripts/test_sensor_hinged_cycle.py','--grasp-run',str(grasp),
                '--output',str(output),'--contact-ik','analytic','--contact-motion-profile','fast',
                '--base-follow-handle','--planar-push','--lateral-base-fraction','1',
                '--contact-retreat','--absolute-open-angle','90','--orientation-gain','.5']
            # 90 deg for cabinets too: at 70 the leaf leans into the side-dock
            # arm space and no bowl placement pose was found (fix69, 192
            # rejections). Closing room comes from choosing the door that
            # swings toward open floor (cabinet survey), not a narrower opening.
            if target=='microwave':command+=['--base-yaw-follow']
            if opening:command+=['--open-only','--auto-probe-direction']
            else:
                T=np.linalg.inv(self.estimated_pose());memory=self.hinges[target]
                local=dict(closed_anchor=(T[:3,:3]@memory['closed_anchor']+T[:3,3]).tolist(),
                    closed_normal=(T[:3,:3]@memory['closed_normal']).tolist(),
                    hinge_xy=(T[:3,:3]@memory['hinge']+T[:3,3])[:2].tolist())
                path=root/'observed_close_reference.json';path.write_text(json.dumps(local))
                command+=['--close-observation',str(path)]
            bound=False
            if opening and target in FLOOR_MEMORY_OPENINGS:
                # The pull makes the base back away with the handle. With only
                # the floor the cameras see during the pull, the fourth step
                # back had two unseen cells, the base stayed, the arm pulled
                # alone and the fridge handle slid out of the hand (fix107).
                # The episode's map already holds that floor: offer it, as the
                # push-close does.
                from ..code_policy.panel_closing import bind_observed_floor
                bound=bind_observed_floor(self)
            try:self.run_script(command,root)
            finally:
                if bound:self.runtime.contact_support_region=None
            result=json.loads((output/'report.json').read_text())
            pushed_open=None
            if result.get('error'):
                measured=result.get('observed_open_angle_degrees') if result.get('observed_hinge_base') is not None else None
                if not (opening and measured is not None and abs(float(measured))>=PARTIAL_OPENING_MIN_DEGREES):
                    raise RuntimeError(result['error'])
                if abs(float(measured))>=90.-15.:
                    # Stopped inside the accepted opening band (fridge at 85 deg
                    # when the last push step was refused, fix116): the door is
                    # open; keep what was measured and go on.
                    print('DOOR_OPENING_ACCEPTED_AFTER_ERROR',json.dumps(dict(target=target,observed_open_degrees=round(abs(float(measured)),1),
                        reason=str(result['error'])[:120])),flush=True)
                    result['error']=None
                # The handle pull stopped part-way (microwave at 47 deg when
                # the base could no longer follow, fix115) and has let go.
                # What it measured of the door is kept, and the closed hand
                # pushes the leaf the rest of the way from inside.
            if result.get('error'):
                print('DOOR_OPENING_PARTIAL',json.dumps(dict(target=target,observed_open_degrees=round(abs(float(measured)),1),
                    reason=str(result['error'])[:120])),flush=True)
                self._remember_door(target,result)
                pushed_open=self._push_rest_open(target,root,abs(float(measured)))
            # The script's `success` is a private simulator-truth score (opening:
            # within 3 deg of the joint limit; push-close: always False) and must
            # not drive execution (fix68: a correct 71.5 deg opening was failed).
            # Use the robot's own observed opening angle instead.
            if opening:
                observed=pushed_open if pushed_open is not None else result.get('observed_open_angle_degrees')
                requested=90.
                if observed is not None and abs(float(observed))<requested-15.:
                    raise RuntimeError(f'Door observed open only {abs(float(observed)):.0f} deg (requested {requested:.0f})')
            if opening and pushed_open is None:self._remember_door(target,result)
        progress('articulate_release_withdraw','completed')
        self.navigation_posture(root,progress)
        if opening and (getattr(self,'restood_for_push_open',False) or target in FRONT_STAND_AFTER_OPENING) and 'open_handle' in self.hinges.get(target,{}):
            # The opening was finished from a stand spot beside the door's free
            # edge. Left there, the robot looks at the back of the open leaf:
            # the food inside the microwave was then 'not visible' for the pick
            # (fix143, call 18). Return to the front of the opening.
            # The fridge pull always ends beside the door (the base follows the
            # handle round): from there the line of sight to the drink runs
            # along the fridge front and every stand spot on it was blocked
            # (fix143, call 26). So the fridge gets the front stand every time.
            self.restood_for_push_open=False
            self._stand_before_opening(target,root/'front_stand')
        return SkillOutcome(True,'commands_completed_fixture_goal_not_scored')

    def _remember_door(self,target,result):
        """Store what an opening measured of the door (hinge, closed face, opened handle) in the map frame."""
        T=self.estimated_pose()
        self.hinges[target]=dict(
            hinge=T[:3,:3]@np.asarray(result['observed_hinge_base'])+T[:3,3],
            closed_handle=T[:3,:3]@np.asarray(result['observed_closed_handle_base'])+T[:3,3],
            closed_anchor=T[:3,:3]@np.asarray(result['observed_closed_panel_anchor_base'])+T[:3,3],
            closed_normal=T[:3,:3]@np.asarray(result['observed_closed_panel_normal_base']))
        if result.get('observed_open_handle_base') is not None:
            self.hinges[target]['open_handle']=T[:3,:3]@np.asarray(result['observed_open_handle_base'])+T[:3,3]

    def observed_door_angle(self,target,expected_degrees):
        """Opening angle (deg from closed) of a remembered door in the current depth views, or None."""
        from ..code_policy.panel_closing import leaf_angle_from_points
        memory=self.hinges[target];T=np.linalg.inv(self.estimated_pose())
        local=lambda key:T[:3,:3]@np.asarray(memory[key],float)+T[:3,3]
        points=np.concatenate([camera_points(cam)[0] for cam in self.packet().cameras.values()])
        return leaf_angle_from_points(points,local('hinge'),local('closed_handle'),local('open_handle'),expected_degrees)

    def _push_rest_open(self,target,root,opened_degrees,requested=90.):
        """Push a part-opened door on from inside; returns the opening (deg) the robot then sees it at."""
        from ..code_policy.panel_closing import set_open_memory
        total=float(opened_degrees);restood=False
        for attempt in range(PUSH_OPEN_ATTEMPTS):
            try:turned,stop=self.door_hook_push(target,root/f'push_open_{attempt}',opening=True)
            except RuntimeError as exc:
                if 'unreachable' not in str(exc) or restood:raise
                # From in front of the leaf on its hinge side the hand cannot
                # get round the free edge (fix126: fridge at 25 deg, base 0.36 m
                # to the hinge side of the handle, every hand pose without an
                # arm solution). Stand past the free edge and try once more.
                restood=True;self.restood_for_push_open=True;tried=[]
                while True:
                    # One stand spot after another: from the first one that
                    # had a route the hand pose can still be out of reach
                    # (replay of the fix143 state, spot 30 deg outside).
                    turn=self._stand_past_free_edge(target,root/f'push_open_{attempt}_dock_{len(tried)}',skip=tuple(tried))
                    if turn is None:raise
                    tried.append(turn)
                    try:
                        turned,stop=self.door_hook_push(target,root/f'push_open_{attempt}_restood_{len(tried)}',opening=True);break
                    except RuntimeError as again:
                        if 'unreachable' not in str(again):raise
                        print('PUSH_OPEN_STAND_OUT_OF_REACH',json.dumps(dict(target=target,turn_degrees=turn)),flush=True)
            # The hand's travel says little about the leaf (it can slide along
            # it: 0.44 m of travel turned the microwave door 10 deg in the
            # first isolated try). Look at where the leaf is now.
            seen=self.observed_door_angle(target,total)
            before=total
            if seen is not None:total=float(min(max(seen,total),requested))
            if opened_to_its_stop(seen,requested,stop):
                # The sweep ended against resistance with the leaf seen close
                # to the requested opening: it is at its stop. The look at the
                # leaf reads low there (fix132: 80.8 deg seen, door privately at
                # 90), and with the leaf remembered 9 deg into the opening
                # every grasp inside the microwave was refused as unreachable.
                print('DOOR_AT_ITS_STOP',json.dumps(dict(target=target,seen_open_degrees=round(float(seen),1),
                    remembered_open_degrees=requested)),flush=True)
                total=float(requested)
            set_open_memory(self.hinges[target],total)
            print('DOOR_PUSHED_FURTHER_OPEN',json.dumps(dict(target=target,attempt=attempt+1,hand_estimate_degrees=round(turned,1),
                seen_open_degrees=None if seen is None else round(seen,1),open_degrees=round(total,1),stop=stop[:40])),flush=True)
            if total>=requested-PUSH_OPEN_DONE_WITHIN_DEGREES or seen is None:break
            if total-before<PUSH_OPEN_MIN_PROGRESS_DEGREES:
                if restood or 'motion_path_blocked' not in stop or attempt+1>=PUSH_OPEN_ATTEMPTS:break
                # The arm ran out of a clear path from this stand spot with the
                # leaf still short of open (replays of the fix150 state: fridge
                # held at 55 and 70 deg, 'no collision-clear IK solution'):
                # stand past the free edge, as for an unreachable first push,
                # and push from there.
                print('PUSH_OPEN_BLOCKED_RESTAND',json.dumps(dict(target=target,open_degrees=round(total,1),stop=stop[:60])),flush=True)
                restood=True;self.restood_for_push_open=True
                if self._stand_past_free_edge(target,root/f'push_open_{attempt}_blocked_dock') is None:break
        return total

    def _grasp_from_closer(self,target,root,drawer,opening):
        """Re-dock in front of the handle the grasp step found when it was out of reach, then grasp again.

        fix162 call 24: the re-dock saw only the freezer pull and stood 1.0 m
        from the fridge; the grasp step then found the door bar 1.37 m high but
        no grasp pose had an arm solution from there (96 tried). Its replays
        opened the door from 0.66 m in front of the bar. Returns ((grasp, info),
        new root) or None.
        """
        import json,numpy as np
        from ..code_policy.redocking import SensorRedocking
        from .episode import episode_sensors
        try:bar=np.asarray(np.load(root/'input_0.npz')['target'],float)
        except (OSError,KeyError,ValueError):return None
        reach=float(np.linalg.norm(bar[:2]))
        print('HANDLE_OUT_OF_REACH',json.dumps(dict(target=target,handle_base=np.round(bar,3).tolist(),reach_m=round(reach,3))),flush=True)
        if reach<=CLOSER_GRASP_MIN_REACH:return None
        T=self.estimated_pose();spot=T[:3,:3]@bar+T[:3,3]
        toward=-bar.copy();toward[2]=0.;toward/=np.linalg.norm(toward);normal=T[:3,:3]@toward
        try:
            with episode_sensors(self.runtime.model,self.runtime.data) as sensors:
                dock=SensorRedocking(self.runtime,sensors,self.runtime.mapper.view())
                dock.front_view_hint=(spot,normal)
                result=dock.front_view_dock(CLOSER_GRASP_DISTANCE,root.parent/(root.name+'_closer_dock'))
        except RuntimeError as exc:result=dict(success=False,reason=str(exc)[:120])
        print('HANDLE_CLOSER_DOCK',json.dumps(dict(target=target,success=bool(result and result.get('success')),
            reason=str((result or {}).get('reason'))[:80],final_distance_m=(result or {}).get('final_distance_m'))),flush=True)
        if not (result and result.get('success')):return None
        again=root.parent/(root.name+'_closer')
        try:return self.acquire_grasp(target,again,drawer,opening),again
        except RuntimeError as exc:
            print('HANDLE_CLOSER_GRASP_FAILED',json.dumps(dict(target=target,reason=str(exc)[:120])),flush=True)
            return None

    def _stand_past_free_edge(self,target,root,skip=()):
        """Re-dock beside the free edge of the remembered part-opened leaf (see PastEdgeRedocking).

        Returns the stand direction (deg) that was reached, or None; directions in `skip` are not tried.
        """
        from ..code_policy.closing_redock import PastEdgeRedocking,PAST_EDGE_DISTANCE,PUSH_OPEN_STAND_TURNS
        from .episode import episode_sensors
        guided=getattr(self.runtime,'qwen_docking',False);self.runtime.qwen_docking=False     # stand exactly there
        result=dict(success=False,reason='no stand spot tried');chosen=None
        def attempt(folder):
            nonlocal result,chosen
            for turn in [t for t in PUSH_OPEN_STAND_TURNS if t not in skip]:
                with episode_sensors(self.runtime.model,self.runtime.data) as sensors:
                    dock=PastEdgeRedocking(self.runtime,sensors,self.runtime.mapper.view(),self.hinges[target],turn_degrees=turn)
                    try:result=dock.move(target,'handle',PAST_EDGE_DISTANCE,folder/f'turn_{int(turn)}')
                    except RuntimeError as exc:result=dict(success=False,reason=str(exc)[:160])
                if result.get('success'):chosen=turn;return True
            return False
        try:
            # Fold the arm before driving round the open leaf. The drive was
            # passed with the arm still out at handle height after a refused
            # push approach, and the hand and mast swung the microwave door
            # from 74 to 36 deg on the way (fix156 call 16, private replay).
            # A no-op when the arm is already stowed.
            self.navigation_posture(root,lambda *a:None)
            attempt(root)
        finally:self.runtime.qwen_docking=guided
        print('PUSH_OPEN_REDOCK',json.dumps(dict(target=target,success=bool(result.get('success')),turn_degrees=chosen,
            reason=str(result.get('reason'))[:80],final_distance_m=result.get('final_distance_m'))),flush=True)
        return chosen

    def _stand_before_opening(self,target,root):
        """Stand OPENED_FRONT_DISTANCE square in front of the remembered opening (best effort)."""
        from ..code_policy.redocking import SensorRedocking
        from .episode import episode_sensors
        memory=self.hinges[target]
        centre=(np.asarray(memory['hinge'],float)+np.asarray(memory['closed_handle'],float))/2
        normal=np.asarray(memory['closed_normal'],float).copy();normal[2]=0.;normal/=np.linalg.norm(normal)
        try:
            with episode_sensors(self.runtime.model,self.runtime.data) as sensors:
                dock=SensorRedocking(self.runtime,sensors,self.runtime.mapper.view())
                dock.front_view_hint=(centre,normal)
                result=dock.front_view_dock(OPENED_FRONT_DISTANCE,root)
        except RuntimeError as exc:result=dict(success=False,reason=str(exc)[:160])
        print('OPENED_DOOR_FRONT_STAND',json.dumps(dict(target=target,success=bool(result and result.get('success')),
            reason=str((result or {}).get('reason'))[:80])),flush=True)
        return bool(result and result.get('success'))

    def leaf_width(self,target):
        """Observed door leaf width (hinge to closed handle, metres) of the last opening, or None."""
        memory=self.hinges.get(target)
        if memory is None or 'closed_handle' not in memory:return None
        return float(np.linalg.norm(np.asarray(memory['closed_handle'])[:2]-np.asarray(memory['hinge'])[:2]))

    def base_assisted_push(self,motion,distance,record,speed=.05):
        """Drive the base forward along the push direction while the arm holds the handle.

        The swept floor is where the drawer the hand is pushing stood open, and
        the hand stays the arm's length ahead of the base. Stops when the
        distance is covered, when the base stops following its setpoint (the
        drawer reached its closed stop) or when the fingers lose the handle.
        Base and finger encoders only.
        """
        from ..control.onboard_base import OnboardBaseActuator
        m,d=self.runtime.model,self.runtime.data
        base=OnboardBaseActuator(m,d,motion_guard=lambda f,w,t:True)
        qids=base.qids;start=np.asarray(d.qpos[qids]).copy();heading=start[2]
        push=-motion.normal[:2]/np.linalg.norm(motion.normal[:2])       # base frame
        world=np.array([np.cos(heading)*push[0]-np.sin(heading)*push[1],np.sin(heading)*push[0]+np.cos(heading)*push[1]])
        commanded=0.;stop='distance_covered';window=[]
        hand=motion.tcp()[:3,3].copy()      # tool position in the base frame, from joint encoders
        while commanded<distance:
            base.step(speed*float(push[0]),0.,lateral=speed*float(push[1]),planar_motion_guard=lambda f,l,w,t:True)
            record();commanded+=speed*.02
            moved=float((np.asarray(d.qpos[qids])[:2]-start[:2])@world);window.append(moved)
            if min(motion.fingers())>.79 and not getattr(self,'drawer_push_only',False):stop='grasp_lost';break
            # The closed drawer pushes the hand back toward the base while the
            # base still advances: stop at 1.5 cm of arm compression instead of
            # leaning on the cabinet (the base alone kept going ~10 cm).
            if float((hand-motion.tcp()[:3,3])[:2]@push)>.015:stop='resisted_motion';break
            if len(window)>50 and window[-1]-window[-50]<.2*speed:stop='resisted_motion';break
        for _ in range(25):base.step(0.,0.);record()
        moved=float((np.asarray(d.qpos[qids])[:2]-start[:2])@world)
        print('BASE_ASSISTED_PUSH',json.dumps(dict(requested_m=round(distance,3),base_progress_m=round(moved,3),stop=stop)),flush=True)
        return moved,stop

    def door_hook_push(self,target,root,opening=False,travel_budget=None,withdraw=.06):
        """Push a remembered door leaf with the closed hand: part-way shut from its far side, or further open from inside.

        Closing (opening=False). For a door opened next to a wall (this fridge:
        0.45 m between the open leaf and the wall) there is no room to stand on
        the closing side, and from the room the far face is out of the cameras'
        view. The datagen oracle then reaches round the free edge, puts the arm
        on that far face and pushes to make room. Here the closed hand goes to
        the remembered opened-handle position, a stand-off outside it (as the
        drawer push goes to the remembered drawer handle), and sweeps toward the
        closed side.

        Opening (opening=True). After a handle pull that stopped part-way, the
        closed hand goes to the inner side of the leaf at the remembered handle
        and sweeps in the opening direction until the leaf's stop resists.

        Returns (leaf rotation in degrees estimated from the hand's measured
        travel and the remembered leaf radius, stop reason).
        """
        from .appliance_controls import build_appliance_backend
        from ..control.sensor_drawer import SensorDrawerMotion
        from ..control.observed_grasp_planner import ObservedGraspPlanner,merge_voxels
        from curobo._src.geom.types import Cuboid
        from ..recording.deferred import StateVideoRecorder
        from ..control.contact_retreat import tool_pose
        from ..code_policy.closing_redock import closing_face
        memory=self.hinges[target];handle,normal=closing_face(memory)
        T=np.linalg.inv(self.estimated_pose())
        handle=T[:3,:3]@np.asarray(handle,float)+T[:3,3];normal=T[:3,:3]@np.asarray(normal,float);normal[2]=0.;normal/=np.linalg.norm(normal)
        hinge=T[:3,:3]@np.asarray(memory['hinge'],float)+T[:3,3]
        along=handle-hinge;along[2]=0.;radius=float(np.linalg.norm(along));along/=radius
        # `side` is the leaf face the hand starts beside: the far (outer) face
        # to push shut, the inner face to push open. The sweep goes along -side.
        side=-normal if opening else normal
        standoff,contact_after,thickness=((DOOR_PUSH_OPEN_STANDOFF,DOOR_PUSH_OPEN_CONTACT_AFTER,DOOR_PUSH_OPEN_LEAF_DEPTH) if opening
                                          else (DOOR_HOOK_STANDOFF,DOOR_HOOK_CONTACT_AFTER,DOOR_HOOK_LEAF_HALF_THICKNESS))
        direction=-side;up=np.array([0.,0.,1.])
        def facing(axis,roll=0.):
            axis=axis/np.linalg.norm(axis);frame=np.column_stack((up,np.cross(axis,up),axis))
            c,s_=np.cos(roll),np.sin(roll);return frame@np.array([[c,-s_,0.],[s_,c,0.],[0.,0.,1.]])
        # Hand pointing along the leaf toward the hinge (its side meets the
        # leaf), then turned half-way toward the leaf, then straight at it.
        rotations=[('along_leaf',facing(-along)),('along_leaf_flipped',facing(-along,np.pi)),
                   ('diagonal',facing(-along+direction)),('facing_leaf',facing(direction))]
        root.mkdir(parents=True,exist_ok=True);phase='door_hook_approach'
        with StateVideoRecorder(self.runtime.model,self.runtime.data,root/'hook_recording') as recorder:
            b=build_appliance_backend(self.runtime,output=root/'backend',command_observer=lambda:recorder.record(phase))
            def planner(current,goal,contact_patch):
                pts=np.concatenate([camera_points(cam)[0] for cam in self.packet().cameras.values()])
                pts=pts[(np.linalg.norm(pts,axis=1)<1.8)&(pts[:,2]>.08)]
                if contact_patch:
                    delta=goal[:3,3]-current[:3,3]
                    t=np.clip((pts-current[:3,3])@delta/max(delta@delta,1e-12),0,1)
                    pts=pts[np.linalg.norm(pts-current[:3,3]-t[:,None]*delta,axis=1)>.075]
                    # The leaf being pushed is not an obstacle to the push: drop
                    # what is observed along the remembered leaf (hinge to a
                    # little past the handle) and in the band it sweeps through.
                    # The sweep stopped 2 cm short of the handle when the
                    # leaf's own edge was kept (isolated, fix113 state).
                    rel=pts[:,:2]-hinge[:2];s_=np.clip(rel@along[:2],0.,radius+DOOR_HOOK_LEAF_PAST_HANDLE)
                    off=rel-s_[:,None]*along[:2];across=off@side[:2]
                    pts=pts[~((across<thickness)&(across>-DOOR_HOOK_TRAVEL)&(np.abs(pts[:,2]-handle[2])<DOOR_HOOK_LEAF_BAND)&(rel@along[:2]>.15))]
                vox=np.unique(np.floor(pts/.015).astype(int),axis=0)
                boxes=[Cuboid(name=f'door_{i}',pose=((lo+hi+1)*.0075).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.015).tolist()) for i,(lo,hi) in enumerate(merge_voxels(vox))]
                from ..control.fixture_approach import calibrated_finger_planner
                # The free approach goes round the leaf's edge from the folded
                # arm: a straight joint blend does not clear it, so that move
                # may use the full motion planner; contact moves do not.
                checked=calibrated_finger_planner(b.prepare_arm(),b.geometry,boxes,
                    self.runtime.model,self.runtime.data,arm_boxes=True,allow_curobo_fallback=not contact_patch)
                # User rule (2026-11-18): IK found is enough for the door push;
                # this planner instance only, not the rest of the pipeline.
                # Not when pushing open: the reach round the leaf's free edge
                # must not swing the open door back (fix162 replays).
                checked.ik_only=not opening
                return checked
            motion=SensorDrawerMotion(joint_position=b.joints,
                tcp_pose=lambda q=None:tool_pose(b.geometry,b.joints() if q is None else q),
                fingers=b.fingers,command=b.command,planner_factory=planner,outward_normal=side)
            if not motion.gripper(255.):raise RuntimeError('Gripper did not close for the door push')
            reached=None;errors=[]
            for name,rotation in rotations:
                for outward in (DOOR_PUSH_OPEN_PAST_HANDLE if opening else DOOR_HOOK_PAST_HANDLE):
                    prepose=np.eye(4);prepose[:3,:3]=rotation
                    prepose[:3,3]=handle+standoff*side+outward*along
                    try:
                        if motion.path(prepose,grip=255.,contact_patch=False):reached=(name,outward);break
                        errors.append(name+': not settled')
                    except RuntimeError as exc:errors.append(name+': '+str(exc)[:60])
                if reached is not None:break
            if reached is None:raise RuntimeError('Door push pose unreachable: '+'; '.join(errors)[:600])
            phase='door_hook_push';result=motion.move_until_resistance(opening=False,travel_budget=travel_budget if travel_budget is not None else (DOOR_PUSH_OPEN_TRAVEL if opening else DOOR_HOOK_TRAVEL))
            travel=float(result.measured_travel)
            phase='door_hook_withdraw'
            # Back off from the leaf before the arm is folded. A longer
            # `withdraw` falls back to shorter ones when it cannot be reached.
            for distance in dict.fromkeys((withdraw,min(withdraw,.12),.06)):
                try:
                    back=motion.tcp().copy();back[:3,3]+=distance*side
                    if motion.path(back,grip=255.,contact_patch=True):break
                except RuntimeError:continue
        turned=float(np.degrees(np.arcsin(np.clip((travel-contact_after)/radius,0.,1.))))
        print('DOOR_HOOK_PUSH',json.dumps(dict(target=target,opening=bool(opening),start=reached,hand_travel_m=round(travel,3),stop=result.reason[:60],
            leaf_radius_m=round(radius,3),estimated_turn_degrees=round(turned,1))),flush=True)
        return turned,result.reason

    def remember_drawer(self,target,info,grasp):
        """Where this robot's hand holds the opened drawer's handle, in its map frame.

        Called while the handle is still held, after the pull: tool position
        from joint encoders, pull direction from the observed front panel and
        the pulled distance from encoders. No fixture state is read.
        """
        try:
            from ..navigation.shaped_docking import RobotGeometry
            geometry=RobotGeometry(self.runtime.model);q=self.packet().arm.position
            tool=geometry.fk(q)[geometry.tcp_body]@geometry.tcp_local;tcp=tool[:3,3]
            normal=np.asarray(info['panel_normal'],float).copy();normal[2]=0.;normal/=np.linalg.norm(normal)
            opening=float(json.loads((grasp/'report.json').read_text()).get('opening_distance') or 0.)
            T=self.estimated_pose()
            self.drawers[target]=dict(open_handle=T[:3,:3]@tcp+T[:3,3],normal=T[:3,:3]@normal,opening=opening,
                                      tool_rotation=T[:3,:3]@tool[:3,:3])
            print('DRAWER_REMEMBERED',json.dumps(dict(target=target,open_handle_map=np.round(self.drawers[target]['open_handle'],3).tolist(),
                normal_map=np.round(self.drawers[target]['normal'],3).tolist(),opening_m=round(opening,3))),flush=True)
        except Exception as exc:
            print('DRAWER_NOT_REMEMBERED',str(exc)[:200],flush=True)

    def drawer_observation(self,target,*,map_frame=False):
        """Remembered opened drawer reanchored by current SLAM (base frame unless map_frame)."""
        memory=self.drawers.get(target)
        if memory is None or memory['opening']<MIN_DRAWER_OPENING:return None
        if map_frame:return dict(open_handle=np.asarray(memory['open_handle']).tolist(),normal=np.asarray(memory['normal']).tolist(),opening=memory['opening'])
        T=np.linalg.inv(self.estimated_pose())
        return dict(open_handle=(T[:3,:3]@np.asarray(memory['open_handle'])+T[:3,3]).tolist(),
                    normal=(T[:3,:3]@np.asarray(memory['normal'])).tolist(),opening=memory['opening'])

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

    def drawer_contact(self,root,info,opening,prepose=None):
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
                if getattr(self,'calibrated_drawer_withdrawal',False):
                    from ..control.fixture_approach import calibrated_finger_planner
                    return calibrated_finger_planner(b.prepare_arm(),b.geometry,boxes,
                        self.runtime.model,self.runtime.data,arm_boxes=True,allow_curobo_fallback=False)
                return ObservedGraspPlanner(b.prepare_arm(),b.geometry,boxes)
            motion=SensorDrawerMotion(joint_position=b.joints,
                tcp_pose=lambda q=None:tool_pose(b.geometry,b.joints() if q is None else q),
                fingers=b.fingers,command=b.command,planner_factory=planner,outward_normal=info['panel_normal'])
            if not opening and prepose is not None:
                phase='drawer_push_approach'
                if not motion.gripper(255.):raise RuntimeError('Gripper did not close for the drawer push')
                try:reached=motion.path(prepose,grip=255.,contact_patch=True)
                except RuntimeError as exc:raise RuntimeError('Drawer push pose unreachable: '+str(exc)[:160]) from exc
                if not reached:raise RuntimeError('Drawer push pose not settled')
            if not opening:
                phase='drawer_close';result=motion.move_until_resistance(opening=False)
                remembered=self.drawers.get(getattr(self,'current_target',None)) or next(iter(self.drawers.values()),None)
                short=remembered is not None and result.measured_travel<remembered['opening']-.06
                # The arm alone has closed it only if it felt the stop after
                # covering the opening. Otherwise the base finishes the push:
                # in fix86 the arm pushed 32 of 37.4 cm, ran out of reach
                # ("motion_path_blocked") and the call failed 5 cm from closed.
                arm_closed=result.completed and not short
                if remembered is not None and not arm_closed and (result.reason.startswith('motion_path_blocked') or result.reason=='travel_budget_exhausted' or result.completed):
                    # The arm ran out of reach (or met early resistance) before
                    # covering the distance this robot pulled the drawer open.
                    # Keep the handle and drive the base forward for the rest.
                    # The open drawer drifts when the robot brushes it (37.5 ->
                    # 43.4 cm in fix84), so push until the base stops following
                    # (the closed stop), up to 12 cm past the remembered opening.
                    extra,stop=self.base_assisted_push(motion,max(0.,remembered['opening']-result.measured_travel)+DRAWER_CLOSE_OVERTRAVEL,lambda:recorder.record('drawer_close_base'))
                    from ..control.sensor_drawer import DrawerMotionResult
                    result=DrawerMotionResult(True,'base_assisted_'+stop,result.measured_travel+extra)
                if not result.completed:raise RuntimeError(result.reason)
                # Robot-side check: the push must cover the distance this robot
                # pulled the drawer open. fix79 reported a close "completed"
                # with the drawer still 28 cm open (the push met resistance at once).
                print('DRAWER_CLOSE_PUSH',json.dumps(dict(stop=result.reason,travel_m=round(float(result.measured_travel),3),
                    remembered_opening_m=None if remembered is None else round(float(remembered['opening']),3))),flush=True)
                if remembered is not None and result.measured_travel<remembered['opening']-.06:
                    raise RuntimeError(f"Drawer push stopped after {result.measured_travel:.2f} m of the {remembered['opening']:.2f} m it was opened")
            alternatives=([motion.normal*.03+[0,0,.08],[0,0,.12]]
                          if getattr(self,'calibrated_drawer_withdrawal',False) else ())
            phase='drawer_release';result=motion.release_and_withdraw(alternatives=alternatives)
            if not result.completed:raise RuntimeError(result.reason)
