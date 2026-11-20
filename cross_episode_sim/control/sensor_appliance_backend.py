"""Sensor-only M2T2 knob grasps and geometric fingertip button presses.

All geometry is in the stationary robot-base frame. This backend deliberately
does not navigate: callers dock using the observed map before invoking it.
"""
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .appliance_controls import SensorApplianceControls
from .contact_retreat import tool_pose
from .m2t2_policy import grasp_to_tcp, filter_observed_contacts, parallel_gripper_symmetries
from .observed_grasp_planner import ObservedGraspPlanner, merge_voxels
from ..perception.appliance_controls import fit_control,refit_tracked_knob
from ..perception.handle_surface import align_grasp_to_panel, handle_grasp_rank


def camera_points(camera):
    valid=camera.valid_depth.copy()
    if camera.robot_mask is not None:valid &= ~camera.robot_mask
    v,u=np.where(valid);z=camera.depth_m[v,u]
    points=np.c_[(u-camera.K[0,2])*z/camera.K[0,0],
                 (v-camera.K[1,2])*z/camera.K[1,1],z]
    points=points@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
    return points,camera.rgb[v,u],valid


def pixel_point(camera, valid, pixel):
    u,v=np.round(pixel).astype(int)
    if not (0<=v<valid.shape[0] and 0<=u<valid.shape[1] and valid[v,u]):
        raise ValueError('Pointed control has no valid non-robot depth')
    z=camera.depth_m[v,u]
    return camera.T_base_optical[:3,:3]@np.array([
        (u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z])+camera.T_base_optical[:3,3]



# A second, positional description for keys that sit beside a look-alike neighbour. Benchmark on
# 17 saved Start presses (wrist view, private scoring): the plain description chose Start 13 times,
# the Stop key twice (fix138, fix149) and declined twice; taking the positional answer when the two
# part by more than 2 cm chooses Start 14 times, Stop once (fix138: both answers were Stop) and
# declines twice. The positional description alone pressed 20 and 58 cm off where the plain one
# declined, so it is only a second look after the plain one located a key.
SECOND_KEY_DESCRIPTIONS={'Start key at the bottom of the microwave keypad':
    'Start key: the left one of the two wide keys at the bottom of the microwave keypad'}
SECOND_KEY_AGREE_METRES=.02

# Tilts (deg, about the indicator bar) of observed-geometry knob grasps tried when M2T2 has none.
KNOB_GEOMETRIC_TILTS=(0.,-15.,15.,-30.,30.)
# Straight withdrawal from a released knob: total distance (m) along its axis, in this many steps.
KNOB_WITHDRAW_DISTANCE=.07
KNOB_WITHDRAW_STEPS=7
# A remembered stove knob within this distance (m) of the fitted one is the same knob.
KNOB_MEMORY_MATCH=.06

class SensorApplianceBackend:
    def __init__(self, *, capture, joints, fingers, command, geometry, arm,
                 pad_vertices, pointer, m2t2, precise_geometry=None, prepare_arm=None,
                 motion_epoch=lambda:0, progress=lambda event,detail: None):
        self.capture=capture;self.joints=joints;self.fingers=fingers
        self.command=command;self.geometry=geometry;self.arm=arm
        self.pad_vertices=pad_vertices;self.pointer=pointer;self.m2t2=m2t2
        self.precise_geometry=precise_geometry
        self.prepare_arm=prepare_arm
        self.motion_epoch=motion_epoch;self.knob_tracks={};self.knob_grasps={}
        self.progress=progress;self.events=[];self.grip=0.

    def handlers(self):return SensorApplianceControls(self).handlers()

    def event(self,event,**detail):
        self.events.append(dict(event=event,**detail));self.progress(event,detail)

    def tcp(self):return tool_pose(self.geometry,self.joints())

    def acquire(self,target,kind):
        packet=self.capture();clouds=[];colors=[];views={}
        for name,camera in packet.cameras.items():
            points,rgb,valid=camera_points(camera)
            clouds.append(points);colors.append(rgb);views[name]=(camera,valid)
        self.points=np.concatenate(clouds);self.rgb=np.concatenate(colors)
        keep=(np.linalg.norm(self.points,axis=1)<2.)&(self.points[:,2]>.08)
        self.points=self.points[keep];self.rgb=self.rgb[keep]
        track_key=(target,self.motion_epoch())
        self.active_track_key=track_key
        if kind=='knob' and track_key in self.knob_tracks:
            # Re-identify the same knob after the wrist rotates. Image-centre
            # language alone may select its neighbor in the new camera view.
            # Coordinates here came from the previous RGB-D observation, and
            # are invalidated by every external motion skill / base change.
            previous=self.knob_tracks[track_key]
            observed=refit_tracked_knob(self.points,previous)
            self._accept_control(observed,previous.center,target,kind)
            self.event('reobserved_knob_track',center=observed.center.tolist(),angle=observed.angle)
            return observed
        errors=[]
        # A key is located in the wrist view only: the mast camera never put a
        # zoom-in on the right key in 14 saved presses (button_pointing.py).
        for name in (('wrist',) if kind=='button' else ('wrist','navigation')):
            camera,valid=views[name];im=Image.fromarray(camera.rgb)
            result=self.pointer.point(im,target)
            self.event('point_control',camera=name,result=result)
            for pixel in result['points']:
                # Resolve small control text at a larger image scale. This is
                # an image-derived crop, never a scene object bounding box.
                if kind=='button':
                    from ..perception.button_pointing import locate_key,key_at_view_edge
                    located,_,record=locate_key(self.pointer,im,target,lambda p:pixel_point(camera,valid,p),pixel)
                    if located is not None and key_at_view_edge(located,im.size):
                        record=dict(record,reason='key at the edge of the view');located=None
                    second=SECOND_KEY_DESCRIPTIONS.get(target)
                    if located is not None and second is not None:
                        # The Start key sits beside the Stop key at the bottom of
                        # the keypad; the plain description sometimes lands on
                        # Stop (fix149, the press then switched nothing on).
                        # Locate it again by its position in that row; when the
                        # two answers part, the positional one is taken.
                        hint=(self.pointer.point(im,second).get('points') or [None])[0]
                        if hint is not None:
                            again,_,_=locate_key(self.pointer,im,second,lambda p:pixel_point(camera,valid,p),tuple(hint))
                            if again is not None and not key_at_view_edge(again,im.size):
                                apart=float(np.linalg.norm(pixel_point(camera,valid,again)-pixel_point(camera,valid,located)))
                                record=dict(record,second_description_apart_m=round(apart,3))
                                if apart>SECOND_KEY_AGREE_METRES:located=again;record=dict(record,chosen_by='key position in the row')
                    self.event('refine_button',camera=name,result=record)
                    print('BUTTON_LOCATED',json.dumps(dict(target=target[:40],camera=name,located=located is not None,
                        **{k:(v if not isinstance(v,np.generic) else v.item()) for k,v in record.items()})),flush=True)
                    if located is None:
                        # No blind press of the unrefined spot (17-20 cm off in
                        # three saved presses).
                        errors.append('key not located: '+str(record.get('reason')));continue
                    pixel=located
                try:
                    point=pixel_point(camera,valid,pixel)
                    observed=fit_control(self.points,point,kind)
                    # Not the remembered centre: fix183 call 8 showed its height 1.8 cm off
                    # (the fit was right there), so the knob fit itself is trimmed instead.
                    self.event('control_geometry',kind=kind,center=observed.center.tolist(),
                               normal=observed.outward.tolist(),angle=observed.angle)
                    self._accept_control(observed,point,target,kind)
                    return observed
                except ValueError as exc:errors.append(str(exc))
        raise RuntimeError('control_not_observed: '+str(errors))

    def _remembered_knob_center(self,packet,observed,target):
        """Centre a stove knob on this robot's own empty-hand measurement.

        fix182 call 14: the fit from the knob stand put the centre 2.4 cm above
        the knob axis (extra panel points in the surface), so the pinch closed
        on air. The knobs measured square-on at the start of the task matched
        the axis height. Only the centre is replaced; normal, surface and the
        indicator angle stay from the fresh fit.
        """
        from ..runtime.episode import current_runtime
        runtime=current_runtime();remembered=getattr(runtime,'stove_knobs_map',None) if runtime is not None else None
        if not remembered or 'stove' not in target or getattr(runtime,'mapper',None) is None:return observed
        try:
            estimate=runtime.mapper.view().update(packet.cameras['navigation'],packet.timestamp)
            if not estimate.accepted:return observed
            T=np.asarray(estimate.T_map_base);knobs=(np.asarray(remembered,float)-T[:3,3])@T[:3,:3]
        except Exception as exc:
            print('KNOB_MEMORY_UNUSED',json.dumps(dict(reason=str(exc)[:120])),flush=True);return observed
        gaps=np.linalg.norm(knobs-observed.center,axis=1);i=int(np.argmin(gaps))
        if gaps[i]>KNOB_MEMORY_MATCH:return observed
        from dataclasses import replace
        print('KNOB_CENTER_FROM_MEMORY',json.dumps(dict(target=target,fitted=np.round(observed.center,3).tolist(),
            remembered=np.round(knobs[i],3).tolist(),gap_m=round(float(gaps[i]),3))),flush=True)
        return replace(observed,center=knobs[i].copy())

    def _accept_control(self,observed,point,target,kind):
        self.observed=observed;self.control_kind=kind
        if kind=='knob':
            from ..perception.handle_surface import observed_handle_surface
            front,_=observed_handle_surface(self.points,point)
            self.grasp_crop_center=np.median(front,axis=0)
            depth=front@observed.outward
            self.grasp_surface=front[depth>=np.median(depth)]
            self.knob_tracks[(target,self.motion_epoch())]=observed

    def planner(self,control,contact=False):
        from curobo._src.geom.types import Cuboid
        if self.prepare_arm is not None:self.arm=self.prepare_arm()
        points=self.points
        if contact:
            permitted=cKDTree(control.surface).query(points)[0]<=.008
            if self.control_kind=='button':
                delta=points-control.center;signed=delta@control.outward
                tangent=delta-signed[:,None]*control.outward
                # The fingertip/linkage may touch the local keypad face while
                # pressing. Only this observed coplanar patch is contact-
                # permitted; frame, counter and other geometry remain checked.
                permitted |= (abs(signed)<.006)&(np.linalg.norm(tangent,axis=1)<.045)
            points=points[~permitted]
        resolution=.01
        vox=np.unique(np.floor(points/resolution).astype(int),axis=0)
        boxes=[Cuboid(name=f'control_obs_{i}',pose=((lo+hi+1)*resolution/2).tolist()+[1.,0.,0.,0.],
                      dims=((hi-lo+1)*resolution).tolist()) for i,(lo,hi) in enumerate(merge_voxels(vox))]
        if self.precise_geometry is not None:
            from .convex_finger_contact import ObservedControlPlanner
            exact=self.precise_geometry()
            links=exact.planner_finger_links
            planner=ObservedControlPlanner(self.arm,self.geometry,boxes,allow_curobo_fallback=not contact,fast_ik=True,
                panel_points=np.empty((0,3)),hinge_xy=[0.,0.],reference_tcp=self.tcp()[:3,3],
                finger_geometry=exact,finger_links=links)
        else:planner=ObservedGraspPlanner(self.arm,self.geometry,boxes,allow_curobo_fallback=not contact,fast_ik=True)
        if self.control_kind=='knob':
            # User rule (2026-11-18): a grasp with an IK solution is executed,
            # not refused by the observed-world check. fix167 replays: with the
            # right knob selected, every knob grasp was refused because the
            # fingers overlapped observed panel boxes beside the knob.
            planner.ik_only=True
        return planner

    def execute(self,path,grip):
        for q in path:self.command(q,grip)
        self.grip=grip

    def plan_approach_contact(self,free,contact,pose,pre):
        """Qualify contact first, then reach its IK branch (shared oracle order)."""
        from .local_ik import local_ik
        q=self.joints();errors=[]
        for goal in contact.iter_solutions(q,pose[:3,3],pose[:3,:3]):
            seed=local_ik(self.geometry,goal,pre)
            if seed is None or not free.clear(seed):continue
            try:
                insertion=contact.plan(seed,pose[:3,3],pose[:3,:3])
                reach=free.plan_joint_goal(q,seed)
                return reach,insertion
            except RuntimeError as exc:errors.append(str(exc))
        raise RuntimeError('No complete approach/contact branch: '+str(errors or contact.ik_diagnostics[:2]))

    def set_gripper(self,value,*,require_occupied=True):
        q=self.joints();last=None;stable=0
        for _ in range(100):
            self.command(q,value);now=self.fingers()
            stable=stable+1 if last is not None and np.max(abs(now-last))<.0002 else 0
            last=now.copy()
            if stable>=10:break
        else:raise RuntimeError('gripper_not_settled')
        self.grip=value
        if require_occupied and value==255. and np.min(now)>.79:
            raise RuntimeError('empty_grasp_encoders')

    def grasp(self,control):
        self.set_gripper(0.)
        # Like the oracle's regrasp, retain the contact frame relative to the
        # raised bar. Its source here is a previously executed M2T2 proposal
        # and fresh observed indicator angle, never a fixture pose or joint.
        remembered=self.knob_grasps.get(self.active_track_key)
        if remembered is not None:
            old_control,old_pose=remembered
            r=Rotation.from_rotvec((control.angle-old_control.angle)*control.outward).as_matrix()
            pose=old_pose.copy();pose[:3,:3]=r@old_pose[:3,:3]
            pose[:3,3]=control.center+r@(old_pose[:3,3]-old_control.center)
            free=self.planner(control);contact=self.planner(control,True)
            try:
                pre=pose.copy();pre[:3,3]+=.07*control.outward
                reach,insert=self.plan_approach_contact(free,contact,pose,pre)
            except RuntimeError as exc:self.event('remembered_grasp_rejected',reason=str(exc))
            else:
                self.event('reuse_observed_m2t2_grasp',executed_tcp=pose.tolist())
                self.execute(reach,0.);self.execute(insert,0.);self.set_gripper(255.)
                return
        rows=[]
        for radius in (.08,.16,.3):
            crop=np.linalg.norm(self.points-self.grasp_crop_center,axis=1)<radius
            if crop.sum()<32:continue
            out=self.m2t2.predict(self.points[crop],self.rgb[crop]);proposed=[]
            for grasps,scores,contacts in zip(out['grasps'],out['grasp_confidence'],out['grasp_contacts']):
                proposed.extend(dict(tcp=grasp_to_tcp(g).tolist(),score=float(s),contact=c.tolist())
                                for g,s,c in zip(grasps,scores,contacts))
            rows=filter_observed_contacts(proposed,self.grasp_surface,.006)
            self.event('m2t2_crop',radius=radius,proposed=len(proposed),on_control=len(rows))
            if rows:break
        up=np.array([0.,0.,1.]);up-=control.outward*(control.outward@up);up/=np.linalg.norm(up)
        indicator=Rotation.from_rotvec(control.angle*control.outward).as_matrix()@up
        if not rows and self.control_kind=='knob':
            # M2T2 proposed no grasp touching the observed knob face (fix180
            # call 19: 0 proposals in the 8 cm crop, 387/687 wider, none on the
            # knob; the burner stayed on). Use the observed knob geometry: approach
            # straight into the face, fingers across the indicator bar, at the
            # wrist rolls the ranking below prefers. Each candidate is still
            # planned, executed and released physically like an M2T2 grasp.
            approach=-control.outward;across=np.cross(control.outward,indicator);across/=np.linalg.norm(across)
            for tilt in KNOB_GEOMETRIC_TILTS:
                tilted=Rotation.from_rotvec(np.radians(tilt)*indicator).as_matrix()
                z=tilted@approach;y=tilted@across;x=np.cross(y,z)
                tcp=np.eye(4);tcp[:3,:3]=np.column_stack([x,y,z]);tcp[:3,3]=control.center
                rows.append(dict(tcp=tcp.tolist(),score=0.,contact=control.center.tolist(),source='observed_knob_geometry'))
            self.event('knob_geometric_grasps',count=len(rows))
        rows.sort(key=lambda r:(abs(np.asarray(r['tcp'])[:3,1]@control.outward)>.5,
                                np.asarray(r['tcp'])[:3,2]@control.outward>-.5,
                                abs(np.asarray(r['tcp'])[:3,1]@indicator),
                                handle_grasp_rank(r['tcp'],control.surface,control.outward),-r['score']))
        from .handle_grasp_candidates import refine_handle_candidates
        rows=refine_handle_candidates(rows,control.surface,control.outward,align_panel=True,center_handle=True,limit=12)
        rows=parallel_gripper_symmetries(rows[:12])
        self.event('m2t2_candidates',count=len(rows))
        free=self.planner(control);contact=self.planner(control,True);errors=[]
        for row in rows:
            pose=np.array(row['tcp'])
            # Pinch the raised bar near its centre, not a learned contact on
            # the circular backing rim. A small axial offset keeps the lower
            # linkage clear of adjacent appliance handles.
            centre=control.center+.016*indicator
            shift=centre-pose[:3,3]
            pose[:3,3]+=shift-control.outward*(shift@control.outward)
            # Keep the pinch on the raised front indicator, not the backing
            # disk. This correction uses only the observed depth component.
            front=np.percentile(control.surface@control.outward,70)
            correction=max(.005,front-pose[:3,3]@control.outward)
            pose[:3,3]+=correction*control.outward
            try:
                pre=pose.copy();pre[:3,3]+=.07*control.outward
                reach,insert=self.plan_approach_contact(free,contact,pose,pre)
            except RuntimeError as exc:
                errors.append(str(exc));self.event('grasp_candidate_rejected',index=len(errors),reason=str(exc));continue
            self.event('selected_m2t2_grasp',proposal=row,executed_tcp=pose.tolist())
            self.execute(reach,0.);self.execute(insert,0.);self.set_gripper(255.)
            self.knob_grasps[self.active_track_key]=(control,self.tcp().copy())
            return
        self.event('grasp_plan_rejections',errors=errors)
        raise RuntimeError('no_reachable_m2t2_knob_grasp')

    def follow_contact(self,poses,control):
        planner=self.planner(control,True);q=self.joints();paths=[]
        for pose in poses:
            path=planner.plan(q,pose[:3,3],pose[:3,:3]);paths.append(path);q=path[-1]
        for path in paths:self.execute(path,255.)

    def release_withdraw(self,control):
        # Open fully, wait for the fingers, then back straight out along the
        # knob axis in short steps. fix184 call 19: the knob was OFF (0.00)
        # and held there for 2 s; the old partial opening (180) plus one
        # planned 7 cm withdraw swept the half-open fingers 2.2 cm sideways and
        # twisted them 6 deg across the bar, flicking the knob back to 1.19 rad
        # (ON). From that saved frame a full opening with the arm still leaves
        # the knob at 0.00 and the fingers clear. Two front knobs 43 cm apart
        # leave room for a full opening.
        self.set_gripper(0.)
        planner=self.planner(control,True)
        self.event('withdraw_initial_clearance',clear=planner.clear(self.joints()),detail=getattr(planner,'last_rejection',None))
        start=self.tcp();q=self.joints();paths=[]
        for k in range(1,KNOB_WITHDRAW_STEPS+1):
            pose=start.copy();pose[:3,3]+=KNOB_WITHDRAW_DISTANCE*k/KNOB_WITHDRAW_STEPS*control.outward
            path=planner.plan(q,pose[:3,3],pose[:3,:3]);paths.append(path);q=path[-1]
        for path in paths:self.execute(path,0.)

    def withdraw_further(self,control,distance=.10,steps=5):
        """Back the open gripper further out along the knob axis, for a clearer look."""
        planner=self.planner(control,True);start=self.tcp();q=self.joints();paths=[]
        for k in range(1,steps+1):
            pose=start.copy();pose[:3,3]+=distance*k/steps*control.outward
            path=planner.plan(q,pose[:3,3],pose[:3,:3]);paths.append(path);q=path[-1]
        for path in paths:self.execute(path,0.)

    def press_withdraw(self,control):
        from .coordinated_contact import panel_contact_pose
        from .appliance_controls import button_contact_rotations
        # A compact, closed fingertip avoids sweeping an open jaw across the
        # neighboring keypad. Empty closure is intentional for a button press.
        self.set_gripper(255.,require_occupied=False)
        direction=-control.outward
        free=self.planner(control);contact=self.planner(control,True);errors=[]
        orientations=list(button_contact_rotations(control.outward))
        attempts=[(fallback,tilt,rotation) for fallback in (False,True) for tilt,rotation in orientations]
        for fallback,tilt,rotation in attempts:
            free.allow_curobo_fallback=fallback
            self.event('button_try_orientation',tilt=tilt,rotation=rotation.tolist(),global_fallback=fallback)
            pose=panel_contact_pose(control.center,direction,rotation,self.pad_vertices())
            pre=pose.copy();pre[:3,3]+=.06*control.outward
            try:
                reach,press=self.plan_approach_contact(free,contact,pose,pre)
                withdraw=contact.plan(press[-1],pre[:3,3],pre[:3,:3])
            except RuntimeError as exc:errors.append(str(exc));continue
            self.event('button_contact_plan',tilt=tilt,pose=pose.tolist())
            self.execute(reach,255.);self.execute(press,255.)
            # Measured encoder settling, not appliance-truth contact feedback.
            last=self.joints();stable=0
            for _ in range(25):
                self.command(press[-1],255.);now=self.joints()
                stable=stable+1 if np.max(abs(now-last))<.00015 else 0;last=now
                if stable>=5:break
            self.execute(withdraw,255.);self.set_gripper(0.);return
        self.event('button_plan_rejections',errors=errors)
        raise RuntimeError('button_redock_required')
