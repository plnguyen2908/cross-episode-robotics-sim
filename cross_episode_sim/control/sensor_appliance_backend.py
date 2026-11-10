"""Sensor-only M2T2 knob grasps and geometric fingertip button presses.

All geometry is in the stationary robot-base frame. This backend deliberately
does not navigate: callers dock using the observed map before invoking it.
"""
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
        for name in ('wrist','navigation'):
            camera,valid=views[name];im=Image.fromarray(camera.rgb)
            result=self.pointer.point(im,target)
            self.event('point_control',camera=name,result=result)
            for pixel in result['points']:
                # Resolve small control text at a larger image scale. This is
                # an image-derived crop, never a scene object bounding box.
                if kind=='button':
                    x,y=pixel;w,h=im.size
                    bounds=(max(0,int(x)-90),max(0,int(y)-90),min(w,int(x)+90),min(h,int(y)+90))
                    crop=im.crop(bounds).resize((540,540))
                    refined=self.pointer.point(crop,target)
                    if refined['points']:
                        px,py=refined['points'][0]
                        pixel=(bounds[0]+px*(bounds[2]-bounds[0])/540,
                               bounds[1]+py*(bounds[3]-bounds[1])/540)
                    self.event('refine_button',camera=name,result=refined)
                try:
                    point=pixel_point(camera,valid,pixel)
                    observed=fit_control(self.points,point,kind)
                    self.event('control_geometry',kind=kind,center=observed.center.tolist(),
                               normal=observed.outward.tolist(),angle=observed.angle)
                    self._accept_control(observed,point,target,kind)
                    return observed
                except ValueError as exc:errors.append(str(exc))
        raise RuntimeError('control_not_observed: '+str(errors))

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
            return ObservedControlPlanner(self.arm,self.geometry,boxes,allow_curobo_fallback=not contact,fast_ik=True,
                panel_points=np.empty((0,3)),hinge_xy=[0.,0.],reference_tcp=self.tcp()[:3,3],
                finger_geometry=exact,finger_links=links)
        return ObservedGraspPlanner(self.arm,self.geometry,boxes,allow_curobo_fallback=not contact,fast_ik=True)

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
        # Reuse the established knob release pattern: clear the bar with a
        # partial opening, withdraw, then open fully away from the appliance.
        self.set_gripper(180.)
        pose=self.tcp();pose[:3,3]+=.07*control.outward
        planner=self.planner(control,True)
        self.event('withdraw_initial_clearance',clear=planner.clear(self.joints()),detail=getattr(planner,'last_rejection',None))
        self.execute(planner.plan(self.joints(),pose[:3,3],pose[:3,:3]),180.)
        self.set_gripper(0.)

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
