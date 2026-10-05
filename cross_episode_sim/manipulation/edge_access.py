"""Bounded, contact-driven table-edge access for flat thin objects."""
from dataclasses import dataclass,replace
from pathlib import Path
import traceback
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.grasp_qualification import GraspQualification, equivalent_grasp
def collision_vertices(*args):
    from cross_episode_sim.fixtures.toaster_insertion import collision_vertices as native_vertices
    return native_vertices(*args)
from scipy.spatial import ConvexHull


@dataclass
class EdgeAccess:
    normal: np.ndarray
    tangent: np.ndarray
    boundary: float
    top: float
    overhang: float
    distance: float
    edge_xy: np.ndarray


def edge_frame(normal):
    normal=np.asarray(normal,dtype=float);normal=normal/np.linalg.norm(normal)
    return np.column_stack(([normal[1],-normal[0],0.],[normal[0],normal[1],0.],[0.,0.,1.]))


def outline_interval(points,across):
    """Intersection of the convex horizontal outline with an across-coordinate."""
    points=np.unique(np.asarray(points),axis=0)
    polygon=points[ConvexHull(points).vertices];hits=[]
    for p,q in zip(polygon,np.roll(polygon,-1,axis=0)):
        if min(p[0],q[0])-1e-9<=across<=max(p[0],q[0])+1e-9:
            if abs(q[0]-p[0])<1e-9:hits.extend([p[1],q[1]])
            else:hits.append(float(p[1]+(across-p[0])*(q[1]-p[1])/(q[0]-p[0])))
    if not hits:raise ValueError('Contact line misses thin object outline')
    return min(hits),max(hits)


def side_annotations(vertices,object_pose,normal):
    frame=edge_frame(normal);v=vertices@frame;lo,hi=v.min(0),v.max(0);poses=[]
    for pitch in (0.,15.,30.):
        R=Rotation.from_euler('x',pitch,degrees=True).as_matrix()@np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        for inset in (.006,.014,.022):
            for dx in (0.,.012,-.012):
                center=(lo+hi)/2;center[0]+=dx
                _,front=outline_interval(v[:,:2],center[0]);center[1]=front-inset
                pose=np.eye(4);pose[:3,:3]=frame@R
                pose[:3,3]=frame@(center-.008*R[:,2])
                poses.append(np.linalg.inv(object_pose)@pose)
    return np.asarray(poses)


def object_bodies(model,name):
    bodies={model.body(name).id}
    for bid in range(model.nbody):
        if model.body_parentid[bid] in bodies:bodies.add(bid)
    return bodies


def tabletop_bounds(model,data,bodies):
    points=[];corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        v=(model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:])@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        lo,hi=v.min(0),v.max(0)
        if .3<hi[2]<1.3 and hi[2]-lo[2]<.15:points.append(v)
    if not points:raise ValueError('No physical horizontal tabletop')
    return np.concatenate(points)


def edge_candidates(vertices,table_vertices,base_xy):
    """Rectangular tabletop edges in world coordinates; retain finite support."""
    low,high=table_vertices.min(0),table_vertices.max(0)
    extent=np.ptp(vertices,axis=0)
    if not .006<=extent[2]<=.026:raise ValueError('Edge access requires a flat object 6–26 mm thick')
    if not high[2]-.005<=vertices[:,2].min()<=high[2]+.015:
        raise ValueError('Thin object is not on the selected tabletop')
    footprint=np.unique(table_vertices[:,:2],axis=0)
    polygon=footprint[ConvexHull(footprint).vertices]
    if len(polygon)!=4:raise ValueError('Edge fallback currently requires a rectangular tabletop')
    directions=np.roll(polygon,-1,axis=0)-polygon
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    if np.max(np.abs(np.sum(directions*np.roll(directions,-1,axis=0),axis=1)))>1e-5:
        raise ValueError('Edge fallback currently requires a rectangular tabletop')
    result=[]
    for point,along in zip(polygon,directions):
        normal=np.array([along[1],-along[0]]);tangent=np.array([normal[1],-normal[0]])
        projected=vertices[:,:2]@normal;span=np.ptp(projected)
        base_clearance=min(.040,.35*span)
        if base_clearance<.024:continue
        across=vertices[:,:2]@tangent;mid=(across.min()+across.max())/2
        # A rotated object's furthest corner overhangs before its grasp face.
        # Leave jaw clearance at every lateral side-grasp candidate instead.
        fronts=[outline_interval(np.column_stack((across,projected)),mid+dx)[1]
                for dx in (0.,.012,-.012)]
        overhang=base_clearance+float(projected.max()-min(fronts))+.002
        if overhang>span/2-.010:continue
        # Expose additional book surface so the lower jaw and tilted trailing
        # edge can clear the support on lift. Limit the extra push by support.
        overhang+=min(.020,max(0.,span/2-.020-overhang))
        boundary=float(point@normal)
        distance=boundary+overhang-projected.max()
        if not -.02<=distance<=.30:continue
        distance=max(0.,distance)
        across=vertices[:,:2]@tangent;table_across=table_vertices[:,:2]@tangent
        if across.min()<table_across.min()+.025 or across.max()>table_across.max()-.025:continue
        center=vertices[:,:2].mean(0)
        edge_xy=center+(boundary-center@normal)*normal
        result.append(EdgeAccess(normal,tangent,boundary,float(high[2]),overhang,float(distance),edge_xy))
    result.sort(key=lambda e:e.distance+.2*np.linalg.norm(np.asarray(base_xy)-e.edge_xy))
    if not result:raise ValueError('No edge leaves sufficient object support within bounded push distance')
    return result


def install_thin_scene(scene,obj,table,yaw=0.,inset=.06):
    """Author a flat, fully supported initial pose; no task-time pose modification."""
    tree=ET.parse(scene);m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    b=d.body(obj);vertices=collision_vertices(m,d,object_bodies(m,obj));local=(vertices-b.xpos)@b.xmat.reshape(3,3)
    order=np.argsort(np.ptp(local,axis=0));thin=int(order[0]);wide=[i for i in range(3) if i!=thin]
    R=np.eye(3)[wide+[thin]]
    if np.linalg.det(R)<0:R[0]*=-1
    R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@R
    rotated=local@R.T;surface=tabletop_bounds(m,d,{m.body(table).id});low,high=surface.min(0),surface.max(0)
    xyz=b.xpos.copy();xyz[1]=high[1]-inset-rotated[:,1].max();xyz[2]=high[2]+.001-rotated[:,2].min()
    if np.ptp(rotated[:,2])>.026:raise ValueError('Selected asset is too thick for the thin-object pilot')
    body=tree.find(f".//body[@name='{obj}']");body.set('pos',' '.join(map(str,xyz)))
    body.set('quat',' '.join(map(str,Rotation.from_matrix(R).as_quat(scalar_first=True))))
    tree.write(scene);return xyz


def thin_annotations(scene,obj,output):
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    pose=np.eye(4);pose[:3,:3]=d.body(obj).xmat.reshape(3,3);pose[:3,3]=d.body(obj).xpos
    transforms=side_annotations(collision_vertices(m,d,object_bodies(m,obj)),pose,[0.,1.])
    path=Path(output)/'thin_side_hypotheses.npz';np.savez_compressed(path,transforms=transforms);return path


class ThinEdgeAccessMixin:
    """Explicit thin-object fallback, usable by task controllers with shared pick APIs."""
    def navigation_penetration(self,data,carrying):
        if not getattr(self,'edge_pushing',False):return super().navigation_penetration(data,carrying)
        worst=0.
        for c in data.contact:
            a,b=self.model.geom_bodyid[[c.geom1,c.geom2]]
            na,nb=self.model.body(a).name or '',self.model.body(b).name or ''
            ra,rb=na.startswith(self.profile.namespace),nb.startswith(self.profile.namespace)
            if na.startswith('floor_') or nb.startswith('floor_'):continue
            if ra!=rb:
                other,robot=(b,na) if ra else (a,nb)
                if other in self.bread_bids and (self.embodiment.is_finger(robot) or robot.endswith(('/left_follower','/right_follower'))):continue
                worst=max(worst,-float(c.dist)+.001)
        return worst

    def rebuild_edge_planner(self):
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world()

    def refresh_edge_grasps(self,normal):
        self.local_annotations=side_annotations(self.bread_vertices(),self.bread_pose(),normal)
        attempted=list(getattr(self,'_edge_failed_poses',[]))
        previous=getattr(self.args,'previous_trials',None)
        if previous:
            for path in sorted(Path(previous).glob('trial_*/attempted_grasp.json')):
                history=path.with_name('attempted_grasps.json')
                entries=json.loads(history.read_text()) if history.exists() else [json.loads(path.read_text())]
                attempted.extend(item['planned_object_T_tcp'] for item in entries)
        self.physically_rejected_annotation_variants={i for i,pose in enumerate(self.local_annotations)
            if any(equivalent_grasp(pose,old) for old in attempted)}
        self.preplanned_moves={};self.active_family='any'
        self.annotation_path=self.output/'edge_side_hypotheses.npz'
        np.savez_compressed(self.annotation_path,transforms=self.local_annotations)
        self.args.annotation_source='geometry_hypotheses'
        self.record(edge_grasps_regenerated=True,excluded_previous_contacts=len(self.physically_rejected_annotation_variants))

    def edge_corridor_clear(self,edge):
        v=self.bread_vertices();end=v.copy();end[:,:2]+=edge.normal*edge.distance
        swept=np.concatenate((v,end));lo,hi=swept.min(0),swept.max(0)
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for g in range(self.model.ngeom):
            bid=self.model.geom_bodyid[g];name=self.model.body(bid).name or ''
            if bid in self.bread_bids or bid in self.support_bids or name.startswith(self.profile.namespace):continue
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):continue
            p=(self.model.geom_aabb[g,:3]+corners*self.model.geom_aabb[g,3:])@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
            a,b=p.min(0),p.max(0)
            if b[2]<=edge.top+.003 or a[2]>hi[2]+.005:continue
            if np.all(b[:2]>lo[:2]-.008) and np.all(a[:2]<hi[:2]+.008):return False
        return True

    def edge_fallback_eligible(self):
        if not getattr(self.args, 'thin_edge_fallback', False):return False
        try:
            table=tabletop_bounds(self.model,self.data,self.support_bids)
            edge_candidates(self.bread_vertices(),table,self.base_pose()[:2])
        except ValueError as exc:
            self.record(edge_fallback_skipped=str(exc))
            return False
        return True

    def prepare_pickup(self):
        if not self.edge_fallback_eligible():
            return super().prepare_pickup()
        return self.prepare_flat_object_pickup()

    def prepare_flat_object_pickup(self):
        """Shared supported push -> regenerate side grasps -> physical pickup setup."""
        if not hasattr(self,"_edge_original_gripper_open"):
            self._edge_original_gripper_open = self.profile.gripper_open
        self._edge_flat_rotation = self.bread_pose()[:3,:3].copy()
        self.tuck_for_navigation()
        v=self.bread_vertices();table=tabletop_bounds(self.model,self.data,self.support_bids)
        candidates=edge_candidates(v,table,self.base_pose()[:2]);rejected=[]
        for edge in candidates:
            if not self.edge_corridor_clear(edge):
                rejected.append('Push corridor blocked');continue
            xy=(v.min(0)+v.max(0))[:2]/2+edge.normal*.42
            yaw=float(np.arctan2(-edge.normal[1],-edge.normal[0]))
            try:path=self.plan_route(xy,False,face=yaw)
            except RuntimeError as exc:rejected.append(str(exc));continue
            self._accepted_route=path;self.task_navigate(xy,False,face=yaw)
            self.edge_access=edge;break
        else:raise RuntimeError(f'No reachable clear table edge: {rejected}')
        self._edge_pick_yaw = float(self.base_pose()[2])
        self.report['edge_access']=dict(normal=edge.normal.tolist(),boundary=edge.boundary,
            planned_push_m=edge.distance,target_overhang_m=edge.overhang,top=edge.top)
        self.profile=replace(self.profile,gripper_open=180.);self.embodiment.profile=self.profile
        self.embodiment.open_gripper(self);self.tick(.5);self.untuck_for_manipulation()
        self.rebuild_edge_planner()
        try:
            if (getattr(self.args,"flat_pick_policy","direct-first")=="push-first"
                    or getattr(self,"_edge_force_push",False)):
                raise RuntimeError("No annotated grasp passed robot reachability and collision checks")
            self._edge_selected=GraspQualification.select_annotated_grasp(self)
            self.record(direct_thin_grasp_feasible=True);return
        except RuntimeError as exc:
            if str(exc)!='No annotated grasp passed robot reachability and collision checks':raise
            self.record(flat_pick_policy=getattr(self.args,"flat_pick_policy","direct-first"),
                        edge_fallback_reason="push-first policy" if getattr(self.args,"flat_pick_policy",None)=="push-first" else str(exc))
        self.push_to_edge(edge)
        self.tuck_for_navigation()
        center=(self.bread_vertices().min(0)+self.bread_vertices().max(0))[:2]/2
        self.task_navigate(center+edge.normal*(.52+getattr(self.args,"edge_dock_offset",0.)),False,face=yaw)
        self.embodiment.open_gripper(self);self.tick(.5);self.untuck_for_manipulation()
        self.refresh_edge_grasps(edge.normal);self.rebuild_edge_planner()
        self.review_phase='SIDE PICK AFTER PHYSICAL TABLE EDGE PUSH'
        self._edge_selected=GraspQualification.select_annotated_grasp(self)

    def pickup_support_top(self):
        return float(tabletop_bounds(self.model,self.data,self.support_bids)[:,2].max())

    def clear_held_pickup_support(self):
        """Correct measured settling before the shared vertical-clearance gate.

        A wrist displacement does not imply equal object displacement: a rim
        grasp can rotate as the trailing edge leaves the support. Keep the
        existing clearance gate, and remeasure after bounded local moves.
        """
        # A successful initial lift must not acquire new surface-geometry
        # requirements merely because optional recovery is enabled.
        lifted=float(self.bread_pose()[2,3]-self.pickup_start_height)
        if lifted>=.015 and not self.support_contacts(table=True):
            return
        if len(self.finger_object_contact()['fingers'])!=2:
            return
        surface_z=self.pickup_support_top()
        correction=0.
        for attempt in range(4):
            lifted=float(self.bread_pose()[2,3]-self.pickup_start_height)
            bottom_gap=float(self.bread_vertices()[:,2].min()-surface_z)
            supported=bool(self.support_contacts(table=True))
            if lifted>=.020 and bottom_gap>=.010 and not supported:
                return
            if len(self.finger_object_contact()['fingers'])!=2:
                return
            remaining=.06-correction
            if remaining<.001:
                return
            increment=min(remaining,max(.020-lifted,.012-bottom_gap,.004))
            target=self.tcp().copy();target[2,3]+=increment
            self.record(pickup_clearance_correction=attempt+1,
                        measured_object_lift_m=lifted,object_bottom_gap_m=bottom_gap,
                        supported_object_clearance_lift_m=increment,
                        total_clearance_correction_m=correction+increment)
            # The lift prefix routes this through the same local, actual-mesh
            # checked contact IK as the initial lift, avoiding a free pose plan.
            super().move('lift bread clearance correction',target)
            correction+=increment

    def move(self,stage,pose):
        result=super().move(stage,pose)
        if (stage=='lift bread vertically' and getattr(self.args,'thin_edge_fallback',False)
                and getattr(self,'holding_loaf',False)):
            self.clear_held_pickup_support()
        if (stage=='lift bread' and getattr(self.args,'thin_edge_fallback',False)
                and hasattr(self, '_edge_flat_rotation')
                and getattr(self,'holding_loaf',False)):
            lifted=float(self.bread_pose()[2,3]-self.pickup_start_height)
            if (.07<=lifted<.11 and not self.support_contacts(table=True)
                    and len(self.finger_object_contact()['fingers'])==2):
                extra=min(.05,.12-lifted)
                target=self.tcp().copy();target[2,3]+=extra
                self.record(edge_lift_completion_m=extra,measured_object_lift_m=lifted)
                super().move('complete thin object lift before qualification',target)
        return result

    def placement_reference_pose(self):
        held=self.bread_pose().copy()
        if getattr(self.args, 'thin_edge_fallback', False) and hasattr(self, '_edge_flat_rotation'):
            yaw=float(self.base_pose()[2])-self._edge_pick_yaw
            held[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@self._edge_flat_rotation
            self.record(placement_orientation_policy='restore flat support orientation facing destination')
        return held

    def pick_payload(self):
        try:
            for attempt in range(3):
                try:
                    return super().pick_payload()
                except RuntimeError as exc:
                    pickup_failure=any(text in str(exc) for text in (
                        'did not clear the counter', 'did not remain physically grasped',
                        'Grasp qualification failed', 'Failed pickup:', 'Object dropped'))
                    if (not getattr(self.args,'thin_edge_fallback',False) or not pickup_failure
                            or attempt==2 or not self.support_contacts(table=True)):
                        raise
                    selection=self.report.get('annotation_selection',{})
                    pose=selection.get('local_transform')
                    if pose is not None:
                        self._edge_failed_poses=getattr(self,'_edge_failed_poses',[])+[pose]
                    self.record(edge_physical_retry=attempt+1,reason=str(exc),object_still_supported=True)
                    self.holding_loaf=False
                    if getattr(self,'attached',False):
                        self.planner.detach_block();self.attached=False
                    self.embodiment.open_gripper(self);self.tick(.5)
                    retreat=self.tcp().copy();retreat[2,3]+=.10
                    self.mesh_contact_move('withdraw after supported failed pickup',retreat)
                    self._edge_force_push=True;self._edge_selected=None;self.preplanned_moves={}
        finally:
            # Restore the task's release aperture without commanding the held jaws.
            original=getattr(self, "_edge_original_gripper_open", None)
            if original is not None:
                self.profile=replace(self.profile, gripper_open=original)
                self.embodiment.profile=self.profile
                del self._edge_original_gripper_open

    def select_annotated_grasp(self):
        selected=getattr(self,'_edge_selected',None)
        if selected is not None:self._edge_selected=None;return selected
        return super().select_annotated_grasp()

    def closed_pusher_pose(self,edge,advance):
        from cross_episode_sim.controller.base import geom_box
        frame=edge_frame(edge.normal);R=frame@np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        tcp=self.tcp();points=[];corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for g in range(self.model.ngeom):
            name=self.model.body(self.model.geom_bodyid[g]).name or ''
            if not name.endswith(('/left_pad','/right_pad')) or not self.model.geom_contype[g]:continue
            c,h=geom_box(self.model,g)
            points.extend(self.data.geom_xpos[g]+(c+corners*h)@self.data.geom_xmat[g].reshape(3,3).T)
        offsets=(np.asarray(points)-tcp[:3,3])@tcp[:3,:3]@R.T
        projection=offsets[:,:2]@edge.normal
        front=offsets[projection>projection.max()-1e-5].mean(0)
        v=self.bread_vertices();n=v[:,:2]@edge.normal;t=v[:,:2]@edge.tangent
        across=(t.min()+t.max())/2
        rear_n,_=outline_interval(np.column_stack((t,n)),across)
        rear=v[n<rear_n+.015]
        contact=np.r_[edge.normal*(rear_n+advance)+edge.tangent*across,rear[:,2].max()+.008]
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=contact-front
        pose[2,3]=max(pose[2,3],edge.top+.002-offsets[:,2].min())
        return pose

    def lower_pusher_behind_object(self,pose,normal):
        # Pad extents alone do not bound the wider follower linkages. At the
        # nominal four-millimetre pad setback a linkage can already overlap a
        # rotated/irregular book. Search a collision-checked lowering path before
        # enabling intentional pushing contacts; never relax the approach check.
        rejected=[]
        for setback in (0., .01, .02, .04, .06):
            candidate=pose.copy()
            candidate[:2,3]-=setback*normal
            try:
                path=self.plan_contact_path('lower closed pads behind thin object',candidate)
            except RuntimeError as exc:
                rejected.append(str(exc));continue
            self.record(edge_lowering_extra_setback_m=setback,
                        edge_lowering_rejected_paths=rejected)
            self.mesh_contact_move('lower closed pads behind thin object',candidate,path=path)
            return
        else:
            raise RuntimeError(f'No full-gripper-clear lowering path behind thin object: {rejected}')

    def push_to_edge(self,edge):
        self.review_phase='PHYSICALLY PUSH THIN OBJECT TO TABLE EDGE'
        self.report['edge_push_used']=True
        self.embodiment.command_gripper(self.data,self.profile.gripper_close);self.tick(.6)
        self.rebuild_edge_planner()
        pose=self.closed_pusher_pose(edge,-.004);pre=pose.copy();pre[2,3]+=.10
        self.move('approach behind thin object',pre)
        self.lower_pusher_behind_object(pose,edge.normal)
        initial=self.bread_pose().copy()
        self.edge_pushing=True
        try:
            for step in range(100):
                v=self.bread_vertices();overhang=float((v[:,:2]@edge.normal).max()-edge.boundary)
                com=self.data.subtree_com[self.model.body(self.object_name).id]
                if com[:2]@edge.normal>edge.boundary-.008:raise RuntimeError('Object centre reached support margin; stop pushing')
                if v[:,2].min()<edge.top-.025:raise RuntimeError('Thin object fell during edge push')
                if overhang>=edge.overhang:break
                target=self.closed_pusher_pose(edge,min(.004,edge.overhang-overhang))
                self.mesh_contact_move('push thin object toward table edge',target)
                if np.linalg.norm(self.bread_pose()[:2,3]-initial[:2,3])>edge.distance+.04:
                    raise RuntimeError('Thin object moved beyond bounded edge push')
            else:raise RuntimeError('Thin object did not reach edge within push budget')
            self.record(edge_push_complete=True,physical_displacement_m=float(np.linalg.norm(self.bread_pose()[:3,3]-initial[:3,3])),
                        overhang_m=overhang,centre_support_margin_m=float(edge.boundary-com[:2]@edge.normal))
            retreat=self.tcp().copy();retreat[2,3]+=.10
            self.mesh_contact_move('withdraw pusher from thin object',retreat)
        finally:self.edge_pushing=False
        self.report['edge_push_overhang_m']=overhang
        self.embodiment.open_gripper(self);self.tick(.5)
