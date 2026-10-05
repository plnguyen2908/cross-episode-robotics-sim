"""Flat bread on a free plate, with physical edge-sliding fallback for side pickup."""
import copy
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.fixtures.toaster_insertion import ToasterInsertionTest, collision_vertices, TOASTER
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from cross_episode_sim.paths import MOLMO_OBJECTS_DIR

PLATE='bread_plate'
PLATE_XML=MOLMO_OBJECTS_DIR/'Plate_29/model.xml'


def install_plate(scene,obj):
    tree=ET.parse(scene);root=tree.getroot();world=root.find('worldbody')
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    bread=world.find(f"body[@name='{obj}']")
    old=d.body(obj);points=collision_vertices(m,d,{m.body(obj).id})
    local=(points-old.xpos)@old.xmat.reshape(3,3)
    table_top=points[:,2].min()-.001
    plate_tree=ET.parse(PLATE_XML)
    # Namespace all asset references and names, keeping native meshes and inertia.
    for node in plate_tree.iter():
        for key in ('name','mesh','material','texture'):
            if node.get(key):node.set(key,PLATE+'_'+node.get(key))
    plate=copy.deepcopy(plate_tree.find('worldbody/body'));plate.set('name',PLATE)
    pm=mujoco.MjModel.from_xml_path(str(PLATE_XML));pd=mujoco.MjData(pm);mujoco.mj_forward(pm,pd)
    vertices=collision_vertices(pm,pd,set(range(pm.nbody)))
    center=old.xpos.copy();center[2]=table_top-vertices[:,2].min()+.001
    # Keep the whole plate supported near the front tabletop edge. After a
    # physical slide its overhanging bread edge leaves room for the lower jaw.
    fronts=[]
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        points=(m.geom_aabb[g,:3]+corners*m.geom_aabb[g,3:])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        low,high=points.min(0),points.max(0)
        if abs(high[2]-table_top)<.002 and low[0]<center[0]<high[0] and low[1]<center[1]<high[1]:
            fronts.append(high[1])
    if not fronts:raise ValueError('No supporting tabletop edge for bread plate')
    center[1]=max(fronts)-vertices[:,1].max()-.008
    plate.set('pos',' '.join(map(str,center)));ET.SubElement(plate,'freejoint',name=PLATE+'_joint')
    world.append(plate);root.find('asset').extend(copy.deepcopy(list(plate_tree.find('asset'))))
    bread_pos=center.copy();bread_pos[2]=center[2]+vertices[:,2].max()-local[:,2].min()+.002
    bread.set('quat','1 0 0 0');bread.set('pos',' '.join(map(str,bread_pos)))
    tree.write(scene)
    return bread_pos


def flat_annotations(vertices,object_pose):
    low,high=vertices.min(0),vertices.max(0);poses=[]
    # Upper/lower pad pinch across the slice thickness from its front edge.
    # Final contact/collision validation decides if the plate leaves enough space.
    for pitch in (0.,15.,30.):
        for inset in (.006,.014,.022):
            for dx in (0.,.012,-.012):
                R=Rotation.from_euler('x',pitch,degrees=True).as_matrix()@np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
                pose=np.eye(4);pose[:3,:3]=R
                center=(low+high)/2;center[0]+=dx;center[1]=high[1]-inset
                pose[:3,3]=center-.008*R[:,2]
                poses.append(np.linalg.inv(object_pose)@pose)
    return np.asarray(poses)


def plate_annotations(scene,obj,output):
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    pose=np.eye(4);pose[:3,:3]=d.body(obj).xmat.reshape(3,3);pose[:3,3]=d.body(obj).xpos
    poses=flat_annotations(collision_vertices(m,d,{m.body(obj).id}),pose)
    path=Path(output)/'flat_bread_hypotheses.npz';np.savez_compressed(path,transforms=poses)
    return path


class ToasterPlateTest(ToasterInsertionTest):
    video_filename='plate_to_toaster.mp4'

    def __init__(self,args,selection):
        self.pushing=False
        super().__init__(args,selection)
        self.profile=replace(self.profile,gripper_open=180.)
        self.embodiment.profile=self.profile
        self.plate_bids=self.descendants(PLATE)
        self.report.update(object_initial_pose='flat slice on free native Plate_29',
                           plate_fallback_used=False,tracked_objects=[self.object_name,PLATE],
                           task='pick flat bread from plate; slide to edge if blocked; insert into toaster')

    def plate_vertices(self):return collision_vertices(self.model,self.data,self.plate_bids)

    def navigation_penetration(self,data,carrying):
        if not self.pushing:return super().navigation_penetration(data,carrying)
        worst=0.
        for c in data.contact:
            a,b=self.model.geom_bodyid[[c.geom1,c.geom2]]
            na,nb=self.model.body(a).name or '',self.model.body(b).name or ''
            ra,rb=na.startswith('robot_0/'),nb.startswith('robot_0/')
            if na.startswith('floor_') or nb.startswith('floor_'):continue
            if ra!=rb:
                other,robot=(b,na) if ra else (a,nb)
                if other in self.bread_bids and (self.embodiment.is_finger(robot) or robot.endswith(('/left_follower','/right_follower'))):continue
                worst=max(worst,-float(c.dist)+.001)
        return worst

    def refresh_side_grasps(self):
        self.local_annotations=flat_annotations(self.bread_vertices(),self.bread_pose())
        self.physically_rejected_annotation_variants=set();self.preplanned_moves={}
        self.active_family='any'

    def select_annotated_grasp(self):
        cached=getattr(self,'_plate_selected',None)
        if cached is not None:
            self._plate_selected=None
            return cached
        return super().select_annotated_grasp()

    def prepare_pickup(self):
        super().prepare_pickup()
        self.support_bids=self.plate_bids
        self.table_gids|={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.plate_bids and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
        self.refresh_side_grasps()
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world()
        try:
            self._plate_selected=GraspQualification.select_annotated_grasp(self)
            self.record(plate_direct_pick_feasible=True)
            return
        except RuntimeError as exc:
            self.record(plate_direct_pick_feasible=False,direct_pick_rejection=str(exc))
        self.slide_to_plate_edge()
        self.tuck_for_navigation()
        self.task_navigate(self.bread_pose()[:2,3]+[0.,.52],False,face=-np.pi/2)
        self.embodiment.open_gripper(self);self.tick(.5)
        self.untuck_for_manipulation()
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world()
        self.refresh_side_grasps()
        self._plate_selected=GraspQualification.select_annotated_grasp(self)

    def slide_to_plate_edge(self):
        from cross_episode_sim.controller.base import geom_box
        self.review_phase='PHYSICALLY SLIDE BREAD TO PLATE EDGE'
        self.report['plate_fallback_used']=True
        self.embodiment.command_gripper(self.data,self.profile.gripper_close);self.tick(.6)
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world()
        R=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]);tcp=self.tcp();points=[]
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for g in range(self.model.ngeom):
            name=self.model.body(self.model.geom_bodyid[g]).name or ''
            if not name.endswith(('/left_pad','/right_pad')) or not self.model.geom_contype[g]:continue
            c,h=geom_box(self.model,g)
            points.extend(self.data.geom_xpos[g]+(c+corners*h)@self.data.geom_xmat[g].reshape(3,3).T)
        offsets=(np.asarray(points)-tcp[:3,3])@tcp[:3,:3]@R.T
        # Use the forward pad face as the pusher, above the supporting plate.
        front=offsets[offsets[:,1]>offsets[:,1].max()-1e-5].mean(0)
        vertices=self.bread_vertices();low,high=vertices.min(0),vertices.max(0)
        contact=np.array([(low[0]+high[0])/2,low[1]-.004,high[2]+.008])
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=contact-front
        pre=pose.copy();pre[2,3]+=.12
        self.move('approach behind bread for sliding',pre)
        for extra in (0.,.003,.006,.009):
            candidate=pose.copy();candidate[2,3]+=extra
            try:
                path=self.plan_contact_path('lower pusher behind bread',candidate)
                self.mesh_contact_move('lower pusher behind bread',candidate,path=path)
                self.record(pusher_rim_clearance_adjustment_m=extra)
                break
            except RuntimeError:
                if extra==.009:raise
        initial=self.bread_pose()[:3,3].copy();plate_initial=self.data.body(PLATE).xpos.copy()
        self.pushing=True
        try:
            for index in range(110):
                vertices=self.bread_vertices();plate=self.plate_vertices()
                overhang=float(vertices[:,1].max()-plate[:,1].max())
                if overhang>=.040:break
                goal=self.tcp().copy()
                goal[0,3]=(vertices[:,0].min()+vertices[:,0].max())/2-front[0]
                goal[1,3]=min(goal[1,3]+.006,vertices[:,1].min()-front[1]+.006)
                rear=vertices[vertices[:,1]<vertices[:,1].min()+.015]
                goal[2,3]=rear[:,2].max()+.008+extra-front[2]
                self.mesh_contact_move('slide bread toward plate edge',goal)
                if self.bread_vertices()[:,2].min()<plate[:,2].min()-.015:
                    raise RuntimeError('Bread fell off the plate during sliding')
            else:raise RuntimeError('Bread did not reach graspable plate edge within bounded push')
            self.record(physical_plate_slide=True,bread_displacement_m=float(np.linalg.norm(self.bread_pose()[:3,3]-initial)),
                        plate_displacement_m=float(np.linalg.norm(self.data.body(PLATE).xpos-plate_initial)),
                        bread_overhang_m=overhang)
            retreat=self.tcp().copy();retreat[2,3]+=.10
            self.mesh_contact_move('withdraw pusher above plate',retreat)
        finally:self.pushing=False
        self.embodiment.open_gripper(self);self.tick(.5)
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world()

    def transport_payload(self):
        self.tuck_loaded_for_navigation()
        self.task_navigate(self.data.body(TOASTER).xpos[:2]+[.10,.40],True,face=-np.pi/2)
        return super().transport_payload()

    def place_payload(self):
        # Side pickup needs a narrow aperture near the plate. At the toaster,
        # use the native fully open command for release and withdrawal.
        self.profile=replace(self.profile,gripper_open=0.)
        self.embodiment.profile=self.profile
        return super().place_payload()

    def withdraw_after_insertion(self):
        start=self.tcp().copy();rejected=[]
        for dy,dz,tilt in ((.06,.025,-35.),(.09,.02,-45.),(.04,.025,-25.)):
            retreat=start.copy();retreat[:3,3]+=[0.,dy,dz]
            retreat[:3,:3]=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@start[:3,:3]
            try:
                path=self.plan_contact_path('withdraw from inserted slice',retreat)
            except RuntimeError as exc:
                rejected.append(str(exc));continue
            self.mesh_contact_move('withdraw from inserted slice',retreat,path=path)
            self.tuck_for_navigation();self.tick(.5)
            return
        raise RuntimeError(f'No clear toaster withdrawal: {rejected}')

    def insertion_rotation(self):
        # The pinched front edge becomes the top edge; thickness aligns with slot width.
        return Rotation.from_euler('z',90,degrees=True).as_matrix()@Rotation.from_euler('x',90,degrees=True).as_matrix()

    def plan_curved_insertion(self,obj,inv,tilt):
        """Preflight lower-then-straighten around the bread, in scratch data only."""
        live=self.data
        probe=mujoco.MjData(self.model)
        probe.qpos[:]=live.qpos;probe.qvel[:]=live.qvel;probe.ctrl[:]=live.ctrl
        mujoco.mj_forward(self.model,probe)
        addresses=[self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id] for n in self.planner.names]
        relative=np.linalg.inv(inv)
        targets=[]
        for angle in np.linspace(tilt,0.,max(2,int(abs(tilt)/3)+1)):
            target=obj.copy()
            target[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix()@obj[:3,:3]
            target[2,3]+=.010*abs(angle)/max(abs(tilt),1.)
            targets.append(target@inv)
        path=[]
        try:
            self.data=probe
            for target in targets:
                segment=self.plan_contact_path('lower and straighten bread in slot',target)
                path.extend(segment)
                probe.qpos[addresses]=segment[-1];mujoco.mj_forward(self.model,probe)
                held=self.tcp()@relative
                probe.joint(self.object_joint).qpos[:3]=held[:3,3]
                probe.joint(self.object_joint).qpos[3:]=Rotation.from_matrix(held[:3,:3]).as_quat(scalar_first=True)
                mujoco.mj_forward(self.model,probe)
        finally:self.data=live
        return path

    def reach_insertion_approach(self,obj,local,inv):
        rejected=[]
        current=np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
        for tilt in (-40.,-25.,-55.,0.):
            above=obj.copy();above[:3,:3]=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@obj[:3,:3]
            rotated=local@above[:3,:3].T
            above[2,3]=self.rim_height+.025-rotated[:,2].min()
            target=above@inv
            try:
                goal=list(target[:3,3]-[0,0,self.embodiment.planner_tool_offset()])+list(Rotation.from_matrix(target[:3,:3]).as_quat(scalar_first=True))
                path=self.planner.plan(current.tolist(),goal)
                self.preplanned_moves['align bread above toaster slot']=(target,path)
                self.move('align bread above toaster slot',target)
                descent=self.plan_curved_insertion(obj,inv,tilt)
                return descent
            except RuntimeError as exc:
                rejected.append(str(exc));current=np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
        raise RuntimeError(f'No complete side-grasp insertion path: {rejected}')
