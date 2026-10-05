"""Open the native lid, load a verified object, close, and press power."""
import copy
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.blender_lid import BlenderLidTest, BLENDER, LID, GENERATED, lid_closed_state
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.molmo_objects import OUTPUT
from cross_episode_sim.paths import read_localized

BUTTON='skill_blender_power_button_main'
BUTTON_BODY='skill_blender_power_button'


def next_power_state(on, previous_contact, contact, closed):
    return bool(closed and (not on if previous_contact and not contact else on))


def align_disk_center(pose,closed,local_center):
    result=pose.copy()
    result[:3,3]=closed[:3,3]+closed[:3,:3]@local_center-pose[:3,:3]@local_center
    return result


def install_loading_scene(scene,old_object,table,xyz):
    from cross_episode_sim.fixtures.blender_lid import install_blender
    original=ET.parse(scene).find(f"worldbody/body[@name='{old_object}']")
    food=copy.deepcopy(original)
    lid,position=install_blender(scene,old_object,table,xyz)
    tree=ET.parse(scene);root=tree.getroot();world=root.find('worldbody')
    blender=world.find(f"body[@name='{BLENDER}']")
    lid_body=world.find(f"body[@name='{LID}']")
    center=np.fromstring(blender.get('pos'),sep=' ')
    # Face the entire native appliance toward the robot, including its button.
    blender.set('quat','0 0 0 1')
    lid_body.set('quat','0 0 0 1')
    position=center+Rotation.from_euler('z',np.pi).apply(position-center)
    lid_body.set('pos',' '.join(map(str,position)))
    native=ET.ElementTree(ET.fromstring(read_localized(GENERATED)))
    joint=native.find(f".//body[@name='{BUTTON_BODY}']/joint")
    blender.find(f".//body[@name='{BUTTON_BODY}']").insert(0,copy.deepcopy(joint))
    food_pos=np.asarray(xyz)+[.32,0.,0.]
    food.set('pos',' '.join(map(str,food_pos)));world.append(food)
    tree.write(scene)
    return lid,position,dict(body=old_object,asset='Egg_14',grasp_path=str(OUTPUT/'Egg_14/grasps.npz'),position=food_pos.tolist())


class BlenderLoadTest(BlenderLidTest):
    video_filename='blender_load_close_power.mp4'

    def __init__(self,args,selection):
        self.button_phase=False;self.turned_on=False;self.previous_button_contact=False
        self.power_events=[];self.force_history=[]
        super().__init__(args,selection)
        self.food=next(obj for obj in self.objects if obj!=LID)
        self.lid_release_clearance=.008
        self.food_metadata=json.loads((Path(args.recording)/'setup.json').read_text())
        self.lid_metadata=self.asset_metadata.copy()
        meta=json.loads(GENERATED.with_suffix('.json').read_text())
        self.anchor=self.data.body(BLENDER).xpos+self.data.body(BLENDER).xmat.reshape(3,3)@meta['anchor_offset']
        self.dock=self.anchor[:2]+[0.,.44]
        self.button_gid=self.model.geom(BUTTON).id
        self.report.update(task='open blender, load object, close lid, press power',
                           tracked_objects=[LID,self.food],power_events=self.power_events,
                           gripper_force_history=self.force_history,fixture_update_hz=20,
                           initialization='closed native lid, object on worktop, power off; no live resets')

    def object_label(self):return 'blender lid' if self.object_name==LID else 'Egg_14'

    def state(self):
        body=self.data.body(LID);position=body.xpos;rotation=body.xmat.reshape(3,3)
        return dict(lid_on_blender=lid_closed_state(position,rotation,self.anchor),
                    anchor_error_m=float(np.linalg.norm(position-self.anchor)),
                    gripper_distance_m=float(np.linalg.norm(self.tcp()[:3,3]-position)),
                    upright_degrees=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True).tolist())

    def align_lid_center(self,pose):
        center=self.model.geom_pos[self.model.geom('skill_blender_lid_g1').id]
        return align_disk_center(pose,self.closed_pose,center)

    def button_contact(self):
        touched=False;force=0.
        for i,c in enumerate(self.data.contact):
            if self.button_gid not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.button_gid else c.geom1
            if not self.model.body(self.model.geom_bodyid[other]).name.startswith('robot_0/gripper/'):continue
            touched=True;f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);force+=max(0.,float(f[0]))
        return touched,force

    def before_step(self):
        super().before_step()
        if not hasattr(self,'button_gid'):return
        # Kitchen._post_action calls fixture.update_state at control_freq=20,
        # not at every 500 Hz physics substep.
        if self.data.time+1e-9<getattr(self,'next_fixture_update',0.):return
        self.next_fixture_update=float(self.data.time)+.05
        contact,force=self.button_contact()
        on=next_power_state(self.turned_on,self.previous_button_contact,contact,self.state()['lid_on_blender'])
        if on!=self.turned_on:
            event=dict(time=float(self.data.time),turned_on=on,button_contact=contact)
            self.power_events.append(event);self.record(blender_power_transition=event)
        self.turned_on=on;self.previous_button_contact=contact

    def annotation_approach_allowed(self,pose):
        if self.object_name==LID:return super().annotation_approach_allowed(pose)
        return CrossRoomManipulation.annotation_approach_allowed(self,pose)

    def select_target(self,obj):
        self.select_object(obj)
        self.asset_metadata=self.lid_metadata.copy() if obj==LID else self.food_metadata.copy()
        self.args.annotation_source='geometry_hypotheses' if obj==LID else 'droid'
        self.args.approach_policy='any-above-table'
        limit=10. if obj==LID else 2.
        self.model.actuator_forcerange[self.model.actuator('robot_0/'+self.profile.gripper_actuator).id]=[-limit,limit]
        self.grasp_force_limit_n=limit
        self.args.grip_force=limit
        self.report['force_profile']['actuator_limit_n']=limit
        self.force_history.append(dict(time=float(self.data.time),limit_n=limit,target=obj))

    def prepare_pickup(self):
        if self.object_name==LID:return super().prepare_pickup()
        self.tuck_for_navigation()
        self.task_navigate(self.dock,False,face=-np.pi/2)
        self.embodiment.open_gripper(self);self.tick(.5)
        self.untuck_for_manipulation();self.active_family='any'
        self.physically_rejected_annotation_variants=set()

    def transport_payload(self):
        if self.object_name==LID:return super().transport_payload()
        self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.support_bids=self.table_bids[BLENDER]

    def food_inside(self):
        gid=self.model.geom('skill_blender_reg_int').id
        position=self.data.body(self.food).xpos
        local=self.data.geom_xmat[gid].reshape(3,3).T@(position-self.data.geom_xpos[gid])
        radius,half=self.model.geom_size[gid,:2]
        # The native placement region is conservative and shallow. Require the
        # actual center in the pitcher footprint and physical blender support.
        return bool(np.linalg.norm(local[:2])<radius and
                    self.data.body(BLENDER).xpos[2]<position[2]<self.anchor[2]-.01 and
                    self.assignment([self.food])[self.food]==BLENDER)

    def place_payload(self):
        if self.object_name==LID:return super().place_payload()
        self.review_phase='PLACE OBJECT INTO BLENDER'
        held=self.bread_pose();local=(self.bread_vertices()-held[:3,3])@held[:3,:3]
        # Release above the opening; fingers remain outside the narrow pitcher.
        rejected=[]
        center=self.data.geom_xpos[self.model.geom('skill_blender_reg_int').id,:2]
        q=[float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names]
        for tilt,yaw in ((15.,0.),(30.,0.),(0.,0.),(15.,25.),(15.,-25.),(30.,25.),(30.,-25.)):
            tool_R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@Rotation.from_euler('x',tilt,degrees=True).as_matrix()@np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
            obj=held.copy();obj[:3,:3]=tool_R@self.grasp_relative[:3,:3];obj[:2,3]=center
            bottom=float((local@obj[:3,:3].T)[:,2].min())
            obj[2,3]=self.anchor[2]+.025-bottom
            target=obj@np.linalg.inv(self.grasp_relative)
            try:
                goal=list(target[:3,3]-[0,0,self.embodiment.planner_tool_offset()])+list(Rotation.from_matrix(target[:3,:3]).as_quat(scalar_first=True))
                trajectory=self.planner.plan(q,goal)
                self.preplanned_moves['above open blender with object']=(target,trajectory)
                break
            except RuntimeError as exc:rejected.append(str(exc))
        else:raise RuntimeError(f'No reachable blender release orientation: {rejected}')
        self.record(blender_release_tilt_deg=tilt,blender_release_yaw_deg=yaw,rejected_release_poses=rejected)
        self.move('above open blender with object',target)
        vertices=self.bread_vertices();center=self.data.geom_xpos[self.model.geom('skill_blender_reg_int').id,:2]
        if np.max(np.linalg.norm(vertices[:,:2]-center,axis=1))>.06:
            raise RuntimeError('Payload footprint does not fit blender opening')
        self.holding_loaf=False;self.embodiment.open_gripper(self);self.tick(1.5)
        self.planner.detach_block();self.attached=False
        if not self.food_inside():raise RuntimeError('Released object is not supported inside blender')
        self.report['atomic_results'].append(dict(skill='place_in_blender',success=True,object=self.food))
        retreat=self.tcp().copy();retreat[:3,3]-=.05*retreat[:3,2]
        self.mesh_contact_move('withdraw after loading blender',retreat)
        self.tuck_for_navigation()

    def kitchen_world_geoms(self):
        geoms=super().kitchen_world_geoms()
        return [g for g in geoms if not (self.button_phase and g==self.button_gid)]

    def navigation_penetration(self,data,carrying):
        if not self.button_phase:return super().navigation_penetration(data,carrying)
        worst=0.
        for c in data.contact:
            b1,b2=self.model.geom_bodyid[[c.geom1,c.geom2]]
            n1,n2=self.model.body(b1).name or '',self.model.body(b2).name or ''
            r1,r2=n1.startswith('robot_0/'),n2.startswith('robot_0/')
            if n1.startswith('floor_') or n2.startswith('floor_'):continue
            if self.button_gid in (c.geom1,c.geom2) and (n1.startswith('robot_0/gripper/') or n2.startswith('robot_0/gripper/')):continue
            if r1!=r2:worst=max(worst,-float(c.dist)+.001)
        return worst

    def press_power(self):
        self.review_phase='PRESS BLENDER POWER';self.button_phase=True
        self.tuck_for_navigation();self.task_navigate(self.dock,False,face=-np.pi/2)
        self.embodiment.command_gripper(self.data,self.profile.gripper_close);self.tick(.6)
        self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world()
        from cross_episode_sim.controller.base import geom_box
        R=Rotation.from_euler('x',55,degrees=True).as_matrix()@np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        points=[]
        for gid in range(self.model.ngeom):
            if self.model.body(self.model.geom_bodyid[gid]).name!='robot_0/gripper/left_pad' or not self.model.geom_contype[gid]:continue
            c,h=geom_box(self.model,gid);points.extend(self.data.geom_xpos[gid]+(c+corners*h)@self.data.geom_xmat[gid].reshape(3,3).T)
        tcp=self.tcp();offsets=(np.asarray(points)-tcp[:3,3])@tcp[:3,:3]@R.T
        tip=offsets[offsets[:,1]<offsets[:,1].min()+1e-5].mean(0)
        normal=self.data.geom_xmat[self.button_gid].reshape(3,3)[:,2].copy()
        if normal[1]<0:normal=-normal
        surface=self.data.geom_xpos[self.button_gid]+normal*self.model.geom_size[self.button_gid,1]
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=surface-tip
        pre=pose.copy();pre[:3,3]+=.06*normal
        self.move('approach blender power button',pre)
        for gap in (.02,.006):
            target=pose.copy();target[:3,3]+=gap*normal
            self.mesh_contact_move('press blender power button',target);self.tick(.15)
        for attempt in range(3):
            target=pose.copy();target[:3,3]-=.0015*normal
            self.smooth_button_move('press blender power button',target);self.tick(.2)
            contact,force=self.button_contact();self.record(blender_button_contact=contact,pad_force_n=force,
                button_stroke_m=float(self.data.joint('skill_blender_power_button_joint').qpos[0]),press_attempt=attempt+1)
            if not contact or force<=.05:raise RuntimeError('No force-bearing blender button press')
            self.smooth_button_move('release blender power button',pre);self.tick(.5)
            if not self.button_contact()[0] and self.turned_on:break
        else:raise RuntimeError('Blender power did not latch on after three physical presses')
        self.report['atomic_results'].append(dict(skill='press_blender_power',success=True,force_n=force,attempts=attempt+1))
        self.button_phase=False;self.tuck_for_navigation();self.tick(.5)

    def smooth_button_move(self,stage,pose):
        path=self.plan_contact_path(stage,pose)
        self.stage=stage
        addresses=[self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id] for n in self.planner.names]
        previous=self.data.qpos[addresses].copy()
        bias=np.clip(self.data.ctrl[self.arm_aids]-previous,-.08,.08)
        for q in path:
            q=np.asarray(q)
            # Continuous small setpoint changes avoid repeated contact release
            # at the boundaries of the general 320 ms contact-move waypoints.
            for fraction in np.linspace(0.,1.,11)[1:]:
                self.data.ctrl[self.arm_aids]=(1-fraction)*previous+fraction*q+bias
                self.tick(.02)
            bias=np.clip(bias+.25*(q-self.data.qpos[addresses]),-.08,.08)
            previous=q
        self.tick(.2)
        error=float(np.linalg.norm(self.tcp()[:3,3]-pose[:3,3]))
        if error>.003:raise RuntimeError(f'Button motion tracking error: {error}')
        self.record(button_motion='mesh-checked continuous joint interpolation',tcp_error_m=error)

    def render_video_frame(self,label,time):
        on=False
        for event in self.power_events:
            if event['time']<=time:on=event['turned_on']
        limit=10.
        for event in self.force_history:
            if event['time']<=time:limit=event['limit_n']
        aid=self.model.actuator('robot_0/'+self.profile.gripper_actuator).id
        saved=self.model.actuator_forcerange[aid].copy()
        try:
            self.model.actuator_forcerange[aid]=[-limit,limit]
            return super().render_video_frame(label+' | blender '+('ON' if on else 'OFF'),time)
        finally:self.model.actuator_forcerange[aid]=saved

    def run_test(self):
        success=False
        try:
            self.tick(1.);self.closed_pose=self.bread_pose().copy()
            if not self.state()['lid_on_blender']:raise RuntimeError('Blender must start closed')
            diagnostic=getattr(self.args,'blender_power_only',False)
            sequence=() if diagnostic else ((LID,BLENDER,self.counter),(self.food,self.counter,BLENDER),(LID,self.counter,BLENDER))
            for obj,source,dest in sequence:
                self.select_target(obj);self.source,self.destination=source,dest
                self.review_phase=('OPEN BLENDER LID' if source==BLENDER else
                                   'PICK OBJECT FOR BLENDER' if obj==self.food else 'CLOSE BLENDER LID')
                self.support_bids=self.table_bids[source]
                self.table_gids={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.support_bids and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
                self.initial_object_pose=self.bread_pose().copy();self.transfer_start=self.data.joint(self.object_joint).qpos.copy()
                self.preplanned_moves={};self._recovery_exhausted=False
                self.execute_transfer()
            self.press_power()
            success=self.turned_on and self.state()['lid_on_blender'] and (diagnostic or self.food_inside()) and self.in_default_travel_posture(False)
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        finally:
            self.report.update(success=success,final_state=self.state(),final_turned_on=self.turned_on,
                               food_inside=self.food_inside(),component_test=getattr(self.args,'blender_power_only',False),
                               scope='power button component only' if getattr(self.args,'blender_power_only',False) else 'native lid and physical object placement, contact/release power latch; no liquefaction')
            self.finish_run_outputs(success)
        return 0 if success else 1
