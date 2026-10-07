"""Continuous Franka/TidyBot counter dosing, cup placement and machine workflow.

Free portafilter and box are moved through physical gripper contacts. Two passive
connect constraints model the seated bayonet axis; they disengage only after a
measured unlock twist. Reinsertion enables them only at measured socket alignment.
A final seated weld models the locked bayonet, never a robot grasp. Grounds are
free rigid granules. Rubber pad contacts include torsional friction; five MuJoCo
friction postprocessing iterations limit numerical creep. Passive buttons move
under finger contact. Brewing uses an explicit timed state model, without fluid
or thermal physics. No live object/particle poses are reset.
"""
import argparse
import json
from pathlib import Path
import random
import hashlib
import sys
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.paths import COFFEE_DIR, DATA_DIR
from cross_episode_sim.tasks.coffee.pour import MoonlakeRobotPour, prepare as prepare_pour
from cross_episode_sim.tasks.coffee.machine import PORTAFILTER, CENTER, RIM, BOTTOM, INNER_RADIUS, numbers
from cross_episode_sim.tasks.coffee.native_scene import COFFEE
from cross_episode_sim.tasks.coffee.placement import LID_PARKING, SLOTS, VALIDATED_MOUNT, offset, sample_offset, shift
from cross_episode_sim.tasks.coffee.brew import BrewActions, prepare_brew, POWER_BUTTON
from cross_episode_sim.tasks.coffee.grounds import PARTICLE_RADIUS, grain_inventory
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies

BUTTON = 'button_01_press_pivot_001'


def body_pose(data, name):
    b=data.body(name); t=np.eye(4); t[:3,:3]=b.xmat.reshape(3,3); t[:3,3]=b.xpos
    return t


def prepare(run, asset, out, machine_offset=(0., 0.)):
    manifest, selection=prepare_pour(run,asset,out,machine_offset)
    off=offset(manifest['coffee'])
    root=ET.parse(out/'robocasa_scene.xml').getroot(); world=root.find('worldbody')
    # Dense Jacobians reduce factorization cost for this contact island while
    # retaining the original Newton solver, timestep and convergence settings.
    # Friction postprocessing limits soft-constraint creep under handle torque.
    root.findall('option')[-1].set('solver','Newton')
    root.findall('option')[-1].set('jacobian','dense')
    root.findall('option')[-1].set('noslip_iterations','5')
    basket=root.find(f".//body[@name='{PORTAFILTER}']")
    seated=np.array(manifest['coffee']['portafilter_original_pose']); seated[:3,3]+=shift(VALIDATED_MOUNT,off)
    basket.set('pos',numbers(seated[:3,3]));basket.set('quat',numbers(Rotation.from_matrix(seated[:3,:3]).as_quat(scalar_first=True)))
    ET.SubElement(basket,'freejoint',name=PORTAFILTER+'_free')
    eq=root.find('equality')
    if eq is None: eq=ET.SubElement(root,'equality')
    for name,anchor in [('bayonet_axis_a','0 0 0'),('bayonet_axis_b','0 0 .04')]:
        ET.SubElement(eq,'connect',name=name,body1=PORTAFILTER,body2=COFFEE,anchor=anchor,solref='.004 1',solimp='.99 .999 .001')
    ET.SubElement(eq,'weld',name='bayonet_locked',body1=PORTAFILTER,body2=COFFEE,active='false',solref='.004 1',solimp='.99 .999 .001')
    button=root.find(f".//body[@name='{BUTTON}']")
    ET.SubElement(button,'joint',name='button_01_press_pivot',type='slide',axis='0 0 1',range='0 .0015',stiffness='180',springref='0',damping='.5',armature='.0001')
    # Adjacent articulated parts overlap inside the source housing. Its slide
    # joint supplies button guidance and travel stops; exclude only this internal
    # pair. Finger/housing and finger/button collision geometries remain active.
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    ET.SubElement(contact,'exclude',body1=BUTTON,body2='base_frame_002')
    # The native portafilter rests directly on the kitchen counter.
    holder=root.find(".//body[@name='moonlake_portafilter_holder']")
    world.remove(holder)
    boxname=manifest['coffee']['dosing_body']; box=root.find(f".//body[@name='{boxname}']")
    oldpos=np.fromstring(box.get('pos'),sep=' ');oldR=Rotation.from_quat(np.fromstring(box.get('quat'),sep=' '),scalar_first=True).as_matrix()
    boxpos=shift([2.39,-.53,.960],off);boxR=Rotation.from_euler('z',-139.2678933,degrees=True).as_matrix()
    box.set('pos',numbers(boxpos));box.set('quat',numbers(Rotation.from_matrix(boxR).as_quat(scalar_first=True)))
    for name in manifest['coffee']['grains']:
        g=root.find(f".//body[@name='{name}']"); pos=np.fromstring(g.get('pos'),sep=' ')
        g.set('pos',numbers(boxpos+boxR@oldR.T@(pos-oldpos)))
    # Park the legacy lid away from the current apparatus.
    lid=root.find(".//body[@name='skill_blender_lid_main']");lid.set('pos',numbers(LID_PARKING))
    local=np.eye(4);local[:3,:3]=Rotation.from_euler("y",65,degrees=True).as_matrix();local[:3,3]=[-.145,0,.040]
    annotations=out/'portafilter_grasps.npz';np.savez(annotations,transforms=np.array([local]))
    info=dict(body=PORTAFILTER,asset='MoonlakePortafilter',key='moonlake__portafilter',role='portafilter',
        source='coffee_machine',destination='kitchen',position=seated[:3,3].tolist(),
        grasp_path=str(annotations),setup=dict(source='Moonlake',asset='MoonlakePortafilter'),task_object=True)
    manifest['bindings'].append(info);selection['selected_objects']=manifest['bindings']
    manifest['coffee']['portafilter_seated_pose']=seated.tolist()
    manifest['coffee']['withdrawal_execute_stride_deg']=15
    manifest['coffee']['box_parking_pose']=np.block([[boxR,boxpos[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()
    manifest['instruction']='Remove portafilter, place on counter, pick up box and pour, set box down, reinstall portafilter, press button.'
    manifest['simulation_contract']=dict(robot='Actuator-only after initialized empty-hand kitchen dock',
        portafilter='Free body held by friction contacts; measured-twist bayonet coupling at machine; direct countertop support',
        grounds='32 free coarse rigid granules; no runtime pose resets',gripper='Finite rubber pad contact patches, condim=4 with existing torsional coefficient .005 m',button='Passive spring-return source slide joint; internal button/housing pair excluded; physical fingertip press',brewing='Not simulated',solver='Newton with dense Jacobian and 5 friction postprocessing iterations; original timestep, tolerance and primary iteration budget')
    conversion=json.loads((asset/'conversion.json').read_text())
    manifest['machine_provenance']={key:conversion[key] for key in ['source_url','source_sha256','license','socket_repair']}
    shutil.copyfile(asset/'LICENSE',out/'MOONLAKE_LICENSE')
    (out/'MOONLAKE_NOTICE').write_text((asset/'NOTICE.txt').read_text()+'\nWorkflow adaptations: free portafilter, measured-state bayonet coupling, direct countertop support, passive spring-return button, internal button/housing collision exclusion, finite rubber-pad torsional contacts, five friction postprocessing iterations. Robot contacts remain physical.\n')
    ET.ElementTree(root).write(out/'robocasa_scene.xml')
    model=mujoco.MjModel.from_xml_path(str(out/'robocasa_scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    counter=object_bodies(model,manifest['supports']['kitchen'])
    groups=model.geom_group.copy();model.geom_group[:]=5
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in counter and (model.geom_contype[gid] or model.geom_conaffinity[gid]):model.geom_group[gid]=4
    loading=np.array(manifest['coffee']['portafilter_reference_pose'])
    ray=np.r_[loading[:2,3],2.]
    distance=mujoco.mj_ray(model,data,ray,np.array([0.,0.,-1.]),np.array([0,0,0,0,1,0],dtype=np.uint8),True,-1,None)
    model.geom_group[:]=groups
    if distance<0:raise RuntimeError('No counter surface below portafilter placement')
    counter_z=float(ray[2]-distance)
    vertices=collision_vertices(model,data,object_bodies(model,PORTAFILTER))
    local=(vertices-data.body(PORTAFILTER).xpos)@data.body(PORTAFILTER).xmat.reshape(3,3)
    loading[2,3]=counter_z+.002-float((local@loading[:3,:3].T)[:,2].min())
    manifest['coffee']['portafilter_reference_pose']=loading.tolist()
    manifest['coffee']['portafilter_counter_z']=counter_z
    (out/'task_manifest.json').write_text(json.dumps(manifest,indent=2));(out/'adapter.json').write_text(json.dumps(selection,indent=2))
    return manifest,selection


class CoffeeWorkflow(BrewActions, MoonlakeRobotPour):
    video_filename='moonlake_full_workflow.mp4'

    def __init__(self,*args):
        self.workflow_events=[];self.fixture_phase='seated';self.button_pressed=False
        self.allow_support=False;self.active_button_body=BUTTON
        super().__init__(*args)
        # Rubber pads have a finite contact patch. Activate the already configured
        # torsional coefficient (0.005 m); 3D point contacts ignore it entirely.
        # Sliding coefficients, force limits, geometry and body mass stay native.
        pads=[]
        for gid in range(self.model.ngeom):
            name=self.model.body(int(self.model.geom_bodyid[gid])).name
            if name in (self.profile.namespace+'gripper/left_pad',self.profile.namespace+'gripper/right_pad') and self.model.geom_contype[gid]:
                self.model.geom_condim[gid]=4;pads.append(self.model.geom(gid).name)
        self.report['pad_contact_model']=dict(dimensions=4,torsional_friction_m=.005,geoms=pads)
        if 'mug_target_pose' in self.spec:self.initialize_machine()
        self.seated=np.array(self.spec['portafilter_seated_pose'])
        self.loading=np.array(self.spec['portafilter_reference_pose'])
        self.axis_eq=[self.model.equality(n).id for n in ['bayonet_axis_a','bayonet_axis_b']]
        self.lock_eq=self.model.equality('bayonet_locked').id
        passive={self.model.joint(PORTAFILTER+'_free').id,self.model.joint('button_01_press_pivot').id}
        if hasattr(self,'machine_cycle'):passive.add(self.model.joint(POWER_BUTTON.removesuffix('_001')).id)
        if any(self.model.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and self.model.actuator_trnid[a,0] in passive for a in range(self.model.nu)):
            raise RuntimeError('Portafilter and button must have no direct actuators')
        self.report.update(scope=__doc__,workflow_events=self.workflow_events,
            execution='Robot actuators only; passive measured-state bayonet coupling; no gripper welds or live object resets',
            full_task_requested=True,brewing_simulated='mug_target_pose' in self.spec,
            fixture_direct_actuators=0,particle_pose_resets_after_initialization=0,
            physics=dict(solver=int(self.model.opt.solver),jacobian=int(self.model.opt.jacobian),timestep_s=float(self.model.opt.timestep),
                         tolerance=float(self.model.opt.tolerance),iterations=int(self.model.opt.iterations),noslip_iterations=int(self.model.opt.noslip_iterations)))

    def event(self,name,**kw):
        e=dict(event=name,time=float(self.data.time),**kw);self.workflow_events.append(e);self.record(workflow_event=e)
        (self.output/'workflow_events.json').write_text(json.dumps(self.workflow_events,indent=2))
        self.machine_event(name)
        if name in ('portafilter_removed','empty_box_parked','portafilter_reinstalled','cup_placed','machine_ready'):
            np.savez(self.output/(name+'_checkpoint.npz'),qpos=self.data.qpos,qvel=self.data.qvel,
                     ctrl=self.data.ctrl,eq_active=self.data.eq_active,time=self.data.time,
                     machine_cycle=json.dumps(self.machine_cycle.__dict__) if hasattr(self,'machine_cycle') else '{}',
                     portafilter_pick_seed=np.array(getattr(self,'portafilter_pick_seed',[])),
                     removal_tcp=np.array(getattr(self,'removal_tcp',[])),removal_joints=np.array(getattr(self,'removal_joints',[])))

    def select_object(self,obj):
        super().select_object(obj)
        if obj==PORTAFILTER:self.args.annotation_source='geometry_hypotheses'

    def object_label(self):
        return 'portafilter' if getattr(self,'object_name',None)==PORTAFILTER else super().object_label()

    def interaction_label(self):
        return 'coffee machine button' if self.press_phase else self.object_label()

    def render_video_frame(self,label,time):
        if hasattr(self,'machine_cycle'):return self.render_machine_frame(label,time)
        return BreakfastEpisode.render_video_frame(self,label+' | robot coffee workflow; simulated bayonet',time)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        self.cameras[1].lookat[:]=shift([2.12,-.40,1.08],offset(self.spec))
        self.cameras[1].distance=.80
        self.cameras[1].azimuth=90.
        self.cameras[1].elevation=-35.
        phase=getattr(self,'review_phase','')
        if 'BUTTON' in phase:
            self.cameras[1].lookat[:]=shift([2.12,-.30,1.19],offset(self.spec))
            self.cameras[1].distance=.60;self.cameras[1].azimuth=135.;self.cameras[1].elevation=-15.
        elif 'CUP' in phase:
            self.cameras[1].lookat[:]=self.data.body('cup_one_test_object_main').xpos+[0,0,.04]
            self.cameras[1].distance=.75;self.cameras[1].azimuth=45.;self.cameras[1].elevation=-30.
        elif 'BREW' in phase:
            self.cameras[1].lookat[:]=shift([2.12,-.30,1.07],offset(self.spec))
            self.cameras[1].distance=.60;self.cameras[1].azimuth=130.;self.cameras[1].elevation=-35.

    def inventory_for(self,data):
        inv=grain_inventory(self.model,data,self.spec,getattr(self,'grain_body_ids',None))
        p=data.xpos[self.grain_body_ids]; t=body_pose(data,PORTAFILTER)
        source=np.array(self.spec['portafilter_original_pose'])
        original=(p-t[:3,3])@t[:3,:3]@source[:3,:3].T+source[:3,3]
        inside=np.linalg.norm(original[:,:2]-CENTER,axis=1)<=INNER_RADIUS-PARTICLE_RADIUS+.001
        inside&=original[:,2]-PARTICLE_RADIUS>=BOTTOM-.002
        inside&=original[:,2]+PARTICLE_RADIUS<=RIM+.001
        result=dict(hopper=[],vessel=[],spilled=[])
        for name,ok in zip(self.spec['grains'],inside):result['hopper' if ok else 'vessel' if name in inv['vessel'] else 'spilled'].append(name)
        return result

    def inventory(self):return self.inventory_for(self.data)

    def active_grains(self):
        if not hasattr(self,'grain_ids') or not getattr(self,'holding_loaf',False):return set()
        key='hopper' if self.object_name==PORTAFILTER else 'vessel' if self.object_name==self.spec['dosing_body'] else None
        return set() if key is None else {self.model.body(n).id for n in self.inventory()[key]}

    def project_payload_contents(self,probe,reference=None):
        live=self.data if reference is None else reference
        if probe is live:return
        key='hopper' if self.object_name==PORTAFILTER else 'vessel' if self.object_name==self.spec['dosing_body'] else None
        if key is None:return
        old=body_pose(live,self.object_name);new=body_pose(probe,self.object_name);r=new[:3,:3]@old[:3,:3].T
        for name in self.inventory_for(live)[key]:
            b=live.body(name);q=probe.joint(name+'_joint').qpos
            q[:3]=new[:3,3]+r@(b.xpos-old[:3,3]);q[3:]=Rotation.from_matrix(r@b.xmat.reshape(3,3)).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model,probe)

    def navigation_penetration(self,data,carrying):
        if not hasattr(self,'grain_ids'):return super().navigation_penetration(data,carrying)
        if carrying and data is not self.data:self.project_payload_contents(data)
        if self.press_phase and data is not self.data:
            # Predict passive button displacement in scratch collision data.
            # The live button is moved only by contact forces in mj_step.
            depths=[]
            for c in data.contact:
                a,b=map(int,self.model.geom_bodyid[[c.geom1,c.geom2]])
                na,nb=self.model.body(a).name,self.model.body(b).name
                if ((na==self.active_button_body and self.is_finger(nb)) or (nb==self.active_button_body and self.is_finger(na))):depths.append(-float(c.dist))
            if depths and max(depths)>0.:
                joint=data.joint(self.active_button_body.removesuffix('_001'))
                joint.qpos[0]=min(.0015,float(joint.qpos[0])+max(depths)+.00001)
                mujoco.mj_forward(self.model,data)
        payload=self.bread_bids|self.active_grains();worst=0.
        for c in data.contact:
            a,b=map(int,self.model.geom_bodyid[[c.geom1,c.geom2]])
            na,nb=self.model.body(a).name,self.model.body(b).name
            ra,rb=na.startswith('robot_0/'),nb.startswith('robot_0/')
            # Granules physically collide while flowing and settling. These
            # intended contents contacts are not arm-path obstructions.
            if a in self.grain_ids and b in self.grain_ids:continue
            if self.pouring and ((a in self.grain_ids and nb==PORTAFILTER) or (b in self.grain_ids and na==PORTAFILTER)):continue
            if na.startswith('floor_') or nb.startswith('floor_') or ra and rb:continue
            if (a in self.bread_bids and rb and self.is_finger(nb)) or (b in self.bread_bids and ra and self.is_finger(na)):continue
            if self.press_phase and ((ra and b==self.model.body(self.active_button_body).id and self.is_finger(na)) or (rb and a==self.model.body(self.active_button_body).id and self.is_finger(nb))):continue
            if carrying and ((a in payload and rb and self.embodiment.allows_payload_contact(nb)) or (b in payload and ra and self.embodiment.allows_payload_contact(na))):continue
            if self.allow_support and ((a in payload and b in (self.table_bids[self.counter]|getattr(self,'extra_support_bids',set()))) or (b in payload and a in (self.table_bids[self.counter]|getattr(self,'extra_support_bids',set())))):
                if c.dist>=-.001:continue
            if ra != rb or carrying and ((a in payload)!=(b in payload)):
                worst=max(worst,.001-float(c.dist))
        return worst

    def before_step(self):
        BreakfastEpisode.before_step(self)
        if not hasattr(self,'axis_eq'):return
        if self.press_phase:
            touch,force=self.button_contact();travel=float(self.data.joint(self.active_button_body.removesuffix('_001')).qpos[0])
            depth=max([0.]+[-float(c.dist) for c in self.data.contact if c.geom1 in self.button_ids or c.geom2 in self.button_ids])
            self.report['max_button_contact_penetration_m']=max(depth,self.report.get('max_button_contact_penetration_m',0.))
            if depth>.0015:raise RuntimeError('Excessive button contact penetration')
            self.report['max_button_travel_m']=max(travel,self.report.get('max_button_travel_m',0.))
            if touch and force>.05 and travel>.0006 and not self.button_pressed:
                if not self.button_prerequisites():raise RuntimeError('Machine button prerequisites not met')
                self.button_pressed=True;self.event('button_pressed',button=self.active_button_body,force_n=force,travel_m=travel)

        self.machine_before_step()

    def allowed_panel_contact(self,robot,other):
        return bool(self.press_phase and other==self.model.body(self.active_button_body).id and self.is_finger(robot))

    def q(self):return np.array([float(self.data.joint(self.profile.namespace+n).qpos[0]) for n in self.planner.names])

    def remember_removal(self,trace):
        rows=[row for row in trace if row.get('active_object')==PORTAFILTER and
              (row['stage'].startswith('grasp') or row['stage'] in ('twist portafilter to unlock','lower unlocked portafilter',
               'withdraw portafilter from machine','position portafilter above counter','lower portafilter onto counter'))]
        rows=rows[::5]+rows[-1:]
        addresses=[self.model.jnt_qposadr[self.model.joint(self.profile.namespace+n).id] for n in self.profile.arm_joints]
        self.removal_tcp=np.array([row['tcp'] for row in rows])
        self.removal_joints=np.array([np.asarray(row['qpos'])[addresses] for row in rows])

    def nearby_ik(self,pose,positions,**kwargs):
        if (getattr(self,'review_phase','')=='REINSTALL LOADED PORTAFILTER' and
                getattr(self,'holding_loaf',False) and self.object_name==PORTAFILTER and len(getattr(self,'removal_tcp',[]))):
            # Preserve the demonstrated redundancy choice along the whole path;
            # a single endpoint seed can drift to an elbow posture in the counter.
            index=int(np.argmin(np.linalg.norm(self.removal_tcp[:,:3,3]-pose[:3,3],axis=1)))
            positions=self.removal_joints[index]
            kwargs['trust_radius']=.7
        return super().nearby_ik(pose,positions,**kwargs)

    def mesh_contact_move(self,stage,pose,path=None):
        if (getattr(self,'review_phase','')=='REINSTALL LOADED PORTAFILTER' and getattr(self,'holding_loaf',False)):
            if path is None:path=self.plan_contact_path(stage,pose)
            self.check_loaded_tuck_path(path)
        return super().mesh_contact_move(stage,pose,path)

    def rebuild(self):
        if self.holding_loaf:self.rebuild_loaded_planner()
        else:
            self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names);self.load_world();self.preplanned_moves={}

    def arm(self,stage,target,global_plan=False):
        self.stage=stage
        if global_plan:
            try:return self.move(stage,target)
            except RuntimeError as e:
                self.record(global_approach_rejection=str(e))
            for seed in [self.q(),np.array([0.,-.3,0.,-1.6,0.,1.8,0.]),np.array([0.,-.3,0.,-1.6,0.,1.8,2.5])]:
                try:
                    endpoint=self.nearby_ik(target,seed,trust_radius=6.,preserve_self_clearance=True)
                    path=self.planner.plan_joints(self.q(),endpoint)
                    self.preplanned_moves[stage]=(target,path)
                    return self.move(stage,target)
                except RuntimeError as e:
                    self.record(joint_approach_rejection=str(e))
        return self.mesh_contact_move(stage,target)

    def held_pose(self,stage,target):
        for correction in range(4):
            self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
            self.arm(stage if not correction else stage+' pose correction',target@np.linalg.inv(self.grasp_relative))
            if len(self.contacts())<2:raise RuntimeError('Lost bilateral grasp at '+stage)
            actual=self.bread_pose();position_error=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
            angle_error=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()))
            self.report.setdefault('payload_pose_checks',[]).append(dict(stage=stage,correction=correction,position_error_m=position_error,orientation_error_deg=angle_error))
            tight=self.object_name=='cup_one_test_object_main'
            if position_error<=(.0005 if tight else .002) and angle_error<=(.1 if tight else 1.):return
            self.record(payload_pose_correction=correction+1,position_error_m=position_error,orientation_error_deg=angle_error)
        raise RuntimeError(f'Payload failed to follow gripper at {stage}: {position_error:.4f} m, {angle_error:.2f} degrees')

    def pickup(self,obj,local):
        self.select_object(obj);self.source=self.counter;self.destination=self.counter
        self.support_bids=(self.table_bids[self.counter]|getattr(self,'extra_support_bids',set()));self.table_gids={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.support_bids}
        self.initial_object_pose=self.bread_pose().copy();self.pickup_start_height=float(self.initial_object_pose[2,3]);self.transfer_start=self.data.joint(self.object_joint).qpos.copy()
        self.allow_support=True;self.embodiment.open_gripper(self);self.tick(.4);self.rebuild()
        target=self.bread_pose()@local;pre=target.copy();pre[:3,3]-=.04*pre[:3,2]
        if obj==PORTAFILTER and self.fixture_phase=='released' and hasattr(self,'portafilter_pick_seed'):
            endpoint=self.nearby_ik(pre,self.portafilter_pick_seed,trust_radius=3.,preserve_self_clearance=True)
            path=self.planner.plan_joints(self.q(),endpoint)
            self.preplanned_moves['approach '+self.object_label()]=(pre,path)
            self.record(regrasp_seed='successful portafilter removal posture',joint_seed=list(self.portafilter_pick_seed))
        self.arm('approach '+self.object_label(),pre,True)
        self.arm('grasp '+self.object_label(),target)
        if obj==PORTAFILTER:
            solver=self.model.opt.solver;self.model.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON
            aid=self.model.actuator(self.profile.namespace+self.profile.gripper_actuator).id
            low,high=self.model.actuator_ctrlrange[aid];self._force_trace_time=-1.
            try:
                for command in np.linspace(low,high,121)[1:]:
                    self.data.ctrl[aid]=command;self.tick(.025)
                self.tick(.3)
                self.holding_loaf=True
                self._drop_reference=(np.linalg.inv(self.tcp())@self.bread_pose())[:3,3].copy();self._drop_seconds=0.
            finally:self.model.opt.solver=solver
        else:
            solver=self.model.opt.solver;self.model.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON
            try:self.close_on_loaf();self.tick(.3)
            finally:self.model.opt.solver=solver
        if len(self.contacts())<2:raise RuntimeError('No bilateral grasp on '+obj)
        self.attached=True;self._pickup_cleared=True;self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.event('grasp',body=obj,contacts=self.contacts(),object_T_tcp=(np.linalg.inv(self.bread_pose())@self.tcp()).tolist())
        self.rebuild()

    def release(self,label,retreat_delta=None):
        self.holding_loaf=False;self.attached=False
        if self.object_name==PORTAFILTER and self.fixture_phase=='released':
            # A snap-open linkage kicks the low handle and overturns the basket.
            # Unload the native fingers gradually while the counter takes weight.
            aid=self.model.actuator(self.profile.namespace+self.profile.gripper_actuator).id
            for command in np.linspace(float(self.data.ctrl[aid]),self.profile.gripper_open,81)[1:]:
                self.data.ctrl[aid]=command;self.tick(.02)
            self.tick(.3)
        else:self.embodiment.open_gripper(self);self.tick(.7)
        self.event('release',body=self.object_name,pose=self.bread_pose().tolist(),counts={k:len(v) for k,v in self.inventory().items()})
        self.rebuild()
        target=self.tcp().copy()
        target[:3,3]+=(-.09*target[:3,2] if retreat_delta is None else np.asarray(retreat_delta))
        self.arm(label,target);self.allow_support=False

    def pf_grasp(self):
        local=np.eye(4);local[:3,:3]=Rotation.from_euler('y',65,degrees=True).as_matrix();local[:3,3]=[-.145,0,.040];return local

    def unlock_remove(self):
        self.review_phase='REMOVE PORTAFILTER'
        self.pickup(PORTAFILTER,self.pf_grasp())
        for angle in range(5,51,5):
            target=self.seated.copy();target[:3,:3]=Rotation.from_euler('z',-angle,degrees=True).as_matrix()@self.seated[:3,:3]
            self.held_pose('twist portafilter to unlock',target)
        actual=body_pose(self.data,PORTAFILTER)
        error=Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()
        if error>np.deg2rad(3):raise RuntimeError('Portafilter did not physically unlock')
        self.data.eq_active[self.axis_eq]=False;self.fixture_phase='released';self.event('bayonet_unlocked',orientation_error_deg=float(np.rad2deg(error)))
        down=actual.copy()
        for amount in (.02,.04,.06,.08):
            p=down.copy();p[2,3]-=amount;self.held_pose('lower unlocked portafilter',p)
        front=p.copy();front[1,3]=self.loading[1,3]
        self.held_pose('withdraw portafilter from machine',front)
        hover=self.loading.copy();hover[2,3]+=.012;self.held_pose('position portafilter above counter',hover)
        self.held_pose('lower portafilter onto counter',self.loading)
        self.portafilter_pick_seed=self.q().tolist()
        self.remember_removal(self.trace)
        self.release('withdraw empty hand from portafilter')
        self.tick(.8)
        if np.linalg.norm(body_pose(self.data,PORTAFILTER)[:3,3]-self.loading[:3,3])>.015:raise RuntimeError('Portafilter unstable on counter')
        own=object_bodies(self.model,PORTAFILTER);counter=self.table_bids[self.counter]
        supported=any((self.model.geom_bodyid[c.geom1] in own and self.model.geom_bodyid[c.geom2] in counter) or
                      (self.model.geom_bodyid[c.geom2] in own and self.model.geom_bodyid[c.geom1] in counter) for c in self.data.contact)
        tilt=Rotation.from_matrix(body_pose(self.data,PORTAFILTER)[:3,:3]@self.loading[:3,:3].T).magnitude()
        speed=float(np.linalg.norm(self.data.joint(PORTAFILTER+'_free').qvel))
        if not supported or tilt>np.deg2rad(5) or speed>.01:raise RuntimeError('Portafilter lacks stable upright counter support')
        self.event('portafilter_removed',pose=body_pose(self.data,PORTAFILTER).tolist(),counter_contact=supported,speed=speed)

    def dose(self):
        self.review_phase='LOAD PORTAFILTER FROM SMALL BOX'
        local=np.array(self.manifest['box_grasp'])
        self.pickup(self.spec['dosing_body'],local)
        lift=self.bread_pose().copy();lift[2,3]+=.10;self.held_pose('lift grounds box',lift)
        self.allow_support=False
        source=np.array(self.spec['portafilter_original_pose']);pf=body_pose(self.data,PORTAFILTER)
        center=pf@np.linalg.inv(source)@np.array([*CENTER,RIM,1.])
        self.spec['hopper'].update(center_xy=center[:2].tolist(),top_z=float(center[2]))
        self.pour_grounds()
        self.event('grounds_loaded',counts={k:len(v) for k,v in self.inventory().items()})
        target=np.array(self.spec['box_parking_pose']);hover=target.copy();hover[2,3]+=.10
        self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.arm('return empty box above counter',hover@np.linalg.inv(self.grasp_relative),True)
        self.allow_support=True
        near=target.copy();near[2,3]+=.003;self.held_pose('set empty box on counter',near)
        self.release('withdraw from empty box')
        self.event('empty_box_parked')

    def installed(self):
        t=body_pose(self.data,PORTAFILTER)
        return bool(self.data.eq_active[self.lock_eq] and np.linalg.norm(t[:3,3]-self.seated[:3,3])<.003 and Rotation.from_matrix(t[:3,:3]@self.seated[:3,:3].T).magnitude()<np.deg2rad(2))

    def reinstall(self):
        self.review_phase='REINSTALL LOADED PORTAFILTER'
        self.pickup(PORTAFILTER,self.pf_grasp())
        lift=self.bread_pose().copy();lift[2,3]+=.025;self.held_pose('lift loaded portafilter',lift);self.allow_support=False
        target=self.seated.copy();target[:3,:3]=Rotation.from_euler('z',-50,degrees=True).as_matrix()@self.seated[:3,:3]
        low=target.copy();low[2,3]-=.08
        front=low.copy();front[1,3]=self.loading[1,3]
        self.held_pose('align loaded portafilter in front of machine',front)
        self.held_pose('insert loaded portafilter below socket',low)
        for amount in (.06,.04,.02,0.):
            p=target.copy();p[2,3]-=amount;self.held_pose('raise portafilter into socket',p)
        actual=body_pose(self.data,PORTAFILTER)
        if np.linalg.norm(actual[:3,3]-target[:3,3])>.002 or Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()>np.deg2rad(2):raise RuntimeError('Portafilter outside bayonet engagement tolerance')
        self.data.eq_active[self.axis_eq]=True;self.event('bayonet_engaged',position_error_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])))
        for angle in range(45,-1,-5):
            p=self.seated.copy();p[:3,:3]=Rotation.from_euler('z',-angle,degrees=True).as_matrix()@self.seated[:3,:3]
            self.held_pose('twist portafilter to lock',p)
        actual=body_pose(self.data,PORTAFILTER)
        if np.linalg.norm(actual[:3,3]-self.seated[:3,3])>.002 or Rotation.from_matrix(actual[:3,:3]@self.seated[:3,:3].T).magnitude()>np.deg2rad(2):raise RuntimeError('Portafilter did not reach locked pose')
        self.data.eq_active[self.lock_eq]=True;self.data.eq_active[self.axis_eq]=False;self.fixture_phase='locked'
        self.release('withdraw hand from installed portafilter');self.tick(.5)
        if not self.installed() or not self.grounds_ready():raise RuntimeError('Loaded portafilter installation failed')
        self.event('portafilter_reinstalled',counts={k:len(v) for k,v in self.inventory().items()})

    def button_prerequisites(self):
        if hasattr(self,'machine_cycle'):return self.machine_button_prerequisites()
        return self.installed() and self.grounds_ready()

    def press(self):
        action=getattr(self,'machine_action',None) or 'start'
        self.review_phase='PRESS MACHINE '+action.upper()+' BUTTON'
        self.embodiment.command_gripper(self.data,self.profile.gripper_close);self.tick(.5)
        self.press_phase=True;self.rebuild()
        pad=object_bodies(self.model,self.profile.namespace+'gripper/left_pad')
        pad_local=(collision_vertices(self.model,self.data,pad)-self.tcp()[:3,3])@self.tcp()[:3,:3]
        points=collision_vertices(self.model,self.data,object_bodies(self.model,self.active_button_body))
        surface=(points.min(0)+points.max(0))/2;surface[1]=points[:,1].min()
        addresses=[self.model.jnt_qposadr[self.model.joint(self.profile.namespace+n).id] for n in self.planner.names]
        probe=mujoco.MjData(self.model)
        selected=None
        seeds=[np.array([0.,-.3,0.,-1.6,0.,1.8,0.]),self.q(),np.array([0.,-.3,0.,-1.6,0.,1.8,2.5])]
        for tilt in (-20.,35.,0.,60.):
            R=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@np.array([[0,1,0],[0,0,1],[1,0,0]])
            offsets=pad_local@R.T
            tip=offsets[offsets[:,1]>offsets[:,1].max()-1e-5].mean(0)
            pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=surface-tip
            pre=pose.copy();pre[1,3]-=.075
            for seed in seeds:
                try:
                    endpoint=self.nearby_ik(pre,seed,trust_radius=6.,preserve_self_clearance=True)
                    q=endpoint
                    for offset in np.linspace(-.075,.0015,17):
                        target=pose.copy();target[1,3]+=offset
                        q=self.nearby_ik(target,q,preserve_self_clearance=True)
                        probe.qpos[:]=self.data.qpos;probe.qpos[addresses]=q
                        mujoco.mj_forward(self.model,probe)
                        if self.navigation_penetration(probe,False)>.001 or self.robot_self_penetration(probe)>.0005:
                            raise RuntimeError(f'Button approach actual-mesh collision at {offset}: {self.contact_pair_detail(probe,False)}')
                    path=self.planner.plan_joints(self.q(),endpoint)
                    selected=(pose,pre,path);break
                except RuntimeError as exc:self.record(button_approach_rejection=dict(tilt_deg=tilt,error=str(exc)))
            if selected is not None:break
        if selected is None:raise RuntimeError('No checked arm posture for complete button press')
        pose,pre,path=selected
        self.record(button_approach_tilt_deg=tilt)
        self.preplanned_moves['approach coffee button']=(pre,path)
        self.arm('approach coffee button',pre,True)
        for offset in (-.025,-.01,-.003,0.,.0005,.001,.0015):
            target=pose.copy();target[1,3]+=offset;self.arm('press coffee button',target);self.tick(.15)
            if self.button_pressed:break
        else:raise RuntimeError('No measured passive button actuation')
        self.arm('withdraw from coffee button',pre);self.tick(.5);self.press_phase=False
        if self.button_contact()[0]:raise RuntimeError('Finger remains on button')
        self.event('button_released',button=self.active_button_body,travel_m=float(self.data.joint(self.active_button_body.removesuffix('_001')).qpos[0]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DATA_DIR/'base_episodes/coffee_counter',help='Prepared kitchen episode to start from')
    p.add_argument('--asset',type=Path,default=COFFEE_DIR/'espresso_machine',help='Converted espresso machine (convert_asset.py)')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--through',choices=['remove','dose','reinstall','cup','power','press','brew'],default='brew')
    p.add_argument('--resume-from',type=Path,help='Diagnostic initial-state checkpoint; final qualification runs without this')
    p.add_argument('--start',choices=['dose','reinstall','cup','power','press','brew'],default='dose')
    p.add_argument('--removal-witness',type=Path,help='Diagnostic removal checkpoint supplying an arm seed')
    p.add_argument('--machine-offset',type=float,nargs=2,metavar=('DX','DY'),help='Shift the coffee apparatus along the counter (metres)')
    p.add_argument('--machine-slot',choices=sorted(SLOTS),help='Place the apparatus at a random spot in this counter slot (uses --placement-seed)')
    p.add_argument('--placement-seed',type=int,default=0)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    machine_offset=(0.,0.)
    if args.machine_slot:machine_offset=tuple(sample_offset(np.random.default_rng(args.placement_seed),args.machine_slot)[1])
    if args.machine_offset:machine_offset=tuple(args.machine_offset)
    manifest,selection=prepare(args.run,args.asset,out,machine_offset)
    prepare_brew(out,manifest,selection)
    sources={}
    for source in (Path(__file__),Path(__file__).with_name('pour.py'),Path(__file__).with_name('brew.py')):
        destination=out/source.name;shutil.copyfile(source,destination)
        sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'reproduction.json').write_text(json.dumps(dict(argv=sys.argv,source_sha256=sources,
        qualification='diagnostic checkpoint resume' if args.resume_from else 'continuous empty-hand start'),indent=2))
    prior=json.loads((args.run/'report.json').read_text());manifest['box_grasp']=prior['annotation_selection']['local_transform']
    (out/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    ca=SimpleNamespace(**prior['arguments']);ca.output=out;ca.assets=Path(ca.assets);ca.scene_xml=manifest['scene_xml'];ca.spawn=[*shift([2.12,-.94],machine_offset),np.pi/2]
    ca.dynamic_objects=list(ca.dynamic_objects)+[PORTAFILTER,BUTTON,POWER_BUTTON]
    import torch
    random.seed(manifest['seed']);np.random.seed(manifest['seed']);torch.manual_seed(manifest['seed'])
    c=CoffeeWorkflow(ca,selection,manifest)
    def initialize():
        if args.resume_from:
            saved=np.load(args.resume_from)
            if saved['qpos'].shape!=c.data.qpos.shape:raise ValueError('Checkpoint scene layout differs; use a checkpoint from this workflow version')
            if 'machine_cycle' in saved:c.machine_cycle.__dict__.update(json.loads(str(saved['machine_cycle'])))
            c.data.qpos[:]=saved['qpos'];c.data.qvel[:]=saved['qvel'];c.data.ctrl[:]=saved['ctrl'];c.data.eq_active[:]=saved['eq_active'];c.data.time=float(saved['time'])
            c.next_trace=float(c.data.time);c.next_video_frame=float(c.data.time)
            mujoco.mj_forward(c.model,c.data)
            c.fixture_phase='locked' if args.start in ('cup','power','press','brew') else 'released';c.poured=args.start!='dose'
            if 'portafilter_pick_seed' in saved and len(saved['portafilter_pick_seed']):c.portafilter_pick_seed=saved['portafilter_pick_seed'].tolist()
            if 'removal_tcp' in saved and len(saved['removal_tcp']):
                c.removal_tcp=saved['removal_tcp'];c.removal_joints=saved['removal_joints']
            c.report['diagnostic_resume_from']=str(args.resume_from.resolve())
        if args.removal_witness:
            witness=np.load(args.removal_witness)
            addresses=[c.model.jnt_qposadr[c.model.joint(c.profile.namespace+n).id] for n in c.profile.arm_joints]
            c.portafilter_pick_seed=witness['qpos'][addresses].tolist()
            c.remember_removal(json.loads((args.removal_witness.parent/'trace.json').read_text()))
        c.select_object(PORTAFILTER);c.source=c.counter;c.destination=c.counter;c.holding_loaf=False;c.attached=False
        c.support_bids=c.table_bids[c.counter];c.table_gids={g for g in range(c.model.ngeom) if c.model.geom_bodyid[g] in c.support_bids}
        c.initial_object_pose=c.bread_pose().copy();c.pickup_start_height=float(c.initial_object_pose[2,3]);c.transfer_start=c.data.joint(c.object_joint).qpos.copy()
        c.embodiment.open_gripper(c);c.tick(.7)
        if len(c.inventory()['vessel'])+len(c.inventory()['hopper'])!=32:raise RuntimeError('Initial contents did not settle intact')
        c.event('initialized',base=c.base_pose().tolist(),grounds=32)
    ops=dict(initialize=Operation(initialize,lambda:len(c.inventory()['spilled'])==0),
        remove=Operation(c.unlock_remove,lambda:c.fixture_phase=='released' and not c.holding_loaf),
        dose=Operation(c.dose,lambda:c.grounds_ready() and not c.holding_loaf),
        reinstall=Operation(c.reinstall,lambda:c.installed() and c.grounds_ready() and not c.holding_loaf),
        cup=Operation(c.place_cup,c.mug_ready),
        power=Operation(lambda:c.operate_button('power'),lambda:c.machine_cycle.ready(c.data.time)),
        press=Operation(lambda:c.operate_button('brew'),lambda:c.button_pressed and not c.button_contact()[0]),
        brew=Operation(c.finish_brew,lambda:c.machine_cycle.completed and c.mug_ready()))
    names=['initialize','remove','dose','reinstall','cup','power','press','brew'];names=names[:names.index(args.through)+1]
    if args.resume_from:names=['initialize']+names[names.index(args.start):]
    def goal():
        result=dict(empty_hand=not c.holding_loaf and not c.contacts(),
                    contact_clearance=bool(c.report['max_unintended_robot_penetration_m']<=.001 and
                                           c.report['max_finger_object_penetration_m']<=.0015 and
                                           c.report.get('max_robot_self_penetration_m',0.)<=.0005))
        if args.through!='remove':result['all_grounds_captured']=c.grounds_ready()
        if args.through in ['reinstall','cup','power','press','brew']:result['portafilter_locked']=c.installed()
        if args.through in ['press','brew']:result['button_pressed_and_released']=bool(c.button_pressed and not c.button_contact()[0] and abs(float(c.data.joint('button_01_press_pivot').qpos[0]))<.0002)
        if args.through in ['cup','power','press','brew']:result['cup_under_spout']=c.mug_ready()
        if args.through=='brew':result['brew_completed']=c.machine_cycle.completed and not c.machine_cycle.aborted
        return result
    return 0 if CompositeEpisode(c,ops,goal).run('moonlake_robot_full_workflow',[dict(operation=n,arguments={}) for n in names]) else 1

if __name__=='__main__':raise SystemExit(main())
