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
from cross_episode_sim.control.collision_probe import forward_collision
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.paths import COFFEE_DIR, DATA_DIR
from cross_episode_sim.tasks.coffee.pour import MoonlakeRobotPour, prepare as prepare_pour
from cross_episode_sim.tasks.coffee.machine import PORTAFILTER, CENTER, RIM, BOTTOM, INNER_RADIUS, numbers
from cross_episode_sim.tasks.coffee.native_scene import COFFEE
from cross_episode_sim.tasks.coffee.placement import (
    LID_PARKING, SLOTS, VALIDATED_DOCK, VALIDATED_MOUNT, rotate, sample_offset, to_validated, to_world,
    workspace_wall_conflicts, yaw_degrees)
from cross_episode_sim.tasks.coffee.brew import BrewActions, DEFAULT_COFFEE_MUG, prepare_brew, POWER_BUTTON, SPOUT_XY
from cross_episode_sim.tasks.coffee.grounds import PARTICLE_RADIUS, grain_inventory
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies

BUTTON = 'button_01_press_pivot_001'


def body_pose(data, name):
    b=data.body(name); t=np.eye(4); t[:3,:3]=b.xmat.reshape(3,3); t[:3,3]=b.xpos
    return t


def prepare(run, asset, out, machine_offset=(0., 0.), support=None):
    manifest, selection=prepare_pour(run,asset,out,machine_offset,support)
    spec=manifest['coffee']
    root=ET.parse(out/'robocasa_scene.xml').getroot(); world=root.find('worldbody')
    # Dense Jacobians reduce factorization cost for this contact island while
    # retaining the original Newton solver, timestep and convergence settings.
    # Friction postprocessing limits soft-constraint creep under handle torque.
    root.findall('option')[-1].set('solver','Newton')
    root.findall('option')[-1].set('jacobian','dense')
    root.findall('option')[-1].set('noslip_iterations','5')
    basket=root.find(f".//body[@name='{PORTAFILTER}']")
    seated=np.array(manifest['coffee']['portafilter_original_pose']); seated[:3,3]+=VALIDATED_MOUNT; seated=to_world(spec,seated)
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
    boxpos=to_world(spec,[2.39,-.53,.960]);boxR=rotate(spec,Rotation.from_euler('z',-139.2678933,degrees=True).as_matrix())
    box.set('pos',numbers(boxpos));box.set('quat',numbers(Rotation.from_matrix(boxR).as_quat(scalar_first=True)))
    for binding in manifest['bindings']:
        if binding['body'] == boxname:
            binding.update(source=spec['support_site'], destination=spec['support_site'], position=boxpos.tolist())
    for name in manifest['coffee']['grains']:
        g=root.find(f".//body[@name='{name}']"); pos=np.fromstring(g.get('pos'),sep=' ')
        g.set('pos',numbers(boxpos+boxR@oldR.T@(pos-oldpos)))
    # Park the legacy lid away from the current apparatus.
    lid=root.find(".//body[@name='skill_blender_lid_main']");lid.set('pos',numbers(LID_PARKING))
    local=np.eye(4);local[:3,:3]=Rotation.from_euler("y",65,degrees=True).as_matrix();local[:3,3]=[-.145,0,.040]
    annotations=out/'portafilter_grasps.npz';np.savez(annotations,transforms=np.array([local]))
    info=dict(body=PORTAFILTER,asset='MoonlakePortafilter',key='moonlake__portafilter',role='portafilter',
        source='coffee_machine',destination=spec['support_site'],position=seated[:3,3].tolist(),
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
    from cross_episode_sim.tasks.scene_variation import floor_area
    mount_xy = to_world(spec, VALIDATED_MOUNT)[:2]
    rooms = [room for room in ('kitchen', 'dining', 'living')
             if np.all(mount_xy >= floor_area(model, data, room)[0])
             and np.all(mount_xy <= floor_area(model, data, room)[1])]
    if len(rooms) != 1:
        raise ValueError(f'Coffee machine must be within one supported room, got {rooms}')
    spec['machine_room'] = rooms[0]
    walls=workspace_wall_conflicts(model,data,spec)
    if walls:raise ValueError(f'Coffee placement leaves no room for the robot; walls in its working area: {walls}')
    counter=object_bodies(model,manifest['supports'][spec['support_site']])
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
        # Preserve socket constraint frames when the standalone scene starts detached.
        for name, values in self.spec.get('bayonet_constraint_data', {}).items():
            self.model.eq_data[self.model.equality(name).id] = values
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
        if name in ('portafilter_removed','empty_box_parked','portafilter_reinstalled','cup_placed','machine_ready','cup_retrieved','next_cup'):
            # Later cups get their own checkpoints rather than overwriting the first.
            tag=name+(f'_{self.coffee_mugs().index(self.mug)+1}' if name in ('cup_placed','cup_retrieved') and len(self.coffee_mugs())>1 else '')
            np.savez(self.output/(tag+'_checkpoint.npz'),qpos=self.data.qpos,qvel=self.data.qvel,
                     ctrl=self.data.ctrl,eq_active=self.data.eq_active,time=self.data.time,
                     machine_cycle=json.dumps(self.machine_cycle.__dict__) if hasattr(self,'machine_cycle') else '{}',
                     portafilter_pick_seed=np.array(getattr(self,'portafilter_pick_seed',[])),
                     removal_tcp=np.array(getattr(self,'removal_tcp',[])),removal_joints=np.array(getattr(self,'removal_joints',[])),
                     mug_state=json.dumps(dict(homes=getattr(self,'mug_homes',{}),release=getattr(self,'mug_release_tcp',{}),
                                               filled=getattr(self,'coffee_filled',[]),active=getattr(self,'active_mug',None),
                                               dock=getattr(self,'machine_dock',None))))

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
        self.cameras[1].lookat[:]=to_world(self.spec,[2.12,-.40,1.08])
        self.cameras[1].distance=.80
        self.cameras[1].azimuth=90.+yaw_degrees(self.spec)
        self.cameras[1].elevation=-35.
        phase=getattr(self,'review_phase','')
        if 'BUTTON' in phase:
            self.cameras[1].lookat[:]=to_world(self.spec,[2.12,-.30,1.19])
            self.cameras[1].distance=.60;self.cameras[1].azimuth=135.+yaw_degrees(self.spec);self.cameras[1].elevation=-15.
        elif 'CUP' in phase and hasattr(self,'machine_cycle'):
            self.cameras[1].lookat[:]=self.data.body(self.mug).xpos+[0,0,.04]
            self.cameras[1].distance=.75;self.cameras[1].azimuth=45.+yaw_degrees(self.spec);self.cameras[1].elevation=-30.
        elif 'BREW' in phase:
            self.cameras[1].lookat[:]=to_world(self.spec,[2.12,-.30,1.07])
            self.cameras[1].distance=.60;self.cameras[1].azimuth=130.+yaw_degrees(self.spec);self.cameras[1].elevation=-35.

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
        forward_collision(self.model, probe)

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
                forward_collision(self.model, data)
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
            # Bodies the payload may push against (the cup sliding out along the drip tray).
            contact_ok=getattr(self,'payload_contact_ok_bids',set())
            if (a in payload and b in contact_ok) or (b in payload and a in contact_ok):continue
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

    def held_pose(self,stage,target,*,path=None):
        for correction in range(4):
            self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
            tcp_target=target@np.linalg.inv(self.grasp_relative)
            if path is not None and correction == 0:
                self.mesh_contact_move(stage,tcp_target,path=path)
            else:
                self.arm(stage if not correction else stage+' pose correction',tcp_target)
            if len(self.contacts())<2:raise RuntimeError('Lost bilateral grasp at '+stage)
            actual=self.bread_pose();position_error=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
            angle_error=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()))
            self.report.setdefault('payload_pose_checks',[]).append(dict(stage=stage,correction=correction,position_error_m=position_error,orientation_error_deg=angle_error))
            tight=self.object_name in self.coffee_mugs()
            # With the drip tray 3 mm lower Mug_1 has ~2 mm above and below it under
            # the basket; 1 mm / 0.5 degrees (rim rise under 0.8 mm) keeps that margin.
            # A swapped, shorter mug has room for 1 mm / 1 degree.
            mug_limit=(.001,.5) if 'coffee_mug_grasp_index' not in self.spec else (.001,1.)
            limit=getattr(self,'payload_tolerance',None) or (mug_limit if tight else (.002,1.))
            if position_error<=limit[0] and angle_error<=limit[1]:return
            self.record(payload_pose_correction=correction+1,position_error_m=position_error,orientation_error_deg=angle_error)
        raise RuntimeError(f'Payload failed to follow gripper at {stage}: {position_error:.4f} m, {angle_error:.2f} degrees')

    def pickup(self,obj,local,pre=None,approach_command=None):
        """approach_command: gripper command for the approach instead of fully open."""
        self.select_object(obj);self.source=self.counter;self.destination=self.counter
        self.support_bids=(self.table_bids[self.counter]|getattr(self,'extra_support_bids',set()));self.table_gids={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.support_bids}
        self.initial_object_pose=self.bread_pose().copy();self.pickup_start_height=float(self.initial_object_pose[2,3]);self.transfer_start=self.data.joint(self.object_joint).qpos.copy()
        self.allow_support=True
        if approach_command is None:self.embodiment.open_gripper(self)
        else:self.embodiment.command_gripper(self.data,approach_command)
        self.tick(.4);self.rebuild()
        target=self.bread_pose()@local
        if pre is None:pre=target.copy();pre[:3,3]-=.04*pre[:3,2]
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
        target=self.tcp().copy();self.last_release_tcp=target.copy()
        target[:3,3]+=(-.09*target[:3,2] if retreat_delta is None else np.asarray(retreat_delta))
        self.arm(label,target);self.allow_support=False

    def toward_loading(self,pose):
        """Move a pose straight out of the machine to the loading spot's depth."""
        local=to_validated(self.spec,pose);local[1,3]=to_validated(self.spec,self.loading)[1,3]
        return to_world(self.spec,local)

    def pf_grasp(self):
        local=np.eye(4);local[:3,:3]=Rotation.from_euler('y',65,degrees=True).as_matrix();local[:3,3]=[-.145,0,.040];return local

    def unlock_remove(self):
        from cross_episode_sim.skills.portafilter import DetachPortafilterSkill
        return DetachPortafilterSkill(self).run()

    def dose(self):
        from cross_episode_sim.skills.coffee import dosing_sequence
        return dosing_sequence(self).run()

    def installed(self):
        t=body_pose(self.data,PORTAFILTER)
        return bool(self.data.eq_active[self.lock_eq] and np.linalg.norm(t[:3,3]-self.seated[:3,3])<.003 and Rotation.from_matrix(t[:3,:3]@self.seated[:3,:3].T).magnitude()<np.deg2rad(2))

    def reinstall(self):
        from cross_episode_sim.skills.portafilter import AttachPortafilterSkill
        return AttachPortafilterSkill(self).run()

    def button_prerequisites(self):
        if hasattr(self,'machine_cycle'):return self.machine_button_prerequisites()
        return self.installed() and self.grounds_ready()

    def press(self):
        from cross_episode_sim.skills.buttons import press_coffee_button
        return press_coffee_button(self)


def main(argv=None, *, execution_handler=None, scene_loader=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DATA_DIR/'base_episodes/coffee_counter',help='Prepared kitchen episode to start from')
    p.add_argument('--asset',type=Path,default=COFFEE_DIR/'espresso_machine',help='Converted espresso machine (convert_asset.py)')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nav-speed', type=float, help='Override saved base travel speed in m/s')
    p.add_argument('--turn-speed', type=float, help='Override saved base turning speed in rad/s')
    p.add_argument('--through',choices=['remove','dose','reinstall','cup','power','press','brew'],default='brew')
    p.add_argument('--pour-step-deg', type=int, choices=(5, 10, 15, 20, 30), default=30,
                   help='Tilt execution increment (default 30); 5-degree collision screening is retained.')
    p.add_argument('--portafilter-step-deg', type=int, choices=(5, 10, 15, 25), default=25,
                   help='Bayonet twist target increment (default 25); checked arc samples stay at most 1.5 degrees apart.')
    p.add_argument('--resume-from',type=Path,help='Diagnostic initial-state checkpoint; final qualification runs without this')
    p.add_argument('--start',choices=['dose','reinstall','cup','power','press','brew','next','serve'],default='dose',help="'next' resumes a --cups run from a next_cup checkpoint, at placing the next mug")
    p.add_argument('--removal-witness',type=Path,help='Diagnostic removal checkpoint supplying an arm seed')
    p.add_argument('--machine-offset',type=float,nargs=2,metavar=('DX','DY'),help='Shift the coffee apparatus along the counter (metres)')
    p.add_argument('--machine-slot',choices=sorted(SLOTS),help='Place the apparatus at a random spot in this counter slot (uses --placement-seed)')
    p.add_argument('--placement-seed',type=int,default=0)
    p.add_argument('--machine-pose',type=float,nargs=4,metavar=('DX','DY','DZ','YAW_DEG'),help='Full apparatus pose relative to the validated layout (yaw about the machine mount)')
    p.add_argument('--support',help='Support body under the apparatus (default: the kitchen counter it is on)')
    p.add_argument('--machine-room', choices=('kitchen', 'living', 'dining'), default='kitchen',
                   help='Place the machine on a dedicated coffee table outside the kitchen; food preparation stays in the kitchen')
    p.add_argument('--coffee-mug',default=DEFAULT_COFFEE_MUG[0],help='Qualified asset key for the coffee mug (default: Mug_1)')
    p.add_argument('--coffee-mug-grasp',type=int,default=DEFAULT_COFFEE_MUG[1],help="Index into the mug's qualified grasps used to place it under the spout")
    p.add_argument('--coffee-mug-yaw',type=float,help='Turn of the mug under the spout in degrees (default -45, validated for Mug_1)')
    p.add_argument('--second-mug-xy',type=float,nargs=2,default=[2.55,-.48],metavar=('X','Y'),help='Counter spot of the second mug, in the machine\'s validated frame (with --cups 2)')
    p.add_argument('--mug-navigation',choices=['auto','on','off'],default='auto',help='Drive to each mug before picking it and to its spot before setting it down, like the atomic pick and place (auto: on with --cups 2 or more)')
    p.add_argument('--clutter',type=int,default=0,help='Loose objects on table and counter tops away from the machine (dining table, side table, right counter in turn)')
    p.add_argument('--dynamic-clutter', action='store_true', help='Keep scattered object joints active for generic evaluation skills')
    p.add_argument('--floor-objects',type=int,default=0,help='Loose objects on the floor, one room after another')
    p.add_argument('--table-setting',choices=('office','dining'),default='office',help='dining: remove the office monitor, keyboard and mouse')
    p.add_argument('--episode-label',help='Banner shown in the videos, e.g. "History 2 (given): coffee"')
    p.add_argument('--extra-vessels', type=int, choices=range(7), default=0, help='Current pool size per type is 3 + this value; 0 disables the pool')
    p.add_argument('--serve-breakfast',action='store_true',help='Full breakfast: one bowl per cup; serve each cup beside its filled bowl at the dining table')
    p.add_argument('--skill-plan', type=Path,
                   help='JSON list of public atomic skill calls for the current breakfast episode (oracle or VLM supplied)')
    p.add_argument('--atomic-skill', choices=('detach-portafilter', 'attach-portafilter'),
                   help='Run one standalone portafilter skill on an authored initial scene')
    p.add_argument('--randomize-scene', action='store_true', help='Seeded furniture translations and independent mug homes')
    p.add_argument('--extra-furniture', type=int, choices=range(9), default=0)
    p.add_argument('--prepare-only', action='store_true', help='Author and check the scene without running robot skills')
    p.add_argument('--cups',type=int,choices=(1,2,3,4),default=1,help='Brew this many cups through the spout, one at a time, retrieving each to its counter spot (only with --through brew)')
    args=p.parse_args(argv)
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (args.nav_speed, args.turn_speed)):
        p.error('Navigation speeds must be positive and finite')
    if args.skill_plan and (args.resume_from or args.through != 'brew' or args.atomic_skill or args.prepare_only):
        p.error('--skill-plan requires continuous coffee or full-breakfast execution')
    if args.serve_breakfast and (args.resume_from or args.through != 'brew'):
        p.error('The current skill-interface episode requires a continuous start and --through brew')
    if args.atomic_skill and (args.resume_from or args.serve_breakfast or args.cups != 1):
        p.error('--atomic-skill requires a fresh single-cup scene')
    if args.machine_room != 'kitchen' and (args.machine_pose or args.support or args.machine_offset):
        p.error('--machine-room outside kitchen cannot be combined with an explicit machine pose/offset/support')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from cross_episode_sim.tasks.setup_timing import SetupTiming
    setup = SetupTiming(out)
    if scene_loader is None:
        machine_offset=(0.,0.)
        if args.machine_slot:machine_offset=tuple(sample_offset(np.random.default_rng(args.placement_seed),args.machine_slot)[1])
        if args.machine_offset:machine_offset=tuple(args.machine_offset)
        if args.machine_pose:machine_offset=(*args.machine_pose[:3],np.radians(args.machine_pose[3]))
        preparation_run, support = args.run, args.support
        if args.machine_room != 'kitchen':
            from cross_episode_sim.tasks.coffee.station import prepare_station
            preparation_run, machine_offset, support = setup('coffee station', prepare_station, args.run, out, args.machine_room, args.placement_seed)
        manifest,selection=setup('coffee assets', prepare, preparation_run,args.asset,out,machine_offset,support)
        manifest['food_preparation_room'] = 'kitchen'
        machine_outside_kitchen = manifest['coffee']['machine_room'] != 'kitchen'
        manifest['coffee']['pour_execute_stride_deg'] = args.pour_step_deg
        manifest['coffee']['portafilter_twist_step_deg'] = args.portafilter_step_deg
        if args.coffee_mug_yaw is not None:manifest['coffee']['mug_target_yaw_deg']=args.coffee_mug_yaw
        setup('brew scene', prepare_brew, out,manifest,selection,(args.coffee_mug,args.coffee_mug_grasp),mugs=args.cups,second_home=args.second_mug_xy)
        manifest['coffee']['navigate_to_mugs']=machine_outside_kitchen or args.mug_navigation=='on' or (args.mug_navigation=='auto' and (args.cups>1 or args.serve_breakfast or args.randomize_scene))
        if args.table_setting == 'dining' and manifest.get('fixed_props'):
            tree=ET.parse(manifest['scene_xml']);world=tree.getroot().find('worldbody')
            for name in manifest['fixed_props']:
                node=world.find(f"body[@name='{name}']")
                if node is not None:world.remove(node)
            tree.write(manifest['scene_xml']);manifest['fixed_props']=[]
        if args.cups >= 3:
            from cross_episode_sim.tasks.scene_variation import add_furniture
            add_furniture(Path(manifest['scene_xml']), manifest, args.placement_seed+100, 1, work_table=True)
        if args.randomize_scene:
            from cross_episode_sim.tasks.scene_variation import move_coffee_furniture, randomize_mugs
            setup('furniture positions', move_coffee_furniture, Path(manifest['scene_xml']), manifest, args.placement_seed+101)
            setup('cup positions', randomize_mugs, Path(manifest['scene_xml']), manifest, args.placement_seed+102)
        elif args.cups >= 3 or machine_outside_kitchen:
            from cross_episode_sim.tasks.scene_variation import randomize_mugs
            setup('cup positions', randomize_mugs, Path(manifest['scene_xml']), manifest, args.placement_seed+102)
        if args.atomic_skill == 'attach-portafilter':
            from cross_episode_sim.fixtures.portafilter import prepare_attachment
            prepare_attachment(out, manifest)
        if args.atomic_skill:
            manifest['atomic_skill'] = args.atomic_skill
            if args.atomic_skill == 'detach-portafilter':
                manifest['instruction'] = 'Detach the portafilter from the coffee machine and place it upright on the counter.'
        if args.serve_breakfast:
            from cross_episode_sim.tasks.full_breakfast.scene import add_breakfast
            setup('breakfast bowls', add_breakfast, out,manifest,selection,args.placement_seed if args.placement_seed is not None else 0)
            from cross_episode_sim.tasks.full_breakfast.vessel_pool import add_vessel_pool
            setup('surplus vessels', add_vessel_pool, manifest, selection, args.placement_seed or 0, args.extra_vessels)
        if args.table_setting=='dining' and manifest.get('fixed_props'):
            # No office monitor, keyboard and mouse on the dining table.
            tree=ET.parse(manifest['scene_xml']);world=tree.getroot().find('worldbody')
            for name in manifest['fixed_props']:
                node=world.find(f"body[@name='{name}']")
                if node is not None:world.remove(node)
            tree.write(manifest['scene_xml']);manifest['fixed_props']=[]
        if args.clutter or args.floor_objects:
            # Kept clear: the robot's dock, the machine, its parked parts and the mugs.
            spec=manifest['coffee']
            keep=[to_world(spec,VALIDATED_DOCK[:2]),to_world(spec,SPOUT_XY),to_world(spec,spec['hopper_xy']),
                  to_world(spec,spec['lid_parking_xy']),to_world(spec,spec['mug_home_xy'])]+([to_world(spec,args.second_mug_xy)] if args.cups>1 else [])
            keep+=[i['position'][:2] for i in manifest['bindings'] if i.get('position')]
            tops=[manifest['supports'][k] for k in ('dining','living','kitchen_right_counter') if k in manifest['supports']]
            rooms={manifest['supports'].get('dining'):'dining',manifest['supports'].get('living'):'living'}
            targets=[dict(room=rooms.get(t,'kitchen'),support=t) for t in (tops*args.clutter)[:args.clutter]]
            targets+=[dict(room=('kitchen','dining','living')[i%3]) for i in range(args.floor_objects)]
            from cross_episode_sim.tasks.floor_objects import scatter_objects
            # Table objects also stay off every place setting.
            settings=[i['serving_position'] for i in manifest['bindings'] if i.get('serving_position')]
            manifest['scattered_objects']=setup('clutter placement', scatter_objects, Path(manifest['scene_xml']),targets,
                args.placement_seed if args.placement_seed is not None else 0,avoid_xy=keep,avoid_top_xy=settings)
        if args.extra_furniture:
            from cross_episode_sim.tasks.scene_variation import add_furniture
            setup('extra furniture', add_furniture, Path(manifest['scene_xml']), manifest, args.placement_seed+103, args.extra_furniture)
        from cross_episode_sim.tasks.scene_variation import sync_selection
        sync_selection(manifest, selection)
        (out/'adapter.json').write_text(json.dumps(selection, indent=2))
        manifest['serving_count'] = args.cups
        if args.episode_label:manifest['episode_label']=args.episode_label
    else:
        if execution_handler is None:
            raise ValueError("A frozen scene loader requires model evaluation")
        manifest, selection = scene_loader(out)
    if args.prepare_only:
        from cross_episode_sim.skills.breakfast_interface import current_skill_plan
        (out/'requested_skill_plan.json').write_text(json.dumps(current_skill_plan(manifest), indent=2))
        manifest['preparation_success'] = True
        manifest['physical_validation'] = 'pending'
        (out/'task_manifest.json').write_text(json.dumps(manifest, indent=2))
        (out/'adapter.json').write_text(json.dumps(selection, indent=2))
        print(json.dumps(dict(prepared=str(out), cups=args.cups, physical_validation='pending')), flush=True)
        return 0
    sources={}
    for source in (Path(__file__),Path(__file__).with_name('pour.py'),Path(__file__).with_name('brew.py')):
        destination=out/source.name;shutil.copyfile(source,destination)
        sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'reproduction.json').write_text(json.dumps(dict(argv=sys.argv,source_sha256=sources,
        qualification='diagnostic checkpoint resume' if args.resume_from else 'continuous empty-hand start'),indent=2))
    prior=json.loads((args.run/'report.json').read_text());manifest['box_grasp']=prior['annotation_selection']['local_transform']
    (out/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    ca=SimpleNamespace(**prior['arguments']);ca.output=out;ca.assets=Path(ca.assets);ca.scene_xml=manifest['scene_xml'];ca.spawn=[*to_world(manifest['coffee'],VALIDATED_DOCK[:2]),VALIDATED_DOCK[2]+np.radians(yaw_degrees(manifest['coffee']))]
    for key in ('nav_speed', 'turn_speed'):
        if getattr(args, key) is not None:
            setattr(ca, key, getattr(args, key))
    ca.dynamic_objects=list(ca.dynamic_objects)+[PORTAFILTER,BUTTON,POWER_BUTTON]+manifest['coffee']['coffee_mugs'][1:]
    if args.serve_breakfast:ca.dynamic_objects+=[i['body'] for i in manifest['bindings'] if i['role'].startswith('bowl')]
    if args.dynamic_clutter:
        ca.dynamic_objects = list(dict.fromkeys([*ca.dynamic_objects,
            *(o['body'] for o in manifest.get('scattered_objects', []))]))
    import torch
    random.seed(manifest['seed']);np.random.seed(manifest['seed']);torch.manual_seed(manifest['seed'])
    if args.serve_breakfast:
        from cross_episode_sim.tasks.full_breakfast.episode import FullBreakfastWorkflow as Workflow
    else:Workflow=CoffeeWorkflow
    c=setup('controller initialization', Workflow, ca,selection,manifest)
    def initialize():
        if args.atomic_skill == 'attach-portafilter':
            c.fixture_phase = 'released'
        if args.resume_from:
            saved=np.load(args.resume_from)
            if saved['qpos'].shape!=c.data.qpos.shape:raise ValueError('Checkpoint scene layout differs; use a checkpoint from this workflow version')
            if 'machine_cycle' in saved:c.machine_cycle.__dict__.update(json.loads(str(saved['machine_cycle'])))
            c.data.qpos[:]=saved['qpos'];c.data.qvel[:]=saved['qvel'];c.data.ctrl[:]=saved['ctrl'];c.data.eq_active[:]=saved['eq_active'];c.data.time=float(saved['time'])
            c.next_trace=float(c.data.time);c.next_video_frame=float(c.data.time)
            mujoco.mj_forward(c.model,c.data)
            c.fixture_phase='locked' if args.start in ('cup','power','press','brew','next','serve') else 'released';c.poured=args.start!='dose'
            if 'portafilter_pick_seed' in saved and len(saved['portafilter_pick_seed']):c.portafilter_pick_seed=saved['portafilter_pick_seed'].tolist()
            if 'removal_tcp' in saved and len(saved['removal_tcp']):
                c.removal_tcp=saved['removal_tcp'];c.removal_joints=saved['removal_joints']
            if 'mug_state' in saved:
                state=json.loads(str(saved['mug_state']))
                c.mug_homes=state['homes'];c.mug_release_tcp=state['release'];c.coffee_filled=state['filled']
                if state.get('active'):c.active_mug=state['active']
                if state.get('dock'):c.machine_dock=state['dock']
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
    from cross_episode_sim.skills.coffee import coffee_operations
    ops = coffee_operations(c, initialize)
    if args.atomic_skill:
        operation = 'remove' if args.atomic_skill == 'detach-portafilter' else 'reinstall'
        c.report.update(full_task_requested=False, atomic_task=args.atomic_skill)
        steps = [dict(operation='initialize', arguments={}), dict(operation=operation, arguments={})]
        return 0 if CompositeEpisode(c, ops, lambda: {
            args.atomic_skill: bool(ops[operation].verify()),
            'empty_hand': not c.holding_loaf and not c.attached and not c.contacts(),
            'no_spilled_grounds': not c.inventory()['spilled'],
            'contact_clearance': bool(c.report['max_unintended_robot_penetration_m'] <= .001
                                      and c.report['max_finger_object_penetration_m'] <= .0015
                                      and c.report.get('max_robot_self_penetration_m', 0.) <= .0005),
        }).run(args.atomic_skill, steps) else 1
    names=['initialize','remove','dose','reinstall','cup','power','warmup','press','brew']
    names=names[:names.index('warmup' if args.through == 'power' else args.through)+1]
    if args.resume_from and args.start not in ('next','serve'):names=['initialize']+names[names.index(args.start):]
    if args.start=='next':names=['initialize','cup','press','brew','retrieve']
    elif args.start=='serve':names=['initialize']
    elif args.cups>1:
        if args.through!='brew':raise SystemExit('--cups needs --through brew')
        # The same spout serves every cup: retrieve, place the next, press, brew; retrieve the last.
        names+=['retrieve','next_mug','cup','press','brew']*(args.cups-1)+['retrieve']
    def goal():
        result=dict(empty_hand=not c.holding_loaf and not c.contacts(),
                    contact_clearance=bool(c.report['max_unintended_robot_penetration_m']<=.001 and
                                           c.report['max_finger_object_penetration_m']<=.0015 and
                                           c.report.get('max_robot_self_penetration_m',0.)<=.0005))
        if args.through!='remove':result['all_grounds_captured']=c.grounds_ready()
        if args.through in ['reinstall','cup','power','press','brew']:result['portafilter_locked']=c.installed()
        if args.through in ['press','brew']:result['button_pressed_and_released']=bool(c.button_pressed and not c.button_contact()[0] and abs(float(c.data.joint('button_01_press_pivot').qpos[0]))<.0002)
        if args.through in ['cup','power','press','brew']:result['cup_under_spout']=c.mug_ready()
        if args.through=='brew' and args.cups==1 and (args.resume_from or args.atomic_skill):result['brew_completed']=c.machine_cycle.completed and not c.machine_cycle.aborted
        if args.cups>1 or (args.through == 'brew' and not args.resume_from and not args.atomic_skill):
            result['all_cups_brewed']=len(c.coffee_filled)==args.cups and not any(e['event']=='brew_aborted' for e in c.workflow_events)
            if not args.serve_breakfast:result['cups_back_on_counter']=all(np.linalg.norm(body_pose(c.data,m)[:3,3]-np.array(p)[:3,3])<.02 for m,p in c.mug_homes.items())
            result['every_cup_filled']=sorted(c.coffee_filled)==sorted(c.coffee_mugs())
            result.pop('cup_under_spout',None)
        return result
    steps=[dict(operation=n,arguments={}) for n in names]
    if args.serve_breakfast:
        # A person fills the bowls; then every cup and bowl goes to its place setting.
        from cross_episode_sim.skills.transfer import TransferOperation
        ops.update(fill=Operation(c.fill_bowls,c.bowls_filled),
                   serve=TransferOperation(c, c.serve, c.served))
        steps+=[dict(operation='fill',arguments={})]
        for setting in manifest['full_breakfast']['settings']:
            steps+=[dict(operation='serve',arguments=dict(role=setting['cup'])),dict(operation='serve',arguments=dict(role=setting['bowl']))]
        coffee_goal=goal
        def goal():
            result=coffee_goal()
            # Coffee steps keep the coffee task's limits (1 mm robot/scene, 1.5 mm
            # finger/object, 0.5 mm self); serving uses breakfast's transfer skill and its
            # 3 mm physical-collision limit, as in the breakfast task.
            r=c.report;phase=lambda k:r.get('coffee_phase_'+k,r.get(k,0.))
            result['contact_clearance']=bool(phase('max_unintended_robot_penetration_m')<=.001 and
                phase('max_finger_object_penetration_m')<=.0015 and phase('max_robot_self_penetration_m')<=.0005)
            result['serving_contact_within_breakfast_limit']=bool(r['max_unintended_robot_penetration_m']<=.003)
            if manifest['full_breakfast'].get('interchangeable_vessels'):
                from cross_episode_sim.tasks.full_breakfast.vessel_pool import count_serving_goal
                result.pop('every_cup_filled', None)
                result.pop('all_cups_brewed', None)
                result.update(count_serving_goal(c))
                return result
            result['bowls_filled']=c.bowls_filled()
            for setting in manifest['full_breakfast']['settings']:
                for role in (setting['cup'],setting['bowl']):result[f'{role}_served']=c.served(role)
                result[f"{setting['cup']}_beside_{setting['bowl']}"]=c.beside(setting['cup'],setting['bowl'])
            return result
    if not args.atomic_skill and not args.resume_from and args.through == 'brew':
        from cross_episode_sim.skills.breakfast_interface import breakfast_skill_interface, current_skill_plan
        from cross_episode_sim.skills.interface import OracleDockRecovery, run_skill_episode
        interface = breakfast_skill_interface(c)
        steps = json.loads(args.skill_plan.read_text()) if args.skill_plan else current_skill_plan(manifest)
        if execution_handler is not None:
            # Model evaluation takes over the same initialized public domain;
            # do not pass the oracle witness plan into the model adapter.
            return execution_handler(c, interface, goal, initialize)
        return 0 if run_skill_episode(c, interface, steps, goal,
                                     'full_breakfast' if args.serve_breakfast else 'coffee',
                                     setup=initialize, plan_source=str(args.skill_plan) if args.skill_plan else None,
                                     recover=None if args.skill_plan else OracleDockRecovery()) else 1
    return 0 if CompositeEpisode(c,ops,goal).run('moonlake_robot_full_workflow',steps) else 1

if __name__=='__main__':raise SystemExit(main())
