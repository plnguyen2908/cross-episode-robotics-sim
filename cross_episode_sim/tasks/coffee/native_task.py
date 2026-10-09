"""Task 2: physically pour coarse grounds, operate coffee machine, serve across rooms."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode, run
from cross_episode_sim.tasks.breakfast.scene import bounds, support_bounds
from cross_episode_sim.fixtures.blender_lid import BlenderLidTest, LID
from cross_episode_sim.fixtures.blender_load import align_disk_center
from cross_episode_sim.tasks.coffee.native_scene import prepare, DEFAULT_CONFIG, COFFEE
from cross_episode_sim.tasks.coffee.grounds import BrewState, grain_inventory, pour_pose, HOPPER
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.tasks.coffee.placement import VALIDATED_FRONT_EDGE_Y, to_validated, to_world
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.edge_access import object_bodies, collision_vertices


class CoffeeEpisode(BreakfastEpisode):
    video_filename='coffee_ground_brew_serve.mp4'

    def __init__(self,args,selection,manifest):
        self.spec=manifest['coffee'];self.brew=BrewState(duration=self.spec['brew_seconds'])
        self.coffee_events=[];self.pouring=False;self.poured=False;self.press_phase=False
        super().__init__(args,selection,manifest)
        self.grain_body_ids=np.array([self.model.body(n).id for n in self.spec['grains']],dtype=int)
        self.grain_ids=set(self.grain_body_ids)
        self.anchor=np.asarray(self.spec['lid_anchor'])
        self.closed_pose=np.eye(4);self.closed_pose[:3,3]=self.anchor
        self.counter=self.task_supports['kitchen'];self.lid_release_clearance=.004
        self.lid_placement_offsets=[(0.,0.),(.06,0.),(-.06,0.),(0.,.06)]
        self.button_ids=[g for g in range(self.model.ngeom)
                         if (self.model.geom(g).name or '').startswith('coffee_machine_')
                         and 'start_button' in self.model.geom(g).name]
        if not self.button_ids:raise ValueError('Coffee machine lacks a contact button')
        self.report.update(scope='task 2: grounds pouring, lid, contact start, brewing surrogate and cross-room serving',
            coffee_events=self.coffee_events,atomic_results=[],simulation_contract=manifest['simulation_contract'])

    def info(self,role):
        return next(i for i in self.manifest['bindings'] if i['role']==role)

    def object_label(self):
        info=getattr(self,'object_info',{}).get(getattr(self,'object_name',None),{})
        if info.get('role')=='grounds_cup' and info.get('vessel_type')=='box':
            return 'small coffee grounds box'
        return {'grounds_lid':'coffee grounds lid','grounds_cup':'coffee grounds cup',
                'coffee_mug':'serving coffee mug'}.get(info.get('role'),super().object_label())

    def navigate_to_site(self,room,point,carrying,skip=0):
        self._coffee_navigation_label=room.replace('_',' ')
        return super().navigate_to_site(room,point,carrying,skip)

    # The video is rendered after physics, so each frame keeps where the robot was headed.
    replay_attributes=('_coffee_navigation_label',)

    def tick(self,seconds):
        first=len(self.trace)
        result=super().tick(seconds)
        place=getattr(self,'_coffee_navigation_label',None)
        if place is not None:
            for row in self.trace[first:]:row['_coffee_navigation_label']=place
        return result

    def describe_stage(self,stage):
        place=getattr(self,'_coffee_navigation_label','coffee station')
        label=super().describe_stage(stage)
        # Carry captions name the kitchen counter; serving goes to the dining table.
        label=label.replace('kitchen counter','dining table') if place=='dining' else label.replace('dining table',place)
        if self.spec.get('dosing_container')=='small_box':
            label=label.replace('dosing cup','grounds box')
        return label

    def inventory(self):return grain_inventory(self.model,self.data,self.spec,getattr(self,'grain_body_ids',None))

    def grounds_ready(self):
        inv=self.inventory();n=len(self.spec['grains'])
        return (len(inv['hopper'])/n>=self.spec['minimum_hopper_fraction']
                and len(inv['spilled'])/n<=self.spec['maximum_spill_fraction'])

    def lid_closed(self):
        lid=self.data.body(LID);center=np.asarray(self.spec['lid_disk_center'])
        actual=lid.xpos+lid.xmat.reshape(3,3)@center
        target=self.anchor+center
        support=object_bodies(self.model,HOPPER);own=object_bodies(self.model,LID)
        contact=any((self.model.geom_bodyid[c.geom1] in own and self.model.geom_bodyid[c.geom2] in support)
                    or (self.model.geom_bodyid[c.geom2] in own and self.model.geom_bodyid[c.geom1] in support)
                    for c in self.data.contact)
        return bool(contact and np.linalg.norm(actual-target)<.018
                    and lid.xmat.reshape(3,3)[2,2]>np.cos(np.deg2rad(7.)))

    def mug_ready(self):
        body=self.info('coffee_mug')['body'];own=object_bodies(self.model,body)
        machine=object_bodies(self.model,COFFEE)
        contact=any((self.model.geom_bodyid[c.geom1] in own and self.model.geom_bodyid[c.geom2] in machine)
                    or (self.model.geom_bodyid[c.geom2] in own and self.model.geom_bodyid[c.geom1] in machine)
                    for c in self.data.contact)
        return bool(contact and np.linalg.norm(self.data.body(body).xpos[:2]-np.asarray(self.spec['dispenser_target'])[:2])<.03
                    and self.data.body(body).xmat.reshape(3,3)[2,2]>.94)

    def button_contact(self):
        force=0.
        for i,contact in enumerate(self.data.contact):
            if contact.geom1 in self.button_ids:other=contact.geom2
            elif contact.geom2 in self.button_ids:other=contact.geom1
            else:continue
            name=self.model.body(self.model.geom_bodyid[other]).name or ''
            if not name.startswith(self.profile.namespace+'gripper/'):continue
            f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f)
            force+=max(0.,float(f[0]))
        return force>.05,force

    def before_step(self):
        super().before_step()
        if not hasattr(self,'button_ids') or self.data.time<getattr(self,'_next_coffee_check',0.):return
        self._next_coffee_check=float(self.data.time)+.05
        contact,force=self.button_contact()
        event=self.brew.update(float(self.data.time),contact,self.grounds_ready(),self.lid_closed(),self.mug_ready())
        if event:
            entry=dict(time=float(self.data.time),event=event,button_force_n=force,
                       grounds_counts={k:len(v) for k,v in self.inventory().items()})
            self.coffee_events.append(entry);self.record(coffee_event=entry)
        self.set_coffee_visuals(self.brew.started_at is not None and not self.brew.completed,self.brew.completed)
        # Once the pouring action starts, airborne particles are expected. They
        # are counted after settling, not asserted against a tilt threshold.
        if not self.pouring and not self.poured:
            spilled=len(self.inventory()['spilled'])/len(self.spec['grains'])
            if spilled>self.spec['maximum_spill_fraction']:
                since=getattr(self,'_spill_since',float(self.data.time));self._spill_since=since
                if self.data.time-since>.5:raise RuntimeError('Coffee grounds spilled before pouring')
            else:self._spill_since=float(self.data.time)

    def active_grains(self):
        if getattr(self,'object_name',None)!=self.spec['dosing_body']:return set()
        return {self.model.body(n).id for n in self.inventory()['vessel']}

    def kitchen_world_geoms(self):
        active=self.active_grains() if hasattr(self,'grain_ids') else set()
        buttons=set(self.button_ids) if self.press_phase and hasattr(self,'button_ids') else set()
        return [g for g in super().kitchen_world_geoms() if self.model.geom_bodyid[g] not in active and g not in buttons]

    def loaded_payload_bodies(self):return self.bread_bids|self.active_grains()

    def loaded_payload_vertices(self):
        active=self.active_grains();vertices=self.bread_vertices()
        return np.concatenate([vertices,collision_vertices(self.model,self.data,active)]) if active else vertices

    def project_payload_contents(self,probe,reference=None):
        live=self.data if reference is None else reference
        if probe is live or self.object_name!=self.spec['dosing_body']:return
        old=live.body(self.object_name);new=probe.body(self.object_name)
        R=new.xmat.reshape(3,3)@old.xmat.reshape(3,3).T
        for name in grain_inventory(self.model,live,self.spec,getattr(self,'grain_body_ids',None))['vessel']:
            joint=probe.joint(name+'_joint');p=live.body(name)
            joint.qpos[:3]=new.xpos+R@(p.xpos-old.xpos)
            joint.qpos[3:]=Rotation.from_matrix(R@p.xmat.reshape(3,3)).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model,probe)

    def navigation_penetration(self,data,carrying):
        grains=self.active_grains() if hasattr(self,'grain_ids') else set()
        if not grains:return super().navigation_penetration(data,carrying)
        if carrying and data is not self.data:self.project_payload_contents(data)
        payload=self.bread_bids|grains;worst=0.
        for c in data.contact:
            a,b=self.model.geom_bodyid[[c.geom1,c.geom2]]
            na,nb=self.model.body(a).name or '',self.model.body(b).name or ''
            if na.startswith('floor_') or nb.startswith('floor_'):continue
            ra,rb=na.startswith('robot_0/'),nb.startswith('robot_0/')
            if ((ra or carrying and a in payload) and not rb and b not in payload
                or (rb or carrying and b in payload) and not ra and a not in payload):
                worst=max(worst,-float(c.dist)+.001)
        return worst

    def select_object(self,obj):
        super().select_object(obj)
        if self.object_info[obj].get('vessel_type')=='box':
            self.args.annotation_source='geometry_hypotheses'
        if obj==LID:
            self.args.annotation_source='geometry_hypotheses';self.initial_lift_height=.025
            self.annotation_standoff=.07;self.active_family='any'

    def annotation_approach_allowed(self,pose):
        if self.object_name==LID:
            return CrossRoomManipulation.annotation_approach_allowed(self,pose) and pose[2,0]>.5
        if self.object_name==self.spec['dosing_body']:
            # A top-down handle grasp lifts successfully but cannot reach the
            # upright vessel pose above the hopper with this arm. Use side
            # grasps (qualified mug poses or box-tab geometry hypotheses).
            return (CrossRoomManipulation.annotation_approach_allowed(self,pose)
                    and -.5 < pose[2,2] < 0.)
        return super().annotation_approach_allowed(pose)

    def bread_vertices(self):
        if self.object_info[self.object_name].get('vessel_type')=='box':
            # Sample each physical face so the finger-pad filter can measure
            # the tab's width, including contacts between primitive corners.
            points=[]
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] not in self.bread_bids or not self.model.geom_contype[gid]:continue
                half=self.model.geom_size[gid]
                for axis in range(3):
                    for sign in (-1,1):
                        local=np.array([[x,y] for x in np.linspace(-1,1,13) for y in np.linspace(-1,1,13)])
                        face=np.empty((len(local),3));face[:,axis]=sign
                        face[:,[i for i in range(3) if i!=axis]]=local
                        points.append((face*half)@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid])
            return np.concatenate(points)
        if self.object_name!=LID:return super().bread_vertices()
        points=[]
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] not in self.bread_bids or not self.model.geom_contype[gid]:continue
            if self.model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_CYLINDER:continue
            radius,half=self.model.geom_size[gid,:2]
            local=np.array([[radius*np.cos(a),radius*np.sin(a),z] for z in np.linspace(-half,half,17)
                            for a in np.linspace(0,2*np.pi,64,endpoint=False)])
            points.append(local@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid])
        return np.concatenate(points)

    def dock_candidates(self,room,point,pickup=False):
        if (pickup and room=='living'
                and getattr(self,'object_name',None)==self.spec['dosing_body']):
            # Approach the annotated handle side instead of reaching around
            # the cup from the table's default west edge.
            local=np.mean(self.local_annotations[:,:3,3],axis=0)
            direction=self.bread_pose()[:3,:3]@local
            axis=int(np.argmax(np.abs(direction[:2])))
            outward=np.zeros(2);outward[axis]=np.sign(direction[axis])
            yield from self.docks_on_surface_side(room,point,outward,pickup)
            return
        if room in ('hopper','coffee_machine'):
            # Appliance coordinates must not place the base under the surface.
            # Docks stand in front of the surface edge in the machine's frame
            # (validated facing -y), so they follow any machine placement.
            local=to_validated(self.spec,point[:2])
            for offset in (.35,.30,.40,.26,.45):
                for lateral in (0.,.08,-.08):
                    xy=to_world(self.spec,[local[0]+lateral,VALIDATED_FRONT_EDGE_Y-offset])
                    if self.room_id(xy)==self.room_id(point[:2]):
                        yield np.r_[xy,np.arctan2(point[1]-xy[1],point[0]-xy[0])]
            return
        yield from super().dock_candidates(room,point,pickup)

    def prepare_pickup(self):
        if self.object_name!=LID:return super().prepare_pickup()
        info=self.object_info[LID];self.tuck_for_navigation()
        self.navigate_to_site(info['source'],self.bread_pose()[:3,3],False)
        self.embodiment.open_gripper(self);self.tick(.5);self.untuck_for_manipulation()
        self.physically_rejected_annotation_variants=set()

    def state(self):
        p=self.data.body(LID)
        return dict(lid_on_blender=self.lid_closed(),gripper_distance_m=float(np.linalg.norm(self.tcp()[:3,3]-p.xpos)),
                    anchor_error_m=float(np.linalg.norm(p.xpos-self.anchor)))

    def align_lid_center(self,pose):
        return align_disk_center(pose,self.closed_pose,np.asarray(self.spec['lid_disk_center']))

    def tuck_loaded_for_navigation(self):
        filled_mug = (self.object_name == self.info('coffee_mug')['body']
                      and self.brew.completed)
        if self.active_grains() or filled_mug:
            return self.tuck_vessel_for_navigation()
        return super().tuck_loaded_for_navigation()

    def transport_payload(self):
        if self.object_name==LID:
            BlenderLidTest.attach_native_lid(self)
            self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
            self.prepare_lid_placement();return
        if self.object_name==self.spec['dosing_body'] and not self.poured:
            self._placement_pick_yaw=float(self.base_pose()[2])
            self.retreat_before_loaded_tuck();self.tuck_loaded_for_navigation()
            self.navigate_to_site('hopper',np.r_[self.spec['hopper']['center_xy'],self.spec['hopper']['top_z']],True)
            self.prepare_loaded_manipulation();self.pour_grounds()
            info=self.object_info[self.object_name]
            info['destination']='kitchen_right_counter';self.destination=self.task_supports[info['destination']]
            lo,_=bounds(self.model,self.data,self.object_name)
            z=support_bounds(self.model,self.data,self.destination)[1][2]+.002-(lo[2]-self.bread_pose()[2,3])
            info['destination_position']=[*self.spec['dosing_parking_xy'],z]
            self.retreat_before_loaded_tuck();self.tuck_loaded_for_navigation()
            self.navigate_to_site(info['destination'],info['destination_position'],True)
            self.prepare_destination();return
        return super().transport_payload()

    def pour_grounds(self):
        if self.lid_closed():raise RuntimeError('Refusing to pour through a closed lid')
        self.review_phase='POUR COFFEE GROUNDS INTO OPEN HOPPER'
        # Seeded cup annotations are reused for pickup. Tilt targets are planned
        # by cuRobo with the current measured grasp; no object/particle resets.
        errors=[];chosen=None
        stage='position dosing cup above hopper'
        positions=[float(self.data.joint(self.profile.namespace+n).qpos[0]) for n in self.planner.names]
        # The lip target stays over the hopper for every yaw. A grasp acquired
        # from another room can require a different wrist heading at this dock.
        for yaw in self.spec.get('pour_yaws_deg',(0.,90.,-90.,180.,45.,-45.,135.,-135.)):
            target=pour_pose(self.spec,0.,yaw)@np.linalg.inv(self.grasp_relative)
            for method in ('pose','local'):
                try:
                    if method=='pose':
                        goal=[*(target[:3,3]-[0,0,self.embodiment.planner_tool_offset()]),
                              *Rotation.from_matrix(target[:3,:3]).as_quat(scalar_first=True)]
                        trajectory=self.planner.plan(positions,goal)
                    else:
                        endpoint=self.nearby_ik(target,positions,trust_radius=3.,preserve_self_clearance=True)
                        trajectory=self.planner.plan_joints(positions,endpoint)
                    q=np.asarray(trajectory[-1])
                    # An initial pose alone does not qualify a pour: screen the
                    # complete tilt range on this same IK branch before moving.
                    for angle in self.spec['pour_angles_deg'][1:]:
                        pose=pour_pose(self.spec,angle,yaw)@np.linalg.inv(self.grasp_relative)
                        q=self.nearby_ik(pose,q,preserve_self_clearance=True)
                    self.check_loaded_tuck_path(trajectory)
                except RuntimeError as exc:
                    errors.append(f'yaw={yaw}, {method}: {exc}')
                    continue
                self.preplanned_moves[stage]=(target,trajectory)
                self.move(stage,target)
                chosen=yaw;break
            if chosen is not None:break
        if chosen is None:raise RuntimeError(f'No reachable pouring approach: {errors}')
        self.record(coffee_pour_yaw_deg=chosen)
        self.pouring=True
        reached=0.
        for angle in self.spec['pour_angles_deg'][1:]:
            # Use the shared mesh-checked Cartesian servo for short tilt arcs;
            # a free-space joint-path detour could pour outside the opening.
            self.mesh_contact_move(f'tilt dosing cup {angle} degrees',pour_pose(self.spec,angle,chosen)@np.linalg.inv(self.grasp_relative))
            reached=angle;self.tick(self.spec['pour_hold_seconds'])
            self.record(coffee_pour_angle_deg=angle,grounds_counts={k:len(v) for k,v in self.inventory().items()})
            if self.grounds_ready():break
        self.tick(2.)
        measured={k:len(v) for k,v in self.inventory().items()}
        if not self.grounds_ready():raise RuntimeError(f'Grounds pour missed required capture/spill limits: {measured}')
        self.report['grounds_capture']=dict(counts=measured,max_angle_deg=reached,yaw_deg=chosen)
        for angle in np.linspace(reached,0.,max(2,int(reached/15)+1))[1:]:
            # The cup can settle rotationally in the pads while emptying.
            # Recompute the measured grasp before every return target so a
            # stale transform cannot lower its bottom through the hopper rim.
            self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
            self.mesh_contact_move('return dosing cup upright',pour_pose(self.spec,float(angle),chosen)@np.linalg.inv(self.grasp_relative))
        self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.poured=True;self.pouring=False
        self.report['pour_evidence']=dict(success=True,counts=measured,max_angle_deg=reached,yaw_deg=chosen)

    def prepare_destination(self):
        if self.object_name==LID:
            self.prepare_loaded_manipulation();BlenderLidTest.attach_native_lid(self)
            return self.prepare_lid_placement()
        if self.destination!=COFFEE:return super().prepare_destination()
        self.support_bids=self.table_bids[COFFEE]
        self.table_gids={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.support_bids}
        self.prepare_loaded_manipulation()
        current=self.bread_pose()
        local=(self.bread_vertices()-current[:3,3])@current[:3,:3]
        initial=Rotation.from_quat(self.info('coffee_mug')['initial_quaternion'],scalar_first=True).as_matrix()
        handle=initial@np.mean(self.local_annotations[:,:3,3],axis=0)
        front=-np.pi/2-np.arctan2(handle[1],handle[0])
        targets=[]
        for offset in (0.,20.,-20.,40.,-40.):
            p=current.copy();p[:3,3]=self.spec['dispenser_target']
            p[:3,:3]=Rotation.from_euler('z',front+np.deg2rad(offset)).as_matrix()@initial
            p[2,3]=self.spec['tray_z']+.002-(local@p[:3,:3].T)[:,2].min()
            targets.append(p)
        self.destination_pose=targets[0];self.placement_pose_options=targets[1:]
        self.record(placement_policy='fixed native coffee dispenser tray; no nearby-surface substitution')

    def prepare_lid_placement(self):
        self.support_bids=self.table_bids[self.destination]
        self.table_gids={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.support_bids}
        current=self.bread_pose();local=(self.bread_vertices()-current[:3,3])@current[:3,:3]
        targets=[]
        for dx,dy in self.lid_placement_offsets if self.destination==self.counter else [(0.,0.)]:
            pose=current.copy()  # preserve the successful wrist orientation
            if self.destination==self.counter:
                pose[:2,3]=np.asarray(self.spec['lid_parking_xy'])+[dx,dy]
                pose[2,3]=support_bounds(self.model,self.data,self.counter)[1][2]-(local@pose[:3,:3].T)[:,2].min()
            else:pose=self.align_lid_center(pose)
            targets.append(pose)
        self.destination_pose=targets[0];self.placement_pose_options=targets[1:]
        self._placement_staging_attempted=False;self.load_world()

    def approach_table_placement(self, candidate, staging):
        if self.destination != COFFEE:
            return super().approach_table_placement(candidate, staging)
        # The shared table release adds 4 cm. Under this dispenser that puts
        # the mug through the spout; insert horizontally at 4 mm instead.
        target=candidate.copy();target[2,3]+=.004-.04
        front=target.copy();front[1,3]-=.14
        self.move('approach coffee dispenser from front',front)
        self.mesh_contact_move('insert coffee mug under dispenser',target)
        self.record(placement_approach_policy='low horizontal dispenser insertion',
                    release_clearance_m=.004)
        return front

    def verify_role(self,role):
        if role=='grounds_lid':
            return bool(not self.attached and (self.lid_closed() if self.info(role)['destination']=='hopper'
                    else self.assignment().get(LID)==self.counter and not self.lid_closed()))
        if role=='coffee_mug' and self.info(role)['destination']=='coffee_machine':return self.mug_ready() and not self.attached
        return super().verify_role(role)

    def transfer(self,role,source,destination):
        info=self.info(role);info['source']=source;info['destination']=destination
        if role=='grounds_lid':info['destination_position']=(self.anchor.tolist() if destination=='hopper' else [*self.spec['lid_parking_xy'],.94])
        elif role=='coffee_mug':
            if destination=='coffee_machine':info['destination_position']=self.spec['dispenser_target']
            else:
                from cross_episode_sim.tasks.breakfast.scene import site_pose
                info['destination_position']=site_pose(self.model,self.data,info['body'],'dining',.65,.05)
        else:info['destination_position']=[*self.spec['hopper']['center_xy'],self.spec['hopper']['top_z']+.12]
        self.report.get('executed_placement_targets',{}).pop(role,None)
        self.transfer_role(role)

    def verify_transfer(self,role,source,destination):return self.verify_role(role)

    def press_start(self):
        if not (self.grounds_ready() and self.lid_closed() and self.mug_ready()):
            raise RuntimeError('Coffee Start requires grounds, closed lid and correctly positioned mug')
        self.tuck_for_navigation();self.navigate_to_site('coffee_machine',self.spec['dispenser_target'],False)
        self.embodiment.command_gripper(self.data,self.profile.gripper_close);self.tick(.5)
        self.untuck_for_manipulation();self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names)
        self.press_phase=True;self.load_world()
        R=Rotation.from_euler('x',35,degrees=True).as_matrix()@np.array([[0,1,0],[0,0,1],[1,0,0]])
        pad=object_bodies(self.model,self.profile.namespace+'gripper/left_pad')
        offsets=(collision_vertices(self.model,self.data,pad)-self.tcp()[:3,3])@self.tcp()[:3,:3]@R.T
        tip=offsets[offsets[:,1]>offsets[:,1].max()-1e-5].mean(0)
        gid=max(self.button_ids,key=lambda g:self.data.geom_xpos[g,2])
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        vertices=self.data.geom_xpos[gid]+(corners*self.model.geom_size[gid])@self.data.geom_xmat[gid].reshape(3,3).T
        surface=self.data.geom_xpos[gid].copy();surface[1]=vertices[:,1].min()
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=surface-tip
        pre=pose.copy();pre[1,3]-=.07;self.move('approach coffee Start',pre)
        for offset in (-.025,-.01,-.003,-.001,0.,.0005):
            goal=pose.copy();goal[1,3]+=offset;self.move('press coffee Start',goal);self.tick(.15)
            if self.brew.started_at is not None:break
        else:raise RuntimeError('Coffee Start did not register a force-bearing physical press')
        self.move('withdraw from coffee Start',pre);self.tick(.4);self.press_phase=False
        self.load_world();self.tuck_for_navigation()
        if self.button_contact()[0]:raise RuntimeError('Gripper still contacts coffee Start after withdrawal')

    def wait_brew(self):
        if self.brew.started_at is None:raise RuntimeError('Coffee brewing has not started')
        remaining=max(0.,self.spec['brew_seconds']-(self.data.time-self.brew.started_at))
        self.stage='wait for simulated coffee brewing';self.tick(remaining+.2)
        if not self.brew.completed:raise RuntimeError('Coffee brew did not complete with valid prerequisites')

    def set_coffee_visuals(self,brewing,complete):
        for sid in range(self.model.nsite):
            if (self.model.site(sid).name or '').startswith('coffee_machine_') and 'coffee_liquid' in self.model.site(sid).name:
                self.model.site_rgba[sid,3]=float(brewing)
        self.model.site_rgba[self.model.site('task_coffee_surface').id,3]=float(complete)

    def render_video_frame(self,label,time):
        events=[e for e in self.coffee_events if e['time']<=time]
        brewing=bool(events and events[-1]['event']=='brew_started')
        complete=any(e['event']=='brew_complete' for e in events)
        self.set_coffee_visuals(brewing,complete)
        return super().render_video_frame(label+' | brewing is a state surrogate',time)

    def run_test(self,validate_only=False):
        through=self.spec.get('run_through','serve')
        steps=[dict(operation='settle',arguments={})]
        if not validate_only:
            for role,source,destination in [('grounds_lid','hopper','kitchen'),('grounds_cup','living','kitchen'),
                    ('grounds_lid','kitchen','hopper'),('coffee_mug','dining','coffee_machine')]:
                steps.append(dict(operation='transfer',arguments=dict(role=role,source=source,destination=destination)))
            steps.extend([dict(operation='start',arguments={}),dict(operation='brew',arguments={}),
                          dict(operation='transfer',arguments=dict(role='coffee_mug',source='coffee_machine',destination='dining'))])
            steps=steps[:{'open-lid':2,'pour':3,'brew':7,'serve':8}[through]]
        def initial():
            self.tick(1.)
            if not self.lid_closed() or len(self.inventory()['vessel'])!=len(self.spec['grains']):
                raise RuntimeError('Coffee scene must start with a closed lid and grounds in the dosing cup')
            self.tuck_for_navigation()
        ops=dict(settle=Operation(initial,lambda:True),transfer=Operation(self.transfer,self.verify_transfer),
                 start=Operation(self.press_start,lambda:self.brew.started_at is not None and not self.button_contact()[0]),
                 brew=Operation(self.wait_brew,lambda:self.brew.completed))
        def goal():
            if validate_only:return dict(initial_state=self.lid_closed())
            if through=='open-lid':return dict(lid_opened=self.verify_role('grounds_lid'))
            if through=='pour':return dict(grounds_loaded=self.grounds_ready(),dosing_cup_stored=self.verify_role('grounds_cup'))
            if through=='brew':return dict(grounds_loaded=self.grounds_ready(),lid_closed=self.lid_closed(),brewed=self.brew.completed,mug_ready=self.mug_ready())
            return dict(grounds_loaded=self.grounds_ready(),lid_closed=self.lid_closed(),brewed=self.brew.completed,
                        mug_served=self.verify_role('coffee_mug'),dosing_cup_stored=self.verify_role('grounds_cup'),
                        empty_hand=not self.attached and not getattr(self,'holding_loaf',False))
        self.report['full_task_requested']=not validate_only and through=='serve'
        self.report['run_through']=through
        return 0 if CompositeEpisode(self,ops,goal).run('task_02_ground_coffee_brew_serve',steps) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--resume-prepared',action='store_true')
    parser.add_argument('--through',choices=('open-lid','pour','brew','serve'),default='serve')
    args=parser.parse_args();output=args.output.resolve()
    if args.resume_prepared:
        if (output/'report.json').exists():raise FileExistsError('Use a fresh output rather than overwrite an existing run')
        manifest=json.loads((output/'task_manifest.json').read_text())
    else:manifest=prepare(output,json.loads(args.config.read_text()))
    manifest['coffee']['run_through']=args.through
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.prepare_only:
        print(json.dumps(dict(prepared=str(output),task=manifest['task'],physics_success=False)));return 0
    return run(output,manifest,validate_only=args.validate_only,episode_class=CoffeeEpisode)


if __name__=='__main__':raise SystemExit(main())
