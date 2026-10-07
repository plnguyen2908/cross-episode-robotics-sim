"""Physical cup/button actions and an explicit, non-fluid brewing state model."""
from dataclasses import dataclass
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.coffee.placement import offset, shift
from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.manipulation.physical_contents import cavity_geometry
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode

POWER_BUTTON = 'button_02_press_pivot_001'
BREW_BUTTON = 'button_01_press_pivot_001'
MUG = 'cup_one_test_object_main'
TRAY = 'drip_tray_grate_lift_pivot_001'


@dataclass
class MachineCycle:
    """Button semantics assigned for this simulation, not manufacturer controls."""
    warmup_seconds: float = 2.
    brew_seconds: float = 8.
    powered_at: float | None = None
    started_at: float | None = None
    completed: bool = False
    aborted: bool = False

    def ready(self, time):
        return self.powered_at is not None and time-self.powered_at >= self.warmup_seconds

    def power_on(self, time):
        if self.powered_at is None:self.powered_at=time

    def start(self, time, locked, grounds, cup):
        if not (self.ready(time) and locked and grounds and cup):
            raise RuntimeError('Brew requires power, ready state, locked loaded portafilter, and supported cup')
        if self.started_at is not None or self.completed:
            raise RuntimeError('Brew cycle already started')
        self.started_at=time;self.aborted=False

    def update(self, time, locked, grounds, cup, released):
        if self.started_at is None or self.completed or self.aborted:return None
        if not (self.ready(time) and locked and grounds and cup):
            self.aborted=True;return 'brew_aborted'
        if released and time-self.started_at >= self.brew_seconds:
            self.completed=True;return 'brew_complete'
        return None


# Coffee spout axis in the validated machine placement.
SPOUT_XY=(2.12,-.2643)

def prepare_brew(out, manifest, selection):
    """Retain native cup scale, make the second source button passive, add sites."""
    tree=ET.parse(out/'robocasa_scene.xml');root=tree.getroot()
    button=root.find(f".//body[@name='{POWER_BUTTON}']")
    ET.SubElement(button,'joint',name=POWER_BUTTON.removesuffix('_001'),type='slide',axis='0 0 1',
                  range='0 .0015',stiffness='180',springref='0',damping='.5',armature='.0001')
    ET.SubElement(root.find('contact'),'exclude',body1=POWER_BUTTON,body2='base_frame_002')
    mug=root.find(f".//body[@name='{MUG}']")
    mug.set('quat','0.7071067811865476 0 0 -0.7071067811865476')
    off=offset(manifest['coffee'])
    mug.set('pos',' '.join(map(str,shift([2.43,-.36,.9746],off))))
    tree.write(out/'robocasa_scene.xml')
    model=mujoco.MjModel.from_xml_path(str(out/'robocasa_scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    cavity=cavity_geometry(model,data,MUG)
    reference=np.asarray(cavity['reference_rotation'])
    center_local=reference.T@np.array([*cavity['center_xy'],0.])
    points=collision_vertices(model,data,object_bodies(model,MUG))
    local=(points-data.body(MUG).xpos)@data.body(MUG).xmat.reshape(3,3)
    tray=collision_vertices(model,data,object_bodies(model,TRAY))
    top=float(tray[:,2].max())
    target=np.eye(4);target[:3,:3]=Rotation.from_euler('z',-45,degrees=True).as_matrix()
    target[:2,3]=shift(SPOUT_XY,off)-(target[:3,:3]@center_local)[:2]
    target[2,3]=top-float((local@reference.T)[:,2].min())+.0005
    ET.SubElement(mug,'site',name='moonlake_coffee_fill',type='cylinder',
                  pos=' '.join(map(str,[*center_local[:2],cavity['bottom_z']+.002])),
                  size=f"{cavity['radius']*.83} .001",rgba='.19 .07 .02 0',group='1')
    ET.SubElement(root.find('worldbody'),'site',name='moonlake_coffee_stream',type='cylinder',
                  pos=' '.join(map(str,shift([*SPOUT_XY,1.04],off))),size='.002 .025',rgba='.22 .09 .03 0',group='1')
    tree.write(out/'robocasa_scene.xml')
    spec=manifest['coffee']
    spec.update(mug_target_pose=target.tolist(),mug_local_vertices=local.tolist(),mug_cavity=cavity,
                mug_center_local=center_local.tolist(),tray_top_z=top,brew_seconds=8.,
                button_roles={'power':POWER_BUTTON,'brew':BREW_BUTTON})
    for info in manifest['bindings']:
        if info['body']==MUG:info.update(source='kitchen',position=shift([2.43,-.36,.9746],off).tolist())
    selection['selected_objects']=manifest['bindings']
    manifest['instruction']='Remove portafilter onto counter, dose from box, reinstall, place cup, power on, start and finish simulated brewing.'
    manifest['simulation_contract']['brewing']='Timed state model with visible fill/stream sites; no fluid, heating, pressure or extraction physics'
    manifest['simulation_contract']['button_roles']='Source button 02 assigned power; source button 01 assigned brew in this simulation'
    (out/'task_manifest.json').write_text(json.dumps(manifest,indent=2));(out/'adapter.json').write_text(json.dumps(selection,indent=2))


class BrewActions:
    """Mixin for CoffeeWorkflow, using its checked physical manipulation methods."""
    def initialize_machine(self):
        self.machine_cycle=MachineCycle(brew_seconds=self.spec['brew_seconds'])
        self.machine_action=None
        # Shared robot rendering hides annotation sites. Give the two brewing
        # indicators their own visible group without showing grasp markers.
        for name in ('moonlake_coffee_fill','moonlake_coffee_stream'):
            self.model.site_group[self.model.site(name).id]=5
        self.render_scene_option.sitegroup[5]=1
        self.report.update(brewing_simulated=True,fluid_simulated=False,
                           machine_cycle=self.machine_cycle.__dict__,button_roles=self.spec['button_roles'])

    def mug_ready(self):
        own=object_bodies(self.model,MUG);tray=object_bodies(self.model,TRAY)
        support=any((self.model.geom_bodyid[c.geom1] in own and self.model.geom_bodyid[c.geom2] in tray) or
                    (self.model.geom_bodyid[c.geom2] in own and self.model.geom_bodyid[c.geom1] in tray) for c in self.data.contact)
        b=self.data.body(MUG);R=b.xmat.reshape(3,3)
        center=b.xpos+R@np.asarray(self.spec['mug_center_local'])
        return bool(support and np.linalg.norm(center[:2]-shift(SPOUT_XY,offset(self.spec)))<.012 and R[2,2]>.995
                    and not (self.holding_loaf and self.object_name==MUG))

    def place_cup(self):
        self.review_phase='PLACE CUP UNDER SPOUT'
        self.extra_support_bids=object_bodies(self.model,TRAY)|object_bodies(self.model,'drip_tray_slide_pivot_001')
        local=np.load(self.info('coffee_mug')['grasp_path'])['transforms'][10]
        self.pickup(MUG,local)
        lift=self.bread_pose().copy();lift[2,3]+=.07;self.held_pose('lift empty coffee cup',lift)
        self.allow_support=False
        target=np.asarray(self.spec['mug_target_pose'])
        # The native mug nearly fills the gap. Enter beside the low forward
        # handle, then slide laterally below the basket; avoid tilting its rim.
        target=target.copy();target[2,3]-=.0003
        off=offset(self.spec)
        front=target.copy();front[:2,3]=shift([2.20,-.46],off)
        self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.arm('approach beside portafilter handle',front@np.linalg.inv(self.grasp_relative),True)
        self.held_pose('level cup beside machine',front)
        self.allow_support=True
        side=front.copy();side[1,3]=-.31+off[1]
        self.held_pose('insert cup beside portafilter handle',side)
        under=side.copy();under[0,3]=target[0,3]
        self.held_pose('slide cup below portafilter basket',under)
        self.held_pose('center cup under coffee spout',target)
        self.release('withdraw hand from coffee cup',retreat_delta=[.04,-.06,0.]);self.tick(.5)
        self.extra_support_bids=set()
        if not self.mug_ready():raise RuntimeError('Cup is not upright, supported and aligned under spout')
        self.event('cup_placed',pose=self.bread_pose().tolist())

    def operate_button(self, action):
        self.machine_action=action;self.active_button_body=self.spec['button_roles'][action]
        bodies=object_bodies(self.model,self.active_button_body)
        self.button_ids=[g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies and self.model.geom_contype[g]]
        self.button_pressed=False
        self.press()
        if action=='power':
            self.stage='wait for simulated machine ready'
            self.tick(max(0.,self.machine_cycle.warmup_seconds-(self.data.time-self.machine_cycle.powered_at))+.1)
            self.event('machine_ready')

    def finish_brew(self):
        if self.machine_cycle.started_at is None:raise RuntimeError('No measured brew start')
        self.review_phase='SIMULATED BREW CYCLE';self.stage='wait for coffee cycle to finish'
        self.tick(max(0.,self.machine_cycle.brew_seconds-(self.data.time-self.machine_cycle.started_at))+.3)
        if not self.machine_cycle.completed:raise RuntimeError('Brew cycle did not finish')

    def machine_before_step(self):
        if not hasattr(self,'machine_cycle'):return
        cycle=self.machine_cycle
        if cycle.started_at is not None and not cycle.completed:
            event=cycle.update(self.data.time,self.installed(),self.grounds_ready(),self.mug_ready(),not self.button_contact()[0])
            if event:self.event(event)
            if cycle.aborted:raise RuntimeError('Simulated brew aborted because a physical prerequisite was lost')

    def machine_button_prerequisites(self):
        if self.machine_action=='power':return True
        return self.machine_cycle.ready(self.data.time) and self.installed() and self.grounds_ready() and self.mug_ready()

    def machine_event(self,name):
        if name!='button_pressed' or not hasattr(self,'machine_cycle'):return
        if self.machine_action=='power':
            self.machine_cycle.power_on(self.data.time);self.event('machine_powered_on')
        elif self.machine_action=='brew':
            self.machine_cycle.start(self.data.time,self.installed(),self.grounds_ready(),self.mug_ready())
            self.event('brew_started',duration_s=self.machine_cycle.brew_seconds)

    def render_machine_frame(self,label,time):
        events=[e for e in self.workflow_events if e['time']<=time]
        start=next((e['time'] for e in events if e['event']=='brew_started'),None)
        completed=any(e['event']=='brew_complete' for e in events)
        aborted=any(e['event']=='brew_aborted' for e in events)
        progress=0. if start is None else min(1.,max(0.,(time-start)/self.machine_cycle.brew_seconds))
        fill=self.model.site('moonlake_coffee_fill').id;stream=self.model.site('moonlake_coffee_stream').id
        cavity=self.spec['mug_cavity'];bottom=cavity['bottom_z']+.002
        height=max(.002,progress*(cavity['rim_z']-.015-bottom))
        self.model.site_pos[fill,2]=bottom+height/2;self.model.site_size[fill,1]=height/2
        self.model.site_rgba[fill,3]=float(start is not None)
        self.model.site_rgba[stream,3]=float(start is not None and not completed and not aborted)
        cup=self.data.body(MUG)
        surface=cup.xpos+cup.xmat.reshape(3,3)@np.array([*self.spec['mug_center_local'][:2],bottom+height])
        spout_z=1.0701
        self.model.site_pos[stream,:]=[*shift(SPOUT_XY,offset(self.spec)),(spout_z+surface[2])/2]
        self.model.site_size[stream,1]=max(.001,(spout_z-surface[2])/2)
        # The older scene's marker is not this machine's dispensing indicator.
        self.model.site_rgba[self.model.site('task_coffee_surface').id,3]=0.
        mujoco.mj_forward(self.model,self.data)
        state='COMPLETE' if completed else 'ABORTED' if aborted else 'BREWING' if start is not None else 'READY' if any(e['event']=='machine_ready' for e in events) else 'OFF'
        return BreakfastEpisode.render_video_frame(self,label+' | '+state+' | simulated brewing; no fluid physics',time)
