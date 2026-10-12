"""Physical cup/button actions and an explicit, non-fluid brewing state model."""
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.manipulation.physical_contents import cavity_geometry
from cross_episode_sim.paths import DATA_DIR
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
from cross_episode_sim.tasks.coffee.placement import machine_pose, rotate, to_validated, to_world

POWER_BUTTON = 'button_02_press_pivot_001'
BREW_BUTTON = 'button_01_press_pivot_001'
MUG = 'cup_one_test_object_main'
# A second Mug_1 for brewing two cups, on the counter right of and in front of the first.
MUG_TWO = 'coffee_mug_two_test_object_main'
MUG_TWO_HOME = (2.55, -.48)
TRAY = 'drip_tray_grate_lift_pivot_001'
# Mug_1 is 104.2 mm tall against 105.1 mm below the spout; seating the drip
# tray 3 mm lower gives it room to slide back out after brewing.
TRAY_DROP = .003
# How far below its resting height the cup is pressed while sliding out. Held by
# the handle, pressing tilts the cup (4.8 degrees at 1.5 mm); with the lowered
# tray a level slide clears, so the cup slides out at its resting height.
PUSH_DOWN = 0.


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

    def next_cup(self):
        """Ready the powered machine for another cup."""
        self.started_at=None;self.completed=False;self.aborted=False

    def update(self, time, locked, grounds, cup, released):
        if self.started_at is None or self.completed or self.aborted:return None
        if not (self.ready(time) and locked and grounds and cup):
            self.aborted=True;return 'brew_aborted'
        if released and time-self.started_at >= self.brew_seconds:
            self.completed=True;return 'brew_complete'
        return None


# Coffee spout axis in the validated machine placement.
SPOUT_XY=(2.12,-.2643)

# The Moonlake spout clears only 105 mm above the drip tray. Mug_1 (104 mm) fits
# with 0.9 mm to spare, enough to place it but not to take it back out reliably.
DEFAULT_COFFEE_MUG=('molmo__Mug_1',10)
MUG_1_COFFEE_GRASP=DATA_DIR/'base_episodes/coffee_prepare/coffee_mug_grasps.npz'


def swap_coffee_mug(root, manifest, out, key, grasp_index):
    """Replace the coffee mug with another qualified asset at its native scale.

    The new mug keeps the body name the coffee code uses; its geometry comes from
    its grasp-sweep recording and its grasps from that model's qualified registry.
    """
    from pathlib import Path

    from cross_episode_sim.tasks.breakfast.scene import merge_object, read_qualified
    old=root.find(f".//body[@name='{MUG}']")
    next(p for p in root.iter() if old in list(p)).remove(old)
    prefix=MUG.removesuffix('_test_object_main')
    for asset in list(root.find('asset')):
        if asset.get('name','').startswith(prefix+'_test_object_'):root.find('asset').remove(asset)
    row=dict(key=key,asset=key.split('__',1)[1],source=key.split('__',1)[0])
    recording,setup,annotation,evidence=read_qualified(row,out,'coffee_mug')
    scene=next(Path(recording).glob('*.xml'))
    merge_object(root,scene,'test_object_main',prefix)
    if grasp_index<0:
        # Mug_1's validated coffee grasp, transferred: the same hand orientation,
        # 4.1 mm in from the handle's tip and 10.4 mm below the rim.
        reference=np.load(MUG_1_COFFEE_GRASP)['transforms'][10]
        model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        body=data.body('test_object_main');points=(collision_vertices(model,data,object_bodies(model,'test_object_main'))-body.xpos)@body.xmat.reshape(3,3)
        grasp=reference.copy();grasp[:3,3]=[points[:,0].max()-.0041,0.,points[:,2].max()-.0104]
        transforms=np.load(annotation)['transforms']
        np.savez_compressed(annotation,transforms=np.concatenate([transforms,grasp[None]]))
        grasp_index=len(transforms)
    info=next(i for i in manifest['bindings'] if i['body']==MUG)
    info.update(asset=row['asset'],key=key,grasp_path=str(annotation.resolve()),setup=setup,
                recording=str(recording),evidence=evidence)
    manifest['coffee']['coffee_mug_grasp_index']=int(grasp_index)


def prepare_brew(out, manifest, selection, coffee_mug=DEFAULT_COFFEE_MUG, mugs=1, second_home=MUG_TWO_HOME):
    """Retain native cup scale, make the second source button passive, add sites."""
    from cross_episode_sim.tasks.servings import validate_people
    validate_people(mugs)
    tree=ET.parse(out/'robocasa_scene.xml');root=tree.getroot()
    manifest['coffee']['coffee_mugs'] = [MUG]
    if tuple(coffee_mug)!=DEFAULT_COFFEE_MUG:swap_coffee_mug(root,manifest,out,*coffee_mug)
    button=root.find(f".//body[@name='{POWER_BUTTON}']")
    ET.SubElement(button,'joint',name=POWER_BUTTON.removesuffix('_001'),type='slide',axis='0 0 1',
                  range='0 .0015',stiffness='180',springref='0',damping='.5',armature='.0001')
    ET.SubElement(root.find('contact'),'exclude',body1=POWER_BUTTON,body2='base_frame_002')
    mug=root.find(f".//body[@name='{MUG}']")
    spec=manifest['coffee']
    mug_quat=Rotation.from_matrix(rotate(spec,Rotation.from_euler('z',-90,degrees=True).as_matrix())).as_quat(scalar_first=True)
    mug.set('quat',' '.join(map(str,mug_quat)))
    # Mug_1's validated counter spot; a swapped mug starts 6 cm further back, as
    # its lower handle grasp otherwise brings a finger against the parked box.
    home=[2.43,-.36] if tuple(coffee_mug)==DEFAULT_COFFEE_MUG else [2.43,-.30]
    spec['mug_home_xy']=home
    mug.set('pos',' '.join(map(str,to_world(spec,[*home,.9746]))))
    tree.write(out/'robocasa_scene.xml')
    if tuple(coffee_mug)!=DEFAULT_COFFEE_MUG:
        # Rest the new mug on the counter from its own geometry (Mug_1's origin
        # sits 5.3 cm above its base; others differ).
        model=mujoco.MjModel.from_xml_path(str(out/'robocasa_scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        bottom=float(collision_vertices(model,data,object_bodies(model,MUG))[:,2].min())
        position=np.fromstring(mug.get('pos'),sep=' ');position[2]+=(.92+machine_pose(spec)[2]+.0015)-bottom
        mug.set('pos',' '.join(map(str,position)));tree.write(out/'robocasa_scene.xml')
    lower_drip_tray(root,out)
    tree.write(out/'robocasa_scene.xml')
    if mugs>=2:
        add_second_mug(root,mug,manifest,spec,second_home)
        tree.write(out/'robocasa_scene.xml')
    if mugs >= 3:
        add_second_mug(root, mug, manifest, spec, (2.18, -.50), role='coffee_mug_three')
        tree.write(out/'robocasa_scene.xml')
    if mugs >= 4:
        add_second_mug(root, mug, manifest, spec, (2.38, -.50), role='coffee_mug_four')
        tree.write(out/'robocasa_scene.xml')
    model=mujoco.MjModel.from_xml_path(str(out/'robocasa_scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    cavity=cavity_geometry(model,data,MUG)
    reference=np.asarray(cavity['reference_rotation'])
    center_local=reference.T@np.array([*cavity['center_xy'],0.])
    points=collision_vertices(model,data,object_bodies(model,MUG))
    local=(points-data.body(MUG).xpos)@data.body(MUG).xmat.reshape(3,3)
    tray=collision_vertices(model,data,object_bodies(model,TRAY))
    top=float(tray[:,2].max())
    # Mug turn under the spout (validated -45 degrees for Mug_1): sets where its
    # handle, and the hand holding it, sit relative to the portafilter.
    target=np.eye(4);target[:3,:3]=rotate(spec,Rotation.from_euler('z',spec.get('mug_target_yaw_deg',-45.),degrees=True).as_matrix())
    target[:2,3]=to_world(spec,SPOUT_XY)-(target[:3,:3]@center_local)[:2]
    target[2,3]=top-float((local@reference.T)[:,2].min())+.0005
    ET.SubElement(mug,'site',name='moonlake_coffee_fill',type='cylinder',
                  pos=' '.join(map(str,[*center_local[:2],cavity['bottom_z']+.002])),
                  size=f"{cavity['radius']*.83} .001",rgba='.19 .07 .02 0',group='1')
    for other_mug in spec['coffee_mugs'][1:]:
        ET.SubElement(root.find(f".//body[@name='{other_mug}']"),'site',name=fill_site(other_mug),type='cylinder',
                      pos=' '.join(map(str,[*center_local[:2],cavity['bottom_z']+.002])),
                      size=f"{cavity['radius']*.83} .001",rgba='.19 .07 .02 0',group='1')
    ET.SubElement(root.find('worldbody'),'site',name='moonlake_coffee_stream',type='cylinder',
                  pos=' '.join(map(str,to_world(spec,[*SPOUT_XY,1.04]))),size='.002 .025',rgba='.22 .09 .03 0',group='1')
    tree.write(out/'robocasa_scene.xml')
    spec.update(mug_target_pose=target.tolist(),mug_local_vertices=local.tolist(),mug_cavity=cavity,
                mug_center_local=center_local.tolist(),tray_top_z=top,brew_seconds=8.,
                button_roles={'power':POWER_BUTTON,'brew':BREW_BUTTON})
    for info in manifest['bindings']:
        if info['body']==MUG:info.update(source='kitchen',position=np.fromstring(mug.get('pos'),sep=' ').tolist())
    selection['selected_objects']=manifest['bindings']
    manifest['instruction']='Remove portafilter onto counter, dose from box, reinstall, place cup, power on, start and finish simulated brewing.'
    manifest['simulation_contract']['brewing']='Timed state model with visible fill/stream sites; no fluid, heating, pressure or extraction physics'
    manifest['simulation_contract']['button_roles']='Source button 02 assigned power; source button 01 assigned brew in this simulation'
    (out/'task_manifest.json').write_text(json.dumps(manifest,indent=2));(out/'adapter.json').write_text(json.dumps(selection,indent=2))


def lower_drip_tray(root, out):
    """Drop the drip tray (with its grate) by TRAY_DROP in world z."""
    slide=root.find(".//body[@name='drip_tray_slide_pivot_001']")
    model=mujoco.MjModel.from_xml_path(str(out/'robocasa_scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    parent=data.xmat[model.body_parentid[model.body('drip_tray_slide_pivot_001').id]].reshape(3,3)
    slide.set('pos',' '.join(map(str,np.fromstring(slide.get('pos','0 0 0'),sep=' ')+parent.T@[0,0,-TRAY_DROP])))


def add_second_mug(root, mug, manifest, spec, home=MUG_TWO_HOME, role="coffee_mug_two"):
    """A copy of the coffee mug on the counter left of the machine, for a second cup."""
    import copy
    twin=copy.deepcopy(mug)
    prefix=MUG.removesuffix('_test_object_main')
    for node in twin.iter():
        name=node.get('name')
        if name:node.set('name',name.replace(prefix,role,1) if name.startswith(prefix) else role+'_'+name)
    # Same height and turn as the first mug, shifted along the counter in the machine frame.
    position=np.fromstring(mug.get('pos'),sep=' ')+rotate(spec,np.eye(3))@[*np.subtract(home,spec['mug_home_xy']),0.]
    twin.set('pos',' '.join(map(str,position)))
    root.find('worldbody').append(twin)
    info=dict(next(i for i in manifest['bindings'] if i['body']==MUG))
    info.update(body=twin.get('name'),role=role,position=position.tolist())
    manifest['bindings'].append(info)
    spec.setdefault('coffee_mugs', [MUG]).append(twin.get('name'))


def body_pose(data, name):
    b=data.body(name);t=np.eye(4);t[:3,:3]=b.xmat.reshape(3,3);t[:3,3]=b.xpos;return t


def fill_site(mug):
    """The visible coffee-level site inside a mug."""
    return 'moonlake_coffee_fill' if mug==MUG else mug+'_coffee_fill'


class BrewActions:
    """Mixin for CoffeeWorkflow, using its checked physical manipulation methods."""
    @property
    def mug(self):
        """The mug currently being served (one at a time through the same spout)."""
        return getattr(self,'active_mug',MUG)

    def initialize_machine(self):
        self.machine_cycle=MachineCycle(brew_seconds=self.spec['brew_seconds'])
        self.machine_action=None
        # Shared robot rendering hides annotation sites. Give the two brewing
        # indicators their own visible group without showing grasp markers.
        self.coffee_filled=[]
        for name in [fill_site(m) for m in self.coffee_mugs()]+['moonlake_coffee_stream']:
            self.model.site_group[self.model.site(name).id]=5
        self.render_scene_option.sitegroup[5]=1
        self.report.update(brewing_simulated=True,fluid_simulated=False,
                           machine_cycle=self.machine_cycle.__dict__,button_roles=self.spec['button_roles'])

    def coffee_mugs(self):
        return list(self.spec.get('coffee_mugs',[MUG]))

    def coffee_mugs_roles(self):
        return [i['role'] for i in self.manifest['bindings'] if i['body'] in self.coffee_mugs()]

    def mug_ready(self):
        own=object_bodies(self.model,self.mug);tray=object_bodies(self.model,TRAY)
        support=any((self.model.geom_bodyid[c.geom1] in own and self.model.geom_bodyid[c.geom2] in tray) or
                    (self.model.geom_bodyid[c.geom2] in own and self.model.geom_bodyid[c.geom1] in tray) for c in self.data.contact)
        b=self.data.body(self.mug);R=b.xmat.reshape(3,3)
        center=b.xpos+R@np.asarray(self.spec['mug_center_local'])
        return bool(support and np.linalg.norm(center[:2]-to_world(self.spec,SPOUT_XY))<.012 and R[2,2]>.995
                    and not (self.holding_loaf and self.object_name==self.mug))

    def place_cup(self):
        from cross_episode_sim.skills.coffee_cups import PlaceCoffeeCupSkill
        return PlaceCoffeeCupSkill(self).run()

    def walk_to_counter(self, obj, point, carrying=False, room='kitchen'):
        """Navigate first, as the atomic pick and place do: dock beside a counter spot."""
        self.__dict__.setdefault('machine_dock',[float(v) for v in self.base_pose()])
        if not carrying:self.select_object(obj)
        # Held cups fold in by the torso, as breakfast carries them.
        self.tuck_for_navigation()
        self.navigate_to_site(room,np.asarray(point),carrying)
        self.rebuild()

    def return_to_machine(self):
        """Drive back to the machine dock; the spout waypoints were validated from there."""
        dock=self.machine_dock
        alternative = getattr(self, '_public_dock_skip', 0)
        if alternative:
            # Explicit recovery docks belong to the machine's actual support,
            # which can be in another room. Keep the cup's kitchen home intact.
            self.tuck_for_navigation()
            target = np.asarray(self.spec['mug_target_pose'])[:3, 3]
            self.navigate_to_site(self.spec.get('support_site', 'kitchen'), target, self.holding_loaf)
            self.rebuild()
            return
        if not self.holding_loaf:
            self.tuck_for_navigation()
            self.task_navigate(np.array(dock[:2]),False,face=dock[2]);self.rebuild();return
        # Carry the cup folded by the torso (as breakfast does). Drive straight to
        # the machine dock if the fold clears it; otherwise dock normally in front
        # of the machine first and drive straight in from there.
        self.tuck_for_navigation()
        goal=np.array(dock[:2])
        try:
            self.plan_route(goal,True,face=dock[2])
        except RuntimeError as exc:
            self.record(machine_return_direct_rejected=str(exc))
            target=np.asarray(self.spec['mug_target_pose']).copy()
            ready_local=to_validated(self.spec,target);ready_local[:2,3]=[2.20,-.46]
            self.navigate_to_site(self.spec.get('support_site', 'kitchen'),to_world(self.spec,ready_local)[:3,3],True)
        self.task_navigate(goal,True,face=dock[2])
        self.rebuild()

    def next_mug(self):
        """Serve the next mug (cycling) through the same spout."""
        mugs=self.coffee_mugs()
        self.active_mug=mugs[(mugs.index(self.mug)+1)%len(mugs)]
        self.event('next_cup',mug=self.active_mug)

    def retrieve_cup(self):
        from cross_episode_sim.skills.coffee_cups import RetrieveCoffeeCupSkill
        return RetrieveCoffeeCupSkill(self).run()

    def operate_button(self, action):
        self.machine_action=action;self.active_button_body=self.spec['button_roles'][action]
        bodies=object_bodies(self.model,self.active_button_body)
        self.button_ids=[g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies and self.model.geom_contype[g]]
        self.button_pressed=False
        self.press()

    def wait_machine_ready(self):
        if self.machine_cycle.powered_at is None:
            raise RuntimeError('Machine warmup requires measured power-on')
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
            if event:self.event(event,mug=self.mug)
            if event=='brew_complete' and self.mug not in self.coffee_filled:self.coffee_filled.append(self.mug)
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
            self.event('brew_started',duration_s=self.machine_cycle.brew_seconds,mug=self.mug)

    def update_machine_visuals(self, time):
        """Refresh visible liquid indicators for live observations and video replay."""
        events=[e for e in self.workflow_events if e['time']<=time]
        cavity=self.spec['mug_cavity'];bottom=cavity['bottom_z']+.002
        stream=self.model.site('moonlake_coffee_stream').id
        # Each mug shows the level of its own latest brew; the stream runs while one is brewing.
        current=None;start=completed=aborted=None
        for mug in self.coffee_mugs():
            starts=[e for e in events if e['event']=='brew_started' and e.get('mug',MUG)==mug]
            fill=self.model.site(fill_site(mug)).id
            if not starts:
                self.model.site_rgba[fill,3]=0.;continue
            begin=starts[-1]['time']
            done=any(e['event']=='brew_complete' and e.get('mug',MUG)==mug and e['time']>=begin for e in events)
            failed=any(e['event']=='brew_aborted' and e.get('mug',MUG)==mug and e['time']>=begin for e in events)
            progress=min(1.,max(0.,(time-begin)/self.machine_cycle.brew_seconds))
            height=max(.002,progress*(cavity['rim_z']-.015-bottom))
            self.model.site_pos[fill,2]=bottom+height/2;self.model.site_size[fill,1]=height/2
            self.model.site_rgba[fill,3]=1.
            if current is None or begin>start:current,start,completed,aborted,level=mug,begin,done,failed,height
        self.model.site_rgba[stream,3]=float(start is not None and not completed and not aborted)
        if current is None:current,level=self.mug,.002
        height=level
        cup=self.data.body(current)
        surface=cup.xpos+cup.xmat.reshape(3,3)@np.array([*self.spec['mug_center_local'][:2],bottom+height])
        spout_z=1.0701+machine_pose(self.spec)[2]
        self.model.site_pos[stream,:]=[*to_world(self.spec,SPOUT_XY),(spout_z+surface[2])/2]
        self.model.site_size[stream,1]=max(.001,(spout_z-surface[2])/2)
        # The older scene's marker is not this machine's dispensing indicator.
        self.model.site_rgba[self.model.site('task_coffee_surface').id,3]=0.
        mujoco.mj_forward(self.model,self.data)
        state='COMPLETE' if completed else 'ABORTED' if aborted else 'BREWING' if start is not None else 'READY' if any(e['event']=='machine_ready' for e in events) else 'OFF'
        return state

    def render_machine_frame(self,label,time):
        state = self.update_machine_visuals(time)
        return BreakfastEpisode.render_video_frame(self,label+' | '+state+' | simulated brewing; no fluid physics',time)
