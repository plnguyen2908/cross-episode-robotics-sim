"""Contact-driven toaster lever with a solver-held, timed catch."""
import copy
from dataclasses import dataclass
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
from cross_episode_sim.fixtures.microwave_button import MicrowaveButtonTest
from cross_episode_sim.fixtures.toaster_insertion import GENERATED, TOASTER, collision_vertices
from cross_episode_sim.paths import read_localized

LEVER='skill_toaster_lever'
JOINT=LEVER+'_joint'
HANDLE=LEVER+'_handle'
CATCH=LEVER+'_catch'


def add_lever_catch(root):
    equality=root.find('equality')
    if equality is None:equality=ET.SubElement(root,'equality')
    ET.SubElement(equality,'joint',name=CATCH,joint1=JOINT,
                  polycoef='0 0 0 0 0',active='false',
                  solref='.02 1',solimp='.999 .999 .001')


class LeverCatch:
    """Hold the measured engagement position through solver forces, not resets."""
    def __init__(self,model,data):
        self.model,self.data=model,data
        self.eq=model.equality(CATCH).id
        self.address=model.jnt_qposadr[model.joint(JOINT).id]

    def update(self,engaged):
        if engaged and not self.data.eq_active[self.eq]:
            self.model.eq_data[self.eq,0]=float(self.data.qpos[self.address])
        self.data.eq_active[self.eq]=bool(engaged)



@dataclass
class ToasterLatch:
    """Single-pair Toaster.update_state semantics, called at native 20 Hz."""
    turned_on: bool=False
    steps_on: int=0
    cooldown: int=0

    def step(self, lever):
        if lever <= .70:self.cooldown=0
        if lever >= .90 and not self.turned_on and self.cooldown==0:
            self.turned_on=True
        latch=False
        if self.turned_on:
            if self.steps_on < 500:
                self.steps_on+=1;latch=True
            else:
                self.turned_on=False;self.steps_on=0;self.cooldown=1
        if 0 < self.cooldown < 1000:self.cooldown+=1
        elif self.cooldown >= 1000:self.cooldown=0
        return latch


def install_lever_scene(scene,obj):
    """Standalone initial state: bread seated, lever up, controls face robot."""
    tree=ET.parse(scene)
    body=tree.find(f".//body[@name='{TOASTER}']");body.set('quat','0 0 0 1')
    native=ET.ElementTree(ET.fromstring(read_localized(GENERATED))).find(f".//body[@name='{LEVER}']/joint")
    tree.find(f".//body[@name='{LEVER}']").insert(0,copy.deepcopy(native))
    add_lever_catch(tree.getroot())
    tree.write(scene)
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    b=d.body(obj);local=(collision_vertices(m,d,{m.body(obj).id})-b.xpos)@b.xmat.reshape(3,3)
    R=Rotation.from_euler('y',90,degrees=True).as_matrix();v=local@R.T
    floor=d.geom('skill_toaster_slotR_floor').xpos.copy()
    xyz=floor.copy();xyz[:2]-=(v.min(0)+v.max(0))[:2]/2
    xyz[2]+=m.geom('skill_toaster_slotR_floor').size[2]-v[:,2].min()+.001
    bread=tree.find(f".//body[@name='{obj}']");bread.set('pos',' '.join(map(str,xyz)))
    bread.set('quat',' '.join(map(str,Rotation.from_matrix(R).as_quat(scalar_first=True))))
    tree.write(scene)
    return xyz


class ToasterLeverTest(MicrowaveButtonTest):
    fixture_body=handle_body=LEVER
    fixture_joint=JOINT
    handle_geometry=HANDLE
    video_filename='toaster_lever.mp4'

    def __init__(self,args,selection):
        self.latch=ToasterLatch();self.lever_events=[];self.physical_trigger=False
        self.maximum_force=0.;self.next_fixture_update=0.
        self.release_time=None;self.stability_min=float('inf');self.stability_max=float('-inf')
        self.stability_max_speed=0.;self.stability_samples=0
        CabinetDoorTest.__init__(self,args,selection)
        self.lever_range=self.model.jnt_range[self.door_joint].copy()
        self.catch=LeverCatch(self.model,self.data)
        self.report.update(task='press native toaster lever, release and tuck',
            fixture_asset='Toaster033',tracked_objects=[self.object_name,LEVER],
            initial_state='free bread seated in slot; passive lever up; power off',
            fixture_update_hz=20,state_events=self.lever_events,
            door_actuation='physical press; solver joint catch after 90% travel; native timer releases catch',
            latch_model='constraint at measured engagement position; no live qpos or qvel resets',
            scope='standalone preloaded toaster control; no simulated heat or browning')

    def object_label(self):return 'toaster lever'
    def interaction_label(self):return 'toaster lever'

    def lever_fraction(self):
        lo,hi=self.lever_range
        return float(np.clip((self.angle()-lo)/(hi-lo),0.,1.))

    def lever_contact(self):
        touched=False;force=0.
        for i,c in enumerate(self.data.contact):
            if self.handle_geom not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.handle_geom else c.geom1
            name=self.model.body(self.model.geom_bodyid[other]).name or ''
            if not name.startswith('robot_0/gripper/'):continue
            touched=True;f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f)
            force+=max(0.,float(f[0]))
        return touched,force

    def before_step(self):
        CabinetDoorTest.before_step(self)
        if not hasattr(self,'lever_range'):return
        touched,force=self.lever_contact();self.maximum_force=max(self.maximum_force,force)
        if self.release_time is not None and self.data.time>=self.release_time+.25 and self.latch.turned_on:
            value=self.angle();speed=abs(float(self.data.qvel[self.model.jnt_dofadr[self.door_joint]]))
            self.stability_min=min(self.stability_min,value);self.stability_max=max(self.stability_max,value)
            self.stability_max_speed=max(self.stability_max_speed,speed);self.stability_samples+=1
        if self.data.time+1e-9 < self.next_fixture_update:return
        self.next_fixture_update=float(self.data.time)+.05
        fraction=self.lever_fraction();was_on=self.latch.turned_on
        latch=self.latch.step(fraction)
        if not was_on and self.latch.turned_on:
            self.physical_trigger=bool(touched and force>.05 and fraction>=.90)
            if not self.physical_trigger:
                raise RuntimeError('Toaster catch cannot engage without physical lever press')
        if was_on!=self.latch.turned_on:
            event=dict(time=float(self.data.time),turned_on=self.latch.turned_on,
                lever_fraction_before_latch=fraction,gripper_contact=touched,contact_force_n=force)
            self.lever_events.append(event);self.record(toaster_state_transition=event)
        self.catch.update(latch)

    def press_pose(self):
        from cross_episode_sim.controller.base import geom_box
        R=Rotation.from_euler('x',-40,degrees=True).as_matrix()@np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        tcp=self.tcp();points=[]
        for g in range(self.model.ngeom):
            name=self.model.body(self.model.geom_bodyid[g]).name or ''
            if not name.endswith(('/left_pad','/right_pad')) or not self.model.geom_contype[g]:continue
            c,h=geom_box(self.model,g)
            points.extend(self.data.geom_xpos[g]+(c+corners*h)@self.data.geom_xmat[g].reshape(3,3).T)
        offsets=(np.asarray(points)-tcp[:3,3])@tcp[:3,:3]@R.T
        tip=offsets[offsets[:,2]<offsets[:,2].min()+1e-5].mean(0)
        surface=self.data.geom_xpos[self.handle_geom].copy()
        surface[2]+=self.model.geom_size[self.handle_geom,2]
        surface[1]+=.003
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=surface-tip
        return pose

    def press_lever(self):
        self.review_phase='PHYSICALLY PRESS TOASTER LEVER'
        self.embodiment.command_gripper(self.data,self.profile.gripper_close);self.tick(.6)
        self.rebuild_planner()
        pose=self.press_pose();pre=pose.copy();pre[2,3]+=.045
        self.move_button('approach toaster lever',pre,seed=np.array([0.,.25,0.,-1.5,0.,3.35,0.]))
        stroke=float(np.ptp(self.lever_range))
        for down in np.arange(-.008,stroke+.004,.003):
            target=pose.copy();target[2,3]-=down
            self.move_button('press toaster lever down',target)
            self.tick(.1)
            self.record(lever_travel_m=self.angle(),lever_fraction=self.lever_fraction(),
                        lever_contact=self.lever_contact(),turned_on=self.latch.turned_on)
            if self.latch.turned_on:break
        else:raise RuntimeError('Physical lever press did not reach native activation threshold')
        if not self.physical_trigger:raise RuntimeError('Toaster activation lacked physical gripper contact')
        self.review_phase='RELEASE TOASTER LEVER'
        retreat=self.tcp().copy();retreat[2,3]+=.04;retreat[1,3]+=.025
        self.move_button('withdraw from toaster lever',retreat)
        self.tick(.5)
        if self.lever_contact()[0] or not self.latch.turned_on:
            raise RuntimeError('Toaster did not remain active after release')
        self.release_time=float(self.data.time)
        self.report['release_state']=dict(turned_on=True,lever_fraction=self.lever_fraction(),gripper_contact=False)
        self.tuck_for_navigation();self.tick(.5)

    def update_recording_cameras(self):
        CabinetDoorTest.update_recording_cameras(self)
        target=self.data.geom_xpos[self.handle_geom]
        self.cameras[0].lookat[:]=target;self.cameras[0].distance=1.8
        self.cameras[0].azimuth=-65.;self.cameras[0].elevation=-20.
        self.cameras[1].lookat[:]=target;self.cameras[1].distance=.45
        self.cameras[1].azimuth=-90.;self.cameras[1].elevation=-15.

    def render_video_frame(self,label,time):
        q=self.angle();fraction=float(np.clip(q/self.lever_range[1],0.,1.))
        events=[e for e in self.lever_events if e['time']<=time]
        on=bool(events and events[-1]['turned_on'])
        return CabinetDoorTest.render_video_frame(self,f'{label} | lever {fraction:.0%} | native power {"ON" if on else "OFF"}',time)

    def run_test(self):
        success=False
        try:
            self.tick(1.);self.report['initial_lever_fraction']=self.lever_fraction()
            self.tuck_for_navigation()
            target=self.data.geom_xpos[self.handle_geom].copy()
            self.task_navigate(target[:2]+[0.,.38],False,face=-np.pi/2)
            self.untuck_for_manipulation()
            self.press_lever()
            floor=self.model.geom('skill_toaster_slotR_floor').id
            supported=any(floor in (c.geom1,c.geom2) and self.model.geom_bodyid[c.geom2 if c.geom1==floor else c.geom1] in self.bread_bids for c in self.data.contact)
            stable=self.stability_samples>=500 and self.stability_max-self.stability_min<.0005 and self.stability_max_speed<.02
            success=bool(self.physical_trigger and self.latch.turned_on and not self.lever_contact()[0]
                         and supported and stable and self.in_default_travel_posture(False))
            self.report.update(bread_supported_on_lowered_tray=supported)
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        finally:
            self.report.update(success=success,physical_trigger=self.physical_trigger,
                final_turned_on=self.latch.turned_on,final_lever_fraction=self.lever_fraction(),
                maximum_gripper_force_n=self.maximum_force,
                released_lever_stability=dict(samples=self.stability_samples,
                    peak_to_peak_m=(self.stability_max-self.stability_min if self.stability_samples else None),
                    maximum_speed_m_s=self.stability_max_speed),
                catch_engaged=bool(self.data.eq_active[self.catch.eq]))
            self.finish_run_outputs(success)
        return 0 if success else 1
