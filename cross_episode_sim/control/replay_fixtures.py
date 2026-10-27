"""Native fixture responses during action replay (never robot planning)."""
from types import SimpleNamespace
import mujoco
import numpy as np


class ReplayFixtures:
    def __init__(self,model,data):
        self.model,self.data=model,data
        self.toaster=None;self.microwave=None;self.next_update=0.
        self.events=[]
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,'skill_toaster_lever_catch')>=0:
            from cross_episode_sim.fixtures.toaster_lever import LeverCatch,ToasterLatch
            self.toaster=(ToasterLatch(),LeverCatch(model,data))
        names=['microwave_main_group_'+n+'_button' for n in ('start','stop')]
        ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in names]
        hinge=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'microwave_main_group_microjoint')
        if min(ids)>=0 and hinge>=0:
            self.microwave=SimpleNamespace(model=model,data=data,button_ids=dict(zip(('start','stop'),ids)),
                pad_bids={model.body('robot_0/gripper/'+side+'_pad').id for side in ('left','right')},
                turned_on=False,door_address=int(model.jnt_qposadr[hinge]))

    def step(self):
        if self.microwave is not None:
            from cross_episode_sim.fixtures.microwave_state import button_contacts,next_state
            c=self.microwave
            start,sf=button_contacts(c,'start');stop,tf=button_contacts(c,'stop')
            state=next_state(c.turned_on,start,stop,abs(c.data.qpos[c.door_address])/(np.pi/2)>=.90)
            if state!=c.turned_on:
                self.events.append(dict(time=float(c.data.time),fixture='microwave',turned_on=state))
            c.turned_on=state
        if self.toaster is not None and self.data.time+1e-9>=self.next_update:
            from cross_episode_sim.fixtures.toaster_lever import JOINT,HANDLE
            latch,catch=self.toaster;self.next_update=float(self.data.time)+.05
            j=self.model.joint(JOINT);lo,hi=j.range
            fraction=float(np.clip((self.data.qpos[j.qposadr[0]]-lo)/(hi-lo),0.,1.))
            before=latch.turned_on;engaged=latch.step(fraction)
            if not before and latch.turned_on:
                gid=self.model.geom(HANDLE).id;force=0.
                for i,contact in enumerate(self.data.contact):
                    if gid not in (contact.geom1,contact.geom2):continue
                    other=contact.geom2 if contact.geom1==gid else contact.geom1
                    name=self.model.body(self.model.geom_bodyid[other]).name or ''
                    if name.endswith(('/left_pad','/right_pad')):
                        wrench=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,wrench);force+=max(0.,wrench[0])
                if force<=.05:raise RuntimeError('Replay toaster catch lacks physical pad contact')
            if before!=latch.turned_on:
                self.events.append(dict(time=float(self.data.time),fixture='toaster',turned_on=latch.turned_on))
            catch.update(engaged)
