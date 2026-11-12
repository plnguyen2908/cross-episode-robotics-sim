"""Live drawer contact motions using robot encoders and observed geometry only.

No simulator fixture names, joint coordinates, contacts or limits are read here.
The supplied planner factory must retain physical collision geometry and mark
only the local intended contact patch as contact-permitted.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DrawerMotionResult:
    completed: bool
    reason: str
    measured_travel: float = 0.


class SensorDrawerMotion:
    def __init__(self, *, joint_position, tcp_pose, fingers, command, planner_factory,
                 outward_normal, progress=lambda stage, detail: None):
        self.joints=joint_position;self.tcp=tcp_pose;self.fingers=fingers
        self.command=command;self.planner_factory=planner_factory;self.progress=progress
        self.normal=np.asarray(outward_normal,float).copy();self.normal[2]=0.
        self.normal/=np.linalg.norm(self.normal)
        self.events=[]

    def settle(self, arm, grip, limit=75):
        last=self.joints().copy();stable=0
        for _ in range(limit):
            self.command(arm,grip);now=self.joints()
            stable=stable+1 if np.max(abs(now-last))<.00015 else 0;last=now
            if stable>=8:return True
        return False

    def gripper(self, value):
        arm=self.joints().copy();last=None;stable=0
        for _ in range(100):
            self.command(arm,value);now=self.fingers()
            stable=stable+1 if last is not None and np.max(abs(now-last))<.0002 else 0;last=now
            if stable>=10:return True
        return False

    def path(self, target, *, grip, contact_patch=False):
        planner=self.planner_factory(self.tcp(),target,contact_patch)
        path=planner.plan(self.joints(),target[:3,3],target[:3,:3])
        for q in path:self.command(q,grip)
        settled=self.settle(path[-1],grip)
        return settled and np.max(np.abs(self.joints()-path[-1]))<.02

    def release_and_withdraw(self, distance=.06, alternatives=()):
        self.progress('release',{})
        if not self.gripper(0.):return DrawerMotionResult(False,'gripper_not_settled')
        self.progress('withdraw',{'distance':distance})
        errors=[]
        for offset in (distance*self.normal,*alternatives):
            target=self.tcp().copy();target[:3,3]+=np.asarray(offset)
            try:
                if not self.path(target,grip=0.,contact_patch=True):
                    return DrawerMotionResult(False,'withdraw_not_settled')
                break
            except RuntimeError as exc:errors.append(str(exc))
        else:return DrawerMotionResult(False,'withdraw_path_blocked: '+'; '.join(errors))
        return DrawerMotionResult(True,'released_and_withdrawn')

    def regrasp(self, observed_handle_pose):
        self.progress('regrasp',{})
        try:
            if not self.path(observed_handle_pose,grip=0.,contact_patch=True):
                return DrawerMotionResult(False,'approach_not_settled')
        except RuntimeError as exc:return DrawerMotionResult(False,'regrasp_path_blocked: '+str(exc))
        if not self.gripper(255.):return DrawerMotionResult(False,'gripper_not_settled')
        return DrawerMotionResult(True,'gripper_closed')

    def move_until_resistance(self, *, opening, step=.04, travel_budget=1.):
        if not 0<step<=.04 or travel_budget<=0:
            raise ValueError('Drawer contact segments must be in (0, 0.04] m')
        direction=self.normal if opening else -self.normal
        origin=self.tcp().copy();last_cmd=self.joints().copy()
        stage='open_pull' if opening else 'close_push'
        for distance in np.arange(step,travel_budget+step/2,step):
            target=origin.copy();target[:3,3]+=distance*direction
            self.progress(stage,{'commanded_distance':float(distance)})
            try:
                planner=self.planner_factory(self.tcp(),target,True)
                path=planner.plan(self.joints(),target[:3,3],target[:3,:3])
            except RuntimeError as exc:
                return DrawerMotionResult(False,'motion_path_blocked: '+str(exc),float(np.dot(self.tcp()[:3,3]-origin[:3,3],direction)))
            resisted=False
            for q in path:
                self.command(q,255.);last_cmd=q.copy()
                # Planner FK is supplied only through the robot-only tcp callback.
                commanded=self.tcp(q)[:3,3];measured=self.tcp()[:3,3]
                if np.dot(commanded-measured,direction)>.006:
                    for _ in range(20):
                        self.command(q,255.);measured=self.tcp()[:3,3]
                        if np.dot(commanded-measured,direction)<.003:break
                    if np.dot(commanded-measured,direction)>.004:
                        resisted=True;break
            travel=float(np.dot(self.tcp()[:3,3]-origin[:3,3],direction))
            self.events.append(dict(stage=stage,commanded=float(distance),measured=travel,resisted=resisted))
            if resisted:
                self.settle(self.joints().copy(),255.)
                return DrawerMotionResult(True,'resisted_motion',travel)
        self.settle(last_cmd,255.)
        return DrawerMotionResult(False,'travel_budget_exhausted',float(np.dot(self.tcp()[:3,3]-origin[:3,3],direction)))
