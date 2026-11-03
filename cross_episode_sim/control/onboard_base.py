"""Simulator-side base command bridge using native MolmoSpaces controllers.

Only this actuation boundary owns simulation state. Policies receive no model,
world pose or scene geometry. Local encoder coordinates are used only to drive
body-relative velocity commands through the native position actuators.
"""
import numpy as np
import mujoco
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.profile import MOBILE_FRANKA


class OnboardBaseActuator:
    """Body forward/yaw commands through the native controller; encoder feedback.

    The planner sees neither encoder world coordinates nor model/data. No
    teleportation, scene collision queries, or actuator/geometry shortcuts.
    """
    def __init__(self, model, data, motion_guard=None):
        self.motion_guard=motion_guard
        self.model,self.data=model,data
        self.controller=MolmoSpacesController(model,data)
        # Aperture does not encode grasp intent: a closed gripper holding a
        # wide object can still exceed the upstream is_open threshold. Retain
        # the actual command instead of recomputing binary intent from width.
        self.hold={g:(self.controller.robot_view.get_move_group(g).ctrl if g=='gripper' else self.controller.robot_view.get_move_group(g).noop_ctrl).copy() for g in ('base','arm','gripper')}
        self.qids=[model.joint('robot_0/'+n).qposadr[0] for n in ('base_x','base_y','base_theta')]
        self.target=np.asarray(data.qpos[self.qids]).copy()
        self.was_moving=False

    def stationary(self, linear_tolerance=.01, angular_tolerance=.02):
        """Readiness from base encoder velocities, not elapsed idle time."""
        names=('base_x','base_y','base_theta')
        velocity=np.array([self.data.joint('robot_0/'+name).qvel[0] for name in names])
        return bool(np.linalg.norm(velocity[:2])<=linear_tolerance and abs(velocity[2])<=angular_tolerance)

    def step(self, forward, yaw, dt=.02, diagnostic_observer=None):
        if not np.isfinite([forward,yaw,dt]).all() or dt<=0 or abs(forward)>1. or abs(yaw)>1.5:
            raise ValueError('Invalid base command or period')
        if not np.isclose(dt/self.model.opt.timestep,round(dt/self.model.opt.timestep)):
            raise ValueError('Control period must contain an integer number of physics steps')
        measured=np.asarray(self.data.qpos[self.qids])
        if forward != 0. or yaw != 0.:
            arm=self.controller.robot_view.get_move_group('arm').joint_pos
            if self.motion_guard is not None and not self.motion_guard(forward,yaw,dt):
                raise RuntimeError('Observed swept-motion guard rejected base command')
            if self.motion_guard is None and not np.allclose(arm,MOBILE_FRANKA.travel_posture,atol=.05):
                raise RuntimeError('Motion requires the qualified stowed envelope; no unchecked automatic tuck')
        if forward==0. and yaw==0.:
            if self.was_moving:self.target=measured.copy()
            self.was_moving=False
        else:
            # Bounded setpoint lead; tracking discrepancy cannot accumulate forever.
            heading=measured[2]
            self.was_moving=True
            self.target[:2]+=forward*dt*np.array([np.cos(heading),np.sin(heading)])
            self.target[:2]=np.clip(self.target[:2],measured[:2]-.08,measured[:2]+.08)
            self.target[2]=np.clip(self.target[2]+yaw*dt,measured[2]-.08,measured[2]+.08)
        action={k:v.copy() for k,v in self.hold.items()};action['base']=self.target
        self.controller.step(action)
        for _ in range(round(dt/self.model.opt.timestep)):
            mujoco.mj_step(self.model,self.data)
            if diagnostic_observer is not None:diagnostic_observer(self.data)

