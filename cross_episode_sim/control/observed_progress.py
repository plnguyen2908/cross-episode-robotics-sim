"""Retiming a checked path from measured progress, without changing its geometry."""
import numpy as np


def contact_progress_length(offset,yaw,tool_position):
    """Bound tool displacement induced by translation plus base rotation.

    Limiting translation alone permits much larger tool error when the arm
    extends away from the base rotation center.
    """
    return float(np.linalg.norm(offset)+abs(yaw)*np.linalg.norm(np.asarray(tool_position)[:2]))


class ObservedProgressGovernor:
    def __init__(self, length, lead=.006):
        if not np.isfinite([length,lead]).all() or length<=0 or lead<=0:
            raise ValueError('Positive finite length and lead required')
        self.length=float(length);self.lead=float(lead);self.command=0.;self.speed=0.

    def advance(self, nominal, measured, dt=.02):
        if not np.isfinite([nominal,measured,dt]).all() or dt<=0:raise ValueError('Invalid progress or control period')
        # Never reverse because of noisy localization. The actuator holds the
        # existing target while measured motion catches up; the watchdog still
        # rejects excessive tracking error or failure to finish.
        target=max(self.command,min(1.,float(nominal),max(0.,float(measured))+self.lead/self.length))
        remaining=(target-self.command)*self.length
        speed=min(.05,self.speed+.15*dt,np.sqrt(2*.15*remaining))
        step=min(remaining,speed*dt)
        self.speed=step/dt
        self.command=min(target,self.command+step/self.length)
        if (target-self.command)*self.length<1e-12:self.command=target
        return self.command


def contact_feedback_velocity(estimated_pose, target_offset, target_yaw):
    """Bounded body-frame correction from camera-estimated planar pose."""
    velocity=estimated_pose[:3,:3].T@(np.asarray(target_offset)-estimated_pose[:3,3])*5.
    speed=np.linalg.norm(velocity[:2])
    if speed>.05:velocity*=.05/speed
    yaw=np.arctan2(estimated_pose[1,0],estimated_pose[0,0])
    error=np.arctan2(np.sin(target_yaw-yaw),np.cos(target_yaw-yaw))
    return velocity,float(np.clip(5.*error,-.15,.15))


def contact_feedforward_velocity(estimated_pose, previous_offset, previous_yaw,
                                 target_offset, target_yaw, dt=.02):
    """Execute checked path increments, with bounded visual residual correction.

    The fast shared executor supplies feed-forward rather than waiting for a
    position error to develop. Correct against the previous command so the
    next increment is not counted twice. The caller retains the measured
    progress governor and physical tracking/clearance checks.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite control period required')
    correction, angular = contact_feedback_velocity(estimated_pose, previous_offset, previous_yaw)
    increment = np.asarray(target_offset)-np.asarray(previous_offset)
    velocity = estimated_pose[:3,:3].T@increment/dt + correction
    speed = np.linalg.norm(velocity[:2])
    if speed > .05:
        velocity *= .05/speed
    angular += (target_yaw-previous_yaw)/dt
    return velocity, float(np.clip(angular, -.15, .15))
