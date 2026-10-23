"""Room-sized passive lever/cam latch, adapted from RoboSuite door_lock.xml.

Like that asset, this uses a rotating catch, not a translating beveled bolt.
Closing requires depressing the lever until the leaf is seated, then releasing
it. No runtime joint locking, collision toggles, or fixture actuators are used.
"""
import math

import mujoco
import numpy as np


def install_latched_leaf(spec, name, position, heading, width, height,
                         initial_state='closed'):
    if initial_state not in ('open', 'closed'):
        raise ValueError('Latched door initial state must be open or closed')
    if width < .6 or height < 1.5:
        raise ValueError('Expected a room-sized door')
    if any(b.name == name+'_assembly' for b in spec.bodies):
        raise ValueError('Door already installed: '+name)
    root=spec.worldbody.add_body(name=name+'_assembly',pos=position,
        quat=[math.cos(heading/2),0,0,math.sin(heading/2)])
    angle=math.pi/2 if initial_state=='open' else 0.
    leaf=root.add_body(name=name+'_leaf_left',
        quat=[math.cos(angle/2),0,0,math.sin(angle/2)])
    mass=8.2*width/.9*height/2.1
    leaf.explicitinertial=True
    leaf.mass=mass;leaf.ipos=[width/2,0,height/2]
    leaf.inertia=mass/12*np.array([height**2+.036**2,width**2+height**2,width**2+.036**2])
    factor=180/math.pi if spec.compiler.degree else 1.
    hinge=name+'_hinge_left'
    leaf.add_joint(name=hinge,type=mujoco.mjtJoint.mjJNT_HINGE,axis=[0,0,1],
        limited=True,range=[0,90 if spec.compiler.degree else math.pi/2],
        ref=angle*factor,damping=2.,frictionloss=.5)
    leaf.add_geom(name=name+'_panel_left',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[width/2,0,height/2],size=[width/2,.018,height/2],
        rgba=[.48,.29,.15,1],group=1,contype=1,conaffinity=1)
    spindle=leaf.add_body(name=name+'_lever_left',pos=[width-.09,0,.95])
    spindle.explicitinertial=True;spindle.mass=.25
    spindle.ipos=[0,0,0];spindle.inertia=[.0008,.0015,.001]
    lever=name+'_handle_joint_left'
    spindle.add_joint(name=lever,type=mujoco.mjtJoint.mjJNT_HINGE,axis=[0,-1,0],
        limited=True,range=[0,70 if spec.compiler.degree else math.radians(70)],
        stiffness=2.,springref=0.,damping=.12,frictionloss=.015)
    spindle.add_geom(name=name+'_spindle_left',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[.012,.065,0],quat=[math.sqrt(.5),math.sqrt(.5),0,0],
        rgba=[.65,.65,.68,1],group=1,contype=1,conaffinity=1)
    for face in (-1,1):
        spindle.add_geom(name=f'{name}_handle_left_{face}',type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-.06,face*.065,0],size=[.07,.014,.014],
            rgba=[.65,.65,.68,1],group=1,contype=1,conaffinity=1)
    spindle.add_geom(name=name+'_latch_cam',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[.06,0,0],size=[.06,.01,.012],
        rgba=[.8,.65,.18,1],group=1,contype=1,conaffinity=1)
    root.add_geom(name=name+'_strike',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[width+.0225,.045,.95],size=[.0125,.015,.028],
        rgba=[.65,.65,.68,1],group=1,contype=1,conaffinity=1)
    # A fixed cheek supports the strike from the jamb, behind the swing path.
    root.add_geom(name=name+'_strike_mount',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[width+.045,.045,.95],size=[.01,.025,.055],
        rgba=[.3,.3,.32,1],group=1,contype=1,conaffinity=1)
    return dict(name=name,mechanism='passive_rotating_cam_latch',initial_state=initial_state,
        joints=[hinge],handle_joints=[lever],initial_angles=[angle],
        handle_geoms=[f'{name}_handle_left_{face}' for face in (-1,1)],
        latch_geom=name+'_latch_cam',strike_geom=name+'_strike',
        requires_handle_depressed_for_closing=True,
        source_asset='external/robosuite/robosuite/models/assets/objects/door_lock.xml',
        robot_skill_validated=False)
