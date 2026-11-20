"""Simulator bridge for the sensor appliance backend; robot data only.

Fixture state is deliberately absent. Real hardware can construct the same
backend with camera, encoder and 50 Hz command callbacks instead.
"""
from pathlib import Path
import numpy as np


def build_appliance_backend(runtime, *, output, command_observer=None, progress=None):
    import mujoco
    from .episode import episode_sensors
    from ..navigation.shaped_docking import RobotGeometry
    from ..control.observed_arm import robot_calibration
    from ..control.planner_cache import sensor_arm_planner
    from ..control.sensor_appliance_backend import SensorApplianceBackend
    from ..control.m2t2_policy import M2T2Policy
    from ..perception.molmo_pointing import MolmoPointer
    from ..finetuning.molmospaces_execution import MolmoSpacesController
    m,d=runtime.model,runtime.data
    if m is None:raise RuntimeError('An active live episode is required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    controller=MolmoSpacesController(m,d)
    geometry=RobotGeometry(m)
    def prepare_arm():
        return sensor_arm_planner(robot_calibration(m,d,output/'robot'),collision_cache={'cuboid':3000,'mesh':2})
    def joints():return controller.robot_view.get_move_group('arm').joint_pos.copy()
    def fingers():return np.array([float(d.joint(n).qpos[0]) for n in ('robot_0/gripper/left_driver_joint','robot_0/gripper/right_driver_joint')])
    def capture():
        nonlocal base
        # The same backend survives navigation between calls. Hold the current
        # proprioceptive base configuration, not its constructor-time dock.
        base=controller.robot_view.get_move_group('base').noop_ctrl.copy()
        with episode_sensors(m,d,**(runtime.sensor_config or {})) as sensors:return sensors.capture()
    # Geometry and poses below are exclusively the robot subtree, using robot
    # proprioception. No fixture IDs, positions, contacts or joints are exported.
    collision_shapes={}
    def precise_geometry():
        from ..control.convex_finger_contact import NativeConvexShape
        fixed={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
               if (m.joint(j).name or '').startswith('robot_0/gripper/')}
        exact=RobotGeometry(m,gripper_joint_positions=fixed)
        exact.planner_finger_links=[m.body(b).name[len('robot_0/'):].replace('/','_')
            for b in exact.geoms if (m.body(b).name or '').startswith('robot_0/gripper/')
            and not m.body(b).name.endswith('/base') and exact.geoms[b]]
        exact.convex_shapes={};exact.mesh_vertices={}
        for b,geoms in exact.geoms.items():
            if not m.body(b).name.startswith('robot_0/gripper/'):continue
            gids=[g for g in range(m.body_geomadr[b],m.body_geomadr[b]+m.body_geomnum[b])
                  if m.geom_contype[g] or m.geom_conaffinity[g]]
            renamed=[]
            for (name,local,center,half),gid in zip(geoms,gids):
                # Several authored collision meshes are unnamed on the same
                # link. Keep them distinct instead of overwriting a hull.
                name=name+'#'+str(gid);renamed.append((name,local,center,half))
                if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                mid=m.geom_dataid[gid];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid]
                vertices=m.mesh_vert[start:start+count].copy()-center
                exact.mesh_vertices[name]=vertices
                if gid not in collision_shapes:collision_shapes[gid]=NativeConvexShape(vertices)
                exact.convex_shapes[name]=collision_shapes[gid]
            exact.geoms[b]=renamed
        return exact
    def pad_vertices():
        exact=precise_geometry()
        tcp=exact.tcp_pose(joints());corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        vertices=[]
        for box in exact.boxes(joints()):
            if '/left_pad' in box.name:
                local=exact.mesh_vertices.get(box.name,corners*box.half)
                vertices.extend((box.center+local@box.axes.T-tcp[:3,3])@tcp[:3,:3])
        if not vertices:raise RuntimeError('Robot fingertip calibration missing')
        return np.asarray(vertices)
    base=controller.robot_view.get_move_group('base').noop_ctrl.copy()
    def command(q,grip):
        controller.step(dict(base=base.copy(),arm=np.asarray(q),gripper=np.array([grip])))
        for _ in range(round(.02/m.opt.timestep)):
            mujoco.mj_step(m,d)
            if command_observer is not None:command_observer()
    backend=SensorApplianceBackend(capture=capture,joints=joints,fingers=fingers,command=command,
        geometry=geometry,arm=None,prepare_arm=prepare_arm,pad_vertices=pad_vertices,precise_geometry=precise_geometry,pointer=MolmoPointer(),
        motion_epoch=lambda:getattr(runtime,'appliance_motion_epoch',0),
        m2t2=M2T2Policy('/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth',
                       'external/M2T2'),progress=progress or (lambda event,detail:None))
    # Burner knobs use the documented seven-knob stove manual (user decision,
    # November 10; docs/appliance_control_conventions.md): count the observed
    # row and take the burner's ordinal. Other controls keep plain pointing.
    # fix167: plain pointing at 'front left stove burner knob' grasped the
    # oven temperature knob (first in the row), so no burner was switched on.
    from ..perception.stove_control_convention import DocumentedStovePointer
    backend.pointer=DocumentedStovePointer(backend.pointer)
    return backend


def invalidate_appliance_targets(runtime):
    """Call before outside motion; base-frame observation tracks become stale."""
    runtime.appliance_motion_epoch=getattr(runtime,'appliance_motion_epoch',0)+1


def live_appliance_handlers(runtime, *, output):
    """Lazy registration: no model/renderer initialization until a skill is used."""
    handlers=None
    def dispatch(name,**kwargs):
        nonlocal handlers
        if handlers is None:handlers=build_appliance_backend(runtime,output=output).handlers()
        return handlers[name](**kwargs)
    return {name:(lambda _name=name,**kw:dispatch(_name,**kw)) for name in ('rotate_knob','push_button')}
