"""Native oven door and rack loop; passive fixture joints, actuator-only robot."""
import copy
import xml.etree.ElementTree as ET
import traceback
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.fixtures.faucet import FaucetTest
from cross_episode_sim.paths import FIXTURE_ASSETS_DIR, read_localized

DOOR = 'stove_main_group_door'
RACK = 'stove_main_group_rack1'

def install_native_oven(scene):
    """Select native Oven031, unscaled, in a supported built-in oven bay."""
    tree = ET.parse(scene)
    root = tree.getroot()
    text = read_localized(FIXTURE_ASSETS_DIR/'Oven031.xml').replace('oven_test_', 'stove_main_group_')
    native = ET.fromstring(text)
    world = root.find('worldbody')
    old = next(b for b in world if b.get('name') == 'stove_main_group_main')
    body = copy.deepcopy(native.find('worldbody/body'))
    pos = np.fromstring(old.get('pos'), sep=' ');pos[2]=.80
    body.set('pos',' '.join(map(str,pos)))
    for b in body.iter('body'):
        for joint in list(b.findall('joint')):
            if joint.get('name') not in (DOOR+'_joint', RACK+'_joint'):
                b.remove(joint)
    world.remove(old);world.append(body)
    # Ordinary fixed cabinet plinth supports the native built-in appliance.
    plinth = ET.SubElement(world,'body',name='oven_support_plinth',pos=f'{pos[0]} {pos[1]} .248')
    ET.SubElement(plinth,'geom',type='box',size='.299 .234 .248',rgba='.22 .24 .25 1',group='2')
    assets=root.find('asset')
    for element in list(assets):
        if element.get('name','').startswith('stove_main_group_'):
            assets.remove(element)
    assets.extend(copy.deepcopy(list(native.find('asset'))))
    tree.write(scene)

class OvenRackTest(CabinetDoorTest):
    fixture_body = DOOR
    fixture_joint = DOOR+'_joint'
    handle_body = DOOR
    handle_geometry = 'stove_main_group_door_handle_main'
    video_filename = 'oven_rack_loop.mp4'
    load_world = FaucetTest.load_world

    def __init__(self, args, selection):
        super().__init__(args, selection)
        self.component = 'door'
        self.args.motion_slowdown = 2.
        self.handle_bids = set()
        self.door_id = self.model.joint(DOOR+'_joint').id
        self.rack_id = self.model.joint(RACK+'_joint').id
        self._grasp_geoms = {self.handle_geom}
        for joint in (self.door_id, self.rack_id):
            assert not any(self.model.actuator_trnid[a,0] == joint and
                self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT for a in range(self.model.nu))

    def select_component(self, name):
        self.component = name
        self.door_joint = self.door_id if name == 'door' else self.rack_id
        self.door_address = self.model.jnt_qposadr[self.door_joint]
        self.handle_geom = self.model.geom('stove_main_group_door_handle_main' if name == 'door' else 'stove_main_group_g50').id
        self._grasp_geoms = {self.handle_geom}
        self.articulating = False
        self._precise_contact_boxes = set()

    def object_label(self):
        return 'oven '+getattr(self, 'component', 'door')

    def interaction_label(self):
        return self.object_label()

    def handle_contacts(self):
        fingers = set()
        for c in self.data.contact:
            if c.geom1 in getattr(self, '_grasp_geoms', set()):
                other = c.geom2
            elif c.geom2 in getattr(self, '_grasp_geoms', set()):
                other = c.geom1
            else:
                continue
            name = self.model.body(self.model.geom_bodyid[other]).name or ''
            if self.embodiment.is_finger(name):
                fingers.add(name)
        return sorted(fingers)

    def door_collision(self, data):
        for c in data.contact:
            if c.dist >= -.001:
                continue
            a,b = self.model.geom_bodyid[[c.geom1,c.geom2]]
            na,nb = self.model.body(a).name or '',self.model.body(b).name or ''
            ra,rb = na.startswith('robot_0/'),nb.startswith('robot_0/')
            if not (ra or rb) or (ra and rb):
                continue
            other,robot,gid = (b,na,c.geom2) if ra else (a,nb,c.geom1)
            if (self.model.body(other).name or '').startswith('floor_'):
                continue
            if gid in getattr(self, '_grasp_geoms', set()) and (self.embodiment.is_finger(robot) or robot.endswith(('/left_follower','/right_follower'))):
                continue
            return dict(bodies=[na,nb], depth_m=-float(c.dist))
        depth = self.robot_self_penetration(data)
        return dict(self_collision_m=float(depth)) if depth > .0005 else None

    def kitchen_world_geoms(self):
        gids = CrossRoomManipulation.kitchen_world_geoms(self)
        return [g for g in gids if g not in getattr(self, '_grasp_geoms', set())]

    def plan_route(self, goal, carrying, face=None):
        if getattr(self, '_oven_retreat', False):
            from cross_episode_sim.controller.manipulation import TableReorder
            self._pickup_pre_nav_undock = .25
            return TableReorder.plan_route(self, goal, carrying, face)
        return super().plan_route(goal, carrying, face)

    def move_pose(self, stage, pose, seed=None, joint_target=None):
        current = np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
        q = self.nearby_ik(pose, current if seed is None else seed,
            trust_radius=.6 if seed is None else 6., preserve_self_clearance=True)
        # Native endpoint validates any conservative cuboid refinement.
        initial = self.data.qpos.copy()
        if joint_target is not None:
            initial[self.door_address] = joint_target
        self.validate_door_path([q], np.asarray(q), initial_qpos=initial)
        self.load_world()
        start_gap = self.planner.world_clearance(current,self.planner_world_boxes)
        end_gap = self.planner.world_clearance(q,self.planner_world_boxes)
        blocked = ({b.name for b in self.planner_world_boxes
                    if min(self.planner.world_clearance(q,[b]), self.planner.world_clearance(current,[b])) < 0}
                   if min(start_gap,end_gap) < 0 else set())
        bad = self.door_collision(self.data)
        if bad:
            raise RuntimeError(f'Native articulation start collision: {bad}')
        self.record(planning_start_world_gap=start_gap, planning_end_world_gap=end_gap,
                    planning_start_self_gap=self.planner.self_clearance(current.tolist()))
        self._precise_contact_boxes = getattr(self,'_precise_contact_boxes',set()) | blocked
        self.load_world()
        path = self.planner.plan_joints(current,q)
        self._door_plans = {stage:path}
        super().plan_and_move(stage,pose,hinge_target=joint_target)
        return q

    def target_pose(self):
        pose = np.eye(4)
        pose[:3,3] = self.data.geom_xpos[self.handle_geom]
        if self.component == 'door':
            rot = self.data.xmat[self.model.body(DOOR).id].reshape(3,3)
            pose[:3,:3] = rot @ Rotation.from_euler('x',90,degrees=True).as_matrix() @ np.diag([-1.,1.,-1.])
            pose[0,3] += .14
            pose[:3,3] -= .015*pose[:3,2]
        else:
            pose[:3,:3] = np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
            pose[1,3] -= .012
            pose[0,3] += .14
        return pose

    def grasp(self):
        self.embodiment.open_gripper(self)
        self.tick(.5)
        self.rebuild_planner()
        pose = self.target_pose()
        pre = pose.copy(); pre[:3,3] -= .08*pose[:3,2]
        self.move_pose('reach '+self.component+' pregrasp',pre,
                       seed=np.array([.513,-.293,.474,-2.003,-.606,2.124,-.346]))
        self.move_pose('grasp '+self.component,pose)
        self.embodiment.command_gripper(self.data,self.profile.gripper_close)
        self.tick(1.)
        self.record(component=self.component, physical_fingers=self.handle_contacts())
        if len(self.handle_contacts()) < 2:
            raise RuntimeError('No bilateral contact on oven '+self.component)
        self.articulating = True
        self.rebuild_planner()

    def follow(self,target):
        direction = np.sign(target-self.angle())
        step = np.radians(3) if self.component == 'door' else .015
        tolerance = .02 if self.component == 'door' else .004
        for _ in range(100):
            start = self.angle()
            if direction*(target-start) <= tolerance:
                return
            q = start+direction*min(step,abs(target-start))
            tcp = self.tcp().copy()
            axis = self.data.xaxis[self.door_joint].copy()
            anchor = self.data.xanchor[self.door_joint].copy()
            pose = tcp.copy()
            if self.component == 'door':
                r = Rotation.from_rotvec(axis*(q-start)).as_matrix()
                pose[:3,3] = anchor+r@(tcp[:3,3]-anchor)
                pose[:3,:3] = r@tcp[:3,:3]
            else:
                pose[:3,3] += axis*(q-start)
            self.move_pose(('open ' if direction>0 else 'close ')+self.component,pose,joint_target=float(q))
            if abs(self.angle()-q) > ( .12 if self.component == 'door' else .015):
                raise RuntimeError('Passive '+self.component+' failed to follow grasp')
        raise RuntimeError('Articulation stopped making progress')

    def release(self):
        self.embodiment.open_gripper(self)
        self.tick(.5)
        self.articulating = False
        self.rebuild_planner()
        pose = self.tcp().copy();pose[:3,3] -= .10*pose[:3,2]
        self.move_pose('release and withdraw from '+self.component,pose)
        self.tick(.5)

    def record(self, **metrics):
        if getattr(self,'component','door') == 'rack' and 'door_angle_deg' in metrics:
            metrics.pop('door_angle_deg')
            metrics['rack_position_m'] = self.angle()
        return super().record(**metrics)

    def render_video_frame(self, label, time):
        self.component = 'rack' if 'RACK' in label.upper() else 'door'
        door = float(self.data.qpos[self.model.jnt_qposadr[self.door_id]])
        rack = float(self.data.qpos[self.model.jnt_qposadr[self.rack_id]])
        return super().render_video_frame(f'{label} | door {np.degrees(door):.1f} deg | rack {100*rack:.1f} cm',time)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        for camera,dist in zip(self.cameras[:2],(2.4,1.35)):
            camera.lookat[:]=[3.13,-.85,.6]
            camera.distance=dist;camera.azimuth=120.;camera.elevation=-20.

    def run_test(self):
        success = False
        try:
            resume = getattr(self.args, 'oven_resume', None)
            if resume:
                saved_report = json.loads((Path(resume)/'report.json').read_text())
                if not str(saved_report.get('fixture_asset','')).startswith('Oven031,'):
                    raise RuntimeError('Diagnostic resume requires the same native Oven031 setup')
                self.data.qpos[:] = json.loads((Path(resume)/'trace.json').read_text())[-1]['qpos']
                self.data.qvel[:] = 0.
                for aid in range(self.model.nu):
                    if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                        self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[aid,0]]]
                for joint, actuator in zip(self.profile.base_joints,self.profile.base_actuators):
                    self.data.actuator(self.profile.namespace+actuator).ctrl[0] = float(self.data.joint(self.profile.namespace+joint).qpos[0])
                self.embodiment.command_gripper(self.data,self.profile.gripper_close)
                mujoco.mj_forward(self.model,self.data)
                self.report['diagnostic_resume'] = str(resume)
            self.tick(.2 if resume else 1.)
            self.report.update(task='open oven, pull rack, push rack, close oven',
                fixture_actuation='native passive joints; physical gripper contact only',
                fixture_asset='Oven031, unscaled, center height 0.80 m on fixed cabinet plinth')
            if not resume:
                self.report.update(initial_door_rad=float(self.data.qpos[self.model.jnt_qposadr[self.door_id]]),
                    initial_rack_m=float(self.data.qpos[self.model.jnt_qposadr[self.rack_id]]),
                    tracked_objects=[DOOR,RACK])
                if abs(self.report['initial_door_rad'])>.005 or abs(self.report['initial_rack_m'])>.005:
                    raise RuntimeError('Oven test must start with door closed and rack in')
                self.tuck_for_navigation()
                self.task_navigate([3.7823,-1.10],False,face=np.pi/2)
            rack_resume = resume and getattr(self.args,'oven_resume_stage','door') == 'rack'
            self.review_phase='OPEN OVEN DOOR'
            self.select_component('door')
            if not rack_resume:
                if not resume:
                    self.grasp()
                else:
                    self.articulating=True
                    self.rebuild_planner()
                self.follow(1.10);self.release()
            self.report['released_open_door_rad']=self.angle()
            if self.angle()<1.05:
                raise RuntimeError('Released oven door did not remain open')
            self.review_phase='PULL UPPER RACK OUT'
            self.select_component('rack')
            if rack_resume:
                self.articulating=True
                self.rebuild_planner()
            else:
                self.grasp();self.follow(.15)
            self.report['rack_out_m']=self.angle()
            self.review_phase='PUSH UPPER RACK BACK'
            self.follow(0.);self.release()
            self.report['released_rack_m']=self.angle()
            self.review_phase='CLOSE OVEN DOOR'
            self.select_component('door');self.grasp();self.follow(0.);self.release()
            self.report['released_closed_door_rad']=self.angle()
            self.review_phase='OVEN CLOSED — TUCK AND RETREAT'
            self.tuck_for_navigation()
            start=self.base_pose().copy();self._oven_retreat=True
            self.task_navigate(start[:2]-.25*np.array([np.cos(start[2]),np.sin(start[2])]),False,face=float(start[2]))
            success=(abs(self.angle())<.05 and abs(self.report['released_rack_m'])<.01
                     and self.report['rack_out_m']>.13 and self.in_default_travel_posture(False))
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        finally:
            self.report.update(final_door_rad=float(self.data.qpos[self.model.jnt_qposadr[self.door_id]]),
                final_rack_m=float(self.data.qpos[self.model.jnt_qposadr[self.rack_id]]),
                success=bool(success),scope='one native oven upper rack and door; no object placement or heating')
            self.finish_run_outputs(success)
        return 0 if success else 1
