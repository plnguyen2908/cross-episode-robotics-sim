"""Physical fingertip pressing of native RoboCasa microwave Start and Stop."""
import traceback
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.fixtures.faucet import FaucetTest

DOOR = 'microwave_main_group_door'

def next_state(on, start, stop, door_open):
    """RoboCasa Microwave.update_state contact-triggered latch semantics."""
    if door_open:
        return False
    if on and stop:
        return False
    if not on and start:
        return True
    return on

class MicrowaveButtonTest(CabinetDoorTest):
    fixture_body = DOOR
    fixture_joint = 'microwave_main_group_microjoint'
    handle_body = 'microwave_main_group_main'
    handle_geometry = 'microwave_main_group_start_button'
    video_filename = 'microwave_start_stop.mp4'
    load_world = FaucetTest.load_world
    def plan_route(self, goal, carrying, face=None):
        if getattr(self, '_faucet_retreat', False):
            from cross_episode_sim.controller.manipulation import TableReorder
            self._pickup_pre_nav_undock = .25
            return TableReorder.plan_route(self, goal, carrying, face)
        return super().plan_route(goal, carrying, face)

    def __init__(self, args, selection):
        self.turned_on = False
        self.state_events = []
        super().__init__(args, selection)
        self.handle_bids = set()
        self.button_ids = {name: self.model.geom('microwave_main_group_'+name+'_button').id
                           for name in ('start', 'stop')}
        self.pad_bids = {self.model.body('robot_0/gripper/'+side+'_pad').id
                         for side in ('left', 'right')}
        self.report.update(task='physical microwave Start, release, Stop, release',
                           state_events=self.state_events)

    def object_label(self):
        return 'microwave buttons'

    def interaction_label(self):
        return self.object_label()

    def kitchen_world_geoms(self):
        return CrossRoomManipulation.kitchen_world_geoms(self)

    def contacts(self, name):
        force = 0.
        touched = False
        for i, c in enumerate(self.data.contact):
            gids = (c.geom1, c.geom2)
            if self.button_ids[name] not in gids:
                continue
            other = gids[1] if gids[0] == self.button_ids[name] else gids[0]
            body = self.model.body(self.model.geom_bodyid[other]).name or ''
            if body.startswith('robot_0/gripper/'):
                touched = True
                if self.model.geom_bodyid[other] in self.pad_bids:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, f)
                    force += max(0., float(f[0]))
        return touched, force

    def before_step(self):
        super().before_step()
        if not hasattr(self, 'button_ids'):
            return
        start, sf = self.contacts('start')
        stop, tf = self.contacts('stop')
        state = next_state(self.turned_on, start, stop, abs(self.angle())/(np.pi/2) >= .90)
        if state != self.turned_on:
            event = dict(time=float(self.data.time), turned_on=state,
                         start_contact=start, stop_contact=stop,
                         start_pad_force_n=sf, stop_pad_force_n=tf)
            self.state_events.append(event)
            self.record(microwave_state_transition=event)
        self.turned_on = state

    def button_pose(self, name):
        from cross_episode_sim.controller.base import geom_box
        R = Rotation.from_euler('x', 35, degrees=True).as_matrix() @ np.array([[0,1,0],[0,0,1],[1,0,0]])
        corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        vertices = []
        pad = self.model.body('robot_0/gripper/left_pad').id
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] != pad or not self.model.geom_contype[gid]:
                continue
            c, h = geom_box(self.model, gid)
            vertices.extend(self.data.geom_xpos[gid]+(c+corners*h)@self.data.geom_xmat[gid].reshape(3,3).T)
        tcp = self.tcp()
        offsets = (np.asarray(vertices)-tcp[:3,3])@tcp[:3,:3]@R.T
        tip = offsets[offsets[:,1] > offsets[:,1].max()-1e-5].mean(axis=0)
        gid = self.button_ids[name]
        surface = self.data.geom_xpos[gid].copy()
        surface[1] -= self.model.geom_size[gid,1]
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = surface-tip
        return pose

    def move_button(self, stage, pose, seed=None):
        current = np.array([float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names])
        q = self.nearby_ik(pose, current if seed is None else seed,
                           trust_radius=6. if seed is not None else .5,
                           preserve_self_clearance=True)
        self.validate_door_path([q], q)
        self.load_world()
        blockers = [b.name for b in self.planner_world_boxes if self.planner.world_clearance(q, [b]) < 0]
        if blockers:
            self._precise_contact_boxes = getattr(self, '_precise_contact_boxes', set()) | set(blockers)
            self.load_world()
        path = self.planner.plan_joints(current, q)
        self._door_plans = {stage: path}
        super().plan_and_move(stage, pose)
        return q

    def press(self, name):
        self.review_phase = 'PRESS MICROWAVE '+name.upper()
        pose = self.button_pose(name)
        pre = pose.copy(); pre[1,3] -= .06
        pre_q = self.move_button('approach '+name+' button', pre,
                         np.array([0., -.07, .02, -1.17, .08, 3.28, -.085]))
        maximum_force = 0.
        for offset in (-.02, -.006, -.002, -.0005, .0002, .0006):
            target = pose.copy(); target[1,3] += offset
            self.move_button('press '+name+' button', target)
            self.tick(.15)
            touched, force = self.contacts(name)
            maximum_force = max(maximum_force, force)
            self.record(button=name, physical_contact=touched, pad_force_n=force,
                        turned_on=self.turned_on)
            if touched and force > .05 and self.turned_on == (name == 'start'):
                break
        else:
            raise RuntimeError('No force-bearing fingertip press on '+name)
        self.move_button('withdraw from '+name+' button', pre, seed=pre_q)
        self.tick(.5)
        if self.contacts(name)[0] or self.turned_on != (name == 'start'):
            raise RuntimeError('Microwave release state failed after '+name)
        self.report[name] = dict(pad_force_n=maximum_force, state_after_release=self.turned_on)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        target = self.data.geom_xpos[self.handle_geom]
        self.cameras[0].lookat[:] = [target[0]-.15,target[1]-.3,.95]
        self.cameras[0].distance = 2.4
        self.cameras[0].azimuth = 120.
        self.cameras[0].elevation = -15.
        self.cameras[1].lookat[:] = target
        self.cameras[1].distance = .5
        self.cameras[1].azimuth = 100.
        self.cameras[1].elevation = -10.

    def render_video_frame(self, label, time):
        on = False
        for event in self.state_events:
            if event['time'] <= time:
                on = event['turned_on']
        return super().render_video_frame(label+' | microwave '+('ON' if on else 'OFF'), time)

    def run_test(self):
        completed = False
        try:
            self.tick(1.)
            if abs(self.angle()) > .01:
                raise RuntimeError('Microwave must start closed')
            self.tuck_for_navigation()
            self.task_navigate([3.41420304,-1.0], False, face=np.pi/2)
            self.embodiment.command_gripper(self.data, self.profile.gripper_close)
            self.tick(1.)
            self.rebuild_planner()
            self.press('start')
            self.press('stop')
            self.tuck_for_navigation()
            start = self.base_pose().copy()
            self._faucet_retreat = True
            self.task_navigate(start[:2]-.25*np.array([np.cos(start[2]),np.sin(start[2])]), False, face=float(start[2]))
            completed = not self.turned_on and self.in_default_travel_posture(False)
        except Exception as exc:
            self.report.update(error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            self.report.update(success=bool(completed), final_turned_on=self.turned_on,
                scope='native fixed button geoms and contact-driven RoboCasa state; no heating simulation')
            self.finish_run_outputs(completed)
        return 0 if completed else 1
