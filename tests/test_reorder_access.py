"""Door sequencing, simulator isolation, and rejection before live mutation."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import mujoco
import numpy as np
from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck
from cross_episode_sim.controller.transfer_chain import Transfer

class AccessTest(unittest.TestCase):
    def controller(self):
        c = PhysicalReorderCheck.__new__(PhysicalReorderCheck)
        from cross_episode_sim.robot.embodiment import embodiment_for
        c.embodiment = embodiment_for()
        c.profile = c.embodiment.profile
        c.args = SimpleNamespace(close_tolerance=3, defer_video=True,
                                 pickup_stance_x=-.89, pickup_stance_y=-1.02)
        c.review_phase = 'TEST'
        c.report = {'stages': []}
        c.trace = [{'stage': 'opening', 'old_cycle': True}]
        c.data = SimpleNamespace(time=0, joint=lambda name: SimpleNamespace(qpos=np.zeros(7)))
        c.attached = c.holding_loaf = False
        c.angle = lambda: 0.
        c.door_histories = {}
        c.test_angles = {'right': 0., 'left': 0.}
        c.door_angles = lambda: dict(c.test_angles)
        def bind(side):
            c.active_door = side
            c.opening_reference = c.door_histories.get(side, [])
        c.bind_door = bind
        return c

    def test_unusable_loaded_departure_is_rejected_before_arm_motion(self):
        c = self.controller()
        c.args.reverse_undock = .3
        c.tcp = lambda: np.eye(4)
        c.base_pose = lambda: np.zeros(3)
        c.plan_contact_path = Mock(return_value=[np.zeros(1)])
        c.preflight_carry_departure = Mock(side_effect=RuntimeError('blocked route'))
        c.mesh_contact_move = Mock(); c.carry_from_fridge = Mock(); c.record = Mock()
        with self.assertRaisesRegex(RuntimeError, 'No clear compact posture'):
            c.compact_and_carry_from_fridge()
        c.mesh_contact_move.assert_not_called()
        c.carry_from_fridge.assert_not_called()
        self.assertEqual(c.data.time, 0.)

    def test_carry_execution_failure_cannot_be_hidden_by_another_posture(self):
        c = self.controller()
        c.args.reverse_undock = .3
        c.tcp = lambda: np.eye(4)
        c.base_pose = lambda: np.zeros(3)
        c.plan_contact_path = Mock(return_value=[np.zeros(1)])
        c.preflight_carry_departure = Mock(return_value=.2)
        c.mesh_contact_move = Mock(); c.record = Mock()
        def fail():
            c.data.time += .002
            raise RuntimeError('Physical collision during carry')
        c.carry_from_fridge = Mock(side_effect=fail)
        with self.assertRaisesRegex(RuntimeError, 'Physical collision'):
            c.compact_and_carry_from_fridge()
        self.assertEqual(c.mesh_contact_move.call_count, 1)
        self.assertEqual(c.args.reverse_undock, .3)

    def alignment_controller(self):
        c = self.controller()
        c.data.joint = lambda name: SimpleNamespace(qpos=[0.])
        c.planner = SimpleNamespace(names=['right_arm_0'], plan_joints=Mock(),
                                    plan=Mock(side_effect=RuntimeError('blocked target')))
        c.nearby_ik = lambda pose, q: [pose[1, 3]]
        c.preflight_loaded_trajectory = Mock()
        c.grasp_relative = np.eye(4)
        c.preplanned_moves = {}
        c.record = Mock(); c.move = Mock()
        pose = np.eye(4); pose[:3, 3] = [.4, 2., 1.2]
        return c, pose, np.array([1., 1.5, 1.4])

    def test_entire_alignment_must_plan_before_any_waypoint_executes(self):
        c, pose, shelf = self.alignment_controller()
        calls = []
        def plan(q, target):
            calls.append(target)
            if len(calls) % 4 == 0:
                raise RuntimeError('fourth waypoint crosses door')
            return np.array([q, target])
        c.planner.plan_joints.side_effect = plan
        with self.assertRaisesRegex(RuntimeError, 'No collision-clear shelf alignment'):
            c.align_payload_for_shelf(pose, shelf)
        c.move.assert_not_called()
        self.assertEqual(c.preplanned_moves, {})

    def test_alignment_adjustment_executes_only_the_complete_accepted_plan(self):
        c, pose, shelf = self.alignment_controller()
        calls = []
        def plan(q, target):
            calls.append(target)
            if len(calls) == 4:
                raise RuntimeError('fourth waypoint crosses door')
            return np.array([q, target])
        c.planner.plan_joints.side_effect = plan
        c.move.side_effect = lambda *args: self.assertEqual(len(calls), 8)
        aligned = c.align_payload_for_shelf(pose, shelf)
        self.assertEqual(c.move.call_count, 4)
        np.testing.assert_allclose(aligned[:3, 3], [.45, 1.5, 1.2])
        np.testing.assert_allclose(pose[:3, 3], [.4, 2., 1.2])

    def test_closing_reference_contains_only_latest_opening(self):
        c = self.controller()
        c.open_fridge = lambda: c.trace.append({'stage': 'opening', 'new_cycle': True})
        c.open_for_access()
        self.assertEqual(c.opening_reference, [{'stage': 'opening', 'new_cycle': True}])
        self.assertEqual(c.review_phase, 'TEST')

    def test_only_right_door_opens_and_closes(self):
        c = self.controller(); closed = []
        def open_door():
            c.trace.append({'stage': 'opening', 'side': c.active_door})
            c.test_angles[c.active_door] = 75.
        c.open_fridge = open_door
        c.open_for_access('right')
        with self.assertRaisesRegex(RuntimeError, 'only the right'):
            c.open_for_access('left')
        self.assertNotIn('left', c.door_histories)
        def close_door():
            self.assertEqual(c.opening_reference[0]['side'], 'right')
            closed.append(c.active_door); c.test_angles[c.active_door] = 0.
        c.close_native_fridge = close_door
        c.close_after_access()
        self.assertEqual(closed, ['right'])
        self.assertEqual(c.test_angles, {'right': 0., 'left': 0.})

    def test_inspection_uses_only_right_compartment(self):
        c = self.controller(); calls = []
        c.observation_index = 0; c.history_cycle = 1
        c.fridge = 'fridge'; c.objects = (); c.assignment = lambda: {}
        c.open_for_access = lambda side: (calls.append(('open',side)), c.test_angles.update({side:75.}))
        c.dock = Mock(); c.close_after_access = lambda: calls.append(('close','right'))
        c.model = SimpleNamespace(nsite=2,site=lambda i: SimpleNamespace(name=f'FridgeBodyMeshf017e276Receptacle{i+2}_{i+2}'))
        c.data.site_xpos = np.zeros((2,3))
        c.tick = Mock(); c.gaze_error_deg = lambda: 0.
        c.record = lambda **kw: calls.append(('inspect',kw['inspected_compartment']))
        c.observe(('fridge',))
        self.assertEqual(calls,[('open','right'),('inspect','right'),('close','right')])
        self.assertEqual(c.review_phase,'REVISIT AFTER DYNAMIC CHANGE 1')

    def test_left_object_is_not_counted_as_observed_through_closed_door(self):
        c = self.controller(); c.observation_index = 0; c.fridge = 'fridge'; c.objects = ('egg',)
        c.assignment = lambda: {'egg': 'fridge'}
        c.fridge_side_of = lambda obj: 'left'; c.open_for_access = Mock()
        with self.assertRaisesRegex(RuntimeError, 'outside the enabled right'):
            c.observe(('fridge',))
        c.open_for_access.assert_not_called()

    def test_disabled_open_left_door_is_rejected_without_operating_it(self):
        c = self.controller(); c.test_angles['left'] = 75.
        c.close_native_fridge = Mock(); c.open_fridge = Mock()
        with self.assertRaisesRegex(RuntimeError, 'left fridge door must remain closed'):
            c.close_after_access()
        with self.assertRaisesRegex(RuntimeError, 'left fridge door must remain closed'):
            c.open_for_access('right')
        c.close_native_fridge.assert_not_called(); c.open_fridge.assert_not_called()

    def test_full_right_compartment_does_not_select_clear_left_slot(self):
        from cross_episode_sim.controller.reorder_chain import EGG
        c = self.controller(); c.object_name = EGG; c.fridge = 'fridge'
        c.bread_pose = lambda: np.eye(4)
        c.bread_vertices = lambda: np.array([[-.01,-.01,-.01],[.01,.01,.01]])
        c.movable_scene_objects = ('blocker',); c.fridge_bids = {1}
        c.data.site_xpos = np.array([[0.,-.5,1.],[0.,.5,1.]])
        c.data.body = lambda name: SimpleNamespace(xpos=np.array([0.,0.,1.]))
        c.model = SimpleNamespace(nsite=2,site=lambda i: SimpleNamespace(name=f'FridgeBodyMeshf017e276Receptacle{i+2}_{i+2}'),
                                 site_size=np.array([[.3,.1,.2],[.3,.1,.2]]),geom_bodyid=np.array([1]))
        def ray(*args):
            args[-1][0] = 0
            return .04
        with patch('cross_episode_sim.controller.reorder_chain.mujoco.mj_ray',side_effect=ray), patch(
                'cross_episode_sim.controller.reorder_chain.collision_mesh',
                return_value=(np.array([[-.2,-.8,.5],[.2,-.2,1.5]]),None)):
            with self.assertRaisesRegex(RuntimeError, 'Right fridge compartment is full'):
                c.select_fridge_slot()
        self.assertFalse(hasattr(c,'manipulation_side'))

    def test_open_before_travelling_to_pickup(self):
        c = self.controller(); calls = []
        c.source, c.destination, c.fridge = 'counter', 'fridge', 'fridge'
        c.manipulation_side = 'right'
        c.select_fridge_slot = Mock()
        c.open_for_access = lambda side: calls.append('open')
        c.dock = lambda *a: calls.append('pickup_navigation')
        c.prepare_pickup()
        self.assertEqual(calls, ['open', 'pickup_navigation'])
        self.assertTrue(c.annotation_mesh_contact_approach)

    def test_close_rejects_held_object_and_unclosed_door(self):
        c = self.controller(); c.attached = True
        c.close_native_fridge = Mock()
        with self.assertRaises(RuntimeError): c.close_after_access()
        c.close_native_fridge.assert_not_called()
        c.attached = False; c.test_angles['right'] = 10.
        with self.assertRaises(RuntimeError): c.close_after_access()
        self.assertNotIn('door_cycles', c.report)

    def test_placement_arm_folds_before_closure_departure(self):
        c = self.controller(); calls = []
        c.tuck_arm = lambda: calls.append('fold_empty_arm')
        c.close_after_access = lambda: calls.append('depart_and_close')
        c.after_placement()
        self.assertEqual(calls, ['fold_empty_arm', 'depart_and_close'])

    def test_placement_retraction_recovers_only_an_unexecuted_plan_failure(self):
        c = self.controller(); calls = []
        def tuck():
            calls.append('tuck')
            if calls == ['tuck']:
                raise RuntimeError('planner found no path')
        c.tuck_arm = tuck
        c.record = Mock()
        c.retract_empty_placement_arm = lambda: calls.append('retract')
        c.close_after_access = lambda: calls.append('close')
        c.after_placement()
        self.assertEqual(calls, ['tuck', 'retract', 'tuck', 'close'])

    def test_placement_retraction_cannot_hide_a_physical_execution_failure(self):
        c = self.controller()
        def tuck():
            c.data.time += .002
            raise RuntimeError('physical collision')
        c.tuck_arm = tuck
        c.retract_empty_placement_arm = Mock()
        c.close_after_access = Mock()
        with self.assertRaisesRegex(RuntimeError, 'physical collision'):
            c.after_placement()
        c.retract_empty_placement_arm.assert_not_called()
        c.close_after_access.assert_not_called()

    def test_held_object_cannot_start_closure_preparation(self):
        c = self.controller(); c.holding_loaf = True
        c.tuck_arm = Mock(); c.close_after_access = Mock()
        with self.assertRaises(RuntimeError): c.after_placement()
        c.tuck_arm.assert_not_called(); c.close_after_access.assert_not_called()

    def transfer_controller(self):
        c = self.controller(); c.fridge = 'fridge'; c.counter = 'counter'
        c.select_object = Mock(); c.support = SimpleNamespace(reset=Mock()); c.record = Mock()
        def open_door():
            c.trace.append({'stage':'opening'}); c.test_angles['right'] = 75.
        c.open_fridge = open_door
        c.close_native_fridge = lambda: c.test_angles.update(right=0.)
        return c

    def test_each_transfer_direction_requires_its_own_door_cycle(self):
        c = self.transfer_controller()
        def execute():
            c.open_for_access('right'); c.close_after_access(); c.report['success'] = True
        c.execute_transfer = execute
        for source,destination in (('counter','fridge'),('fridge','counter')):
            c.transfer(Transfer('egg',source,destination))
        self.assertEqual([(e['action'],e['side']) for e in c.report['door_cycles']],
                         [('open','right'),('close','right')]*2)
        self.assertEqual(c.record.call_count,2)
        self.assertFalse(c.active_transfer)

    def test_consecutive_transfers_share_opening_and_close_after_second(self):
        for first_source in ('counter', 'fridge'):
            with self.subTest(first_source=first_source):
                c = self.transfer_controller(); intermediate = []
                c.tuck_arm = Mock()
                def execute():
                    c.open_for_access('right')
                    # Both shelf and counter release paths use this access finish.
                    c.after_placement()
                    c.report['success'] = True
                c.execute_transfer = execute
                with c.transfer_group(2):
                    c.transfer(Transfer('egg',first_source,
                                        'fridge' if first_source=='counter' else 'counter'))
                    self.assertEqual(c.test_angles, {'right':75., 'left':0.})
                    self.assertEqual(len(c.report['door_cycles']), 1)
                    opening_reference = c.opening_reference
                    intermediate.append(c.record.call_args.kwargs)
                    c.transfer(Transfer('potato','counter','fridge'))
                    self.assertIs(c.opening_reference, opening_reference)
                self.assertEqual([(e['action'],e['side']) for e in c.report['door_cycles']],
                                 [('open','right'),('close','right')])
                self.assertEqual(c.test_angles, {'right':0., 'left':0.})
                self.assertTrue(intermediate[0]['door_left_open_for_next_transfer'])
                self.assertTrue(intermediate[0]['door_access_verified'])
                self.assertFalse(intermediate[0]['door_cycle_verified'])
                self.assertTrue(c.report['transfer_groups'][0]['door_cycle_verified'])
                self.assertEqual(c.transfer_group_remaining, 0)
                self.assertEqual(c.tuck_arm.call_count, 2)

    def test_group_rejects_redundant_closing_after_first_transfer(self):
        c = self.transfer_controller()
        def execute():
            c.open_for_access('right'); c.close_after_access(); c.report['success'] = True
        c.execute_transfer = execute
        with self.assertRaisesRegex(RuntimeError,'do not match batch boundary'):
            with c.transfer_group(2):
                c.transfer(Transfer('egg','counter','fridge'))
        c.record.assert_not_called()
        self.assertFalse(c.transfer_group_size)
        self.assertNotIn('transfer_groups',c.report)

    def test_shared_open_door_does_not_allow_revisit_or_change_between_transfers(self):
        c = self.transfer_controller()
        with c.transfer_group(2):
            with self.assertRaisesRegex(RuntimeError,'Revisits are forbidden'):
                c.observe(('counter','fridge'))
            with self.assertRaisesRegex(RuntimeError,'Dynamic changes are forbidden'):
                c.intervene({'egg':'counter'})
            # Finish the actual declared transfers after checking rejected calls.
            def execute():
                c.open_for_access('right'); c.finish_transfer_access(); c.report['success'] = True
            c.execute_transfer = execute
            c.transfer(Transfer('egg','counter','fridge'))
            c.transfer(Transfer('potato','counter','fridge'))

    def test_failed_second_transfer_does_not_attempt_door_motion_or_mark_group_success(self):
        c = self.transfer_controller()
        def execute_first():
            c.open_for_access('right'); c.finish_transfer_access(); c.report['success'] = True
        c.execute_transfer = execute_first
        with self.assertRaisesRegex(RuntimeError,'physical grasp failed'):
            with c.transfer_group(2):
                c.transfer(Transfer('egg','counter','fridge'))
                c.close_native_fridge = Mock()
                c.execute_transfer = Mock(side_effect=RuntimeError('physical grasp failed'))
                c.transfer(Transfer('potato','counter','fridge'))
        c.close_native_fridge.assert_not_called()
        self.assertFalse(c.transfer_group_size)
        self.assertFalse(c.active_transfer)
        self.assertNotIn('transfer_groups',c.report)
        self.assertFalse(c.report['success'])

    def test_group_cannot_silently_end_early_or_accept_extra_transfer(self):
        c = self.transfer_controller()
        with self.assertRaisesRegex(RuntimeError,'before all transfers finished'):
            with c.transfer_group(2):
                pass
        def execute():
            c.open_for_access('right'); c.finish_transfer_access(); c.report['success'] = True
        c.execute_transfer = execute
        with self.assertRaisesRegex(RuntimeError,'already completed'):
            with c.transfer_group(1):
                c.transfer(Transfer('egg','counter','fridge'))
                c.transfer(Transfer('potato','counter','fridge'))

    def test_closed_door_between_transfers_is_rejected_before_motion(self):
        c = self.transfer_controller()
        def execute():
            c.open_for_access('right'); c.finish_transfer_access(); c.report['success'] = True
        c.execute_transfer = execute
        with self.assertRaisesRegex(RuntimeError,'requires the right door to remain open'):
            with c.transfer_group(2):
                c.transfer(Transfer('egg','counter','fridge'))
                c.test_angles['right'] = 0.
                c.execute_transfer = Mock()
                c.transfer(Transfer('potato','counter','fridge'))
        c.execute_transfer.assert_not_called()

    def test_transfer_cannot_succeed_without_door_cycle(self):
        c = self.transfer_controller()
        c.execute_transfer = lambda: c.report.update(success=True)
        with self.assertRaisesRegex(RuntimeError,'do not match batch boundary'):
            c.transfer(Transfer('egg','counter','fridge'))
        c.record.assert_not_called()
        self.assertFalse(c.report['success']); self.assertFalse(c.active_transfer)

    def test_revisit_is_rejected_inside_locomanip(self):
        c = self.transfer_controller()
        c.execute_transfer = lambda: c.observe(('counter','fridge'))
        with self.assertRaisesRegex(RuntimeError,'Revisits are forbidden during locomanip'):
            c.transfer(Transfer('egg','counter','fridge'))
        c.record.assert_not_called()
        self.assertFalse(c.active_transfer)

    def test_transfer_rejects_an_already_open_fridge(self):
        c = self.transfer_controller(); c.test_angles['right'] = 75.
        c.execute_transfer = Mock()
        with self.assertRaisesRegex(RuntimeError,'must start with the fridge closed'):
            c.transfer(Transfer('egg','counter','fridge'))
        c.execute_transfer.assert_not_called()


    def test_actual_robot_self_collision_is_rejected(self):
        c = self.controller()
        c.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="robot_0/link_torso_2"><freejoint/><geom size=".1" mass="1"/></body><body name="robot_0/link_torso_4" pos=".15 0 0"><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
        d = mujoco.MjData(c.model); mujoco.mj_forward(c.model, d)
        c.robot_bids = {1, 2}
        self.assertGreater(c.robot_self_penetration(d), .0005)
        with self.assertRaisesRegex(RuntimeError, 'Robot self collision'):
            c.validate_grasp_probe(d)




    def change_controller(self):
        c = self.controller()
        c.counter, c.fridge = 'counter', 'fridge'
        c.receptacles = (c.counter, c.fridge); c.objects = ('egg', 'potato', 'shaker')
        c.assignment = lambda: {'egg': 'fridge', 'potato': 'counter', 'shaker': 'counter'}
        c.demonstrated_placements = {'egg': {'counter': {
            'pose': [0., 0., 1., 1., 0., 0., 0.], 'witness_index': 0, 'evidence': 'successful_pickup'}}}
        c.report['successful_transfer_witnesses'] = []
        c.report['intervention_eligibility'] = []
        c.dock = Mock(); c.record = Mock()
        c.check_change_occupancy = Mock(); c.apply_demonstrated_change = Mock()
        return c

    def test_unmanipulated_object_rejected_before_motion_or_mutation(self):
        c = self.change_controller()
        with self.assertRaisesRegex(RuntimeError, 'no successful physical transfer'):
            c.intervene({'shaker': 'fridge'})
        c.dock.assert_not_called(); c.apply_demonstrated_change.assert_not_called()

    def test_manipulated_object_cannot_use_an_undemonstrated_destination(self):
        c = self.change_controller()
        c.demonstrated_placements['potato'] = {'counter': {'pose': [0.]*7}}
        with self.assertRaisesRegex(RuntimeError, 'no successful physical transfer'):
            c.intervene({'potato': 'fridge'})
        c.apply_demonstrated_change.assert_not_called()

    def test_occupied_demonstrated_slot_rejected_before_live_mutation(self):
        c = self.change_controller()
        c.check_change_occupancy.side_effect = RuntimeError('slot is now occupied')
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            c.intervene({'egg': 'counter'})
        c.apply_demonstrated_change.assert_not_called()

    def test_eligible_change_does_not_rehearse_robot_actions(self):
        c = self.change_controller()
        state = c.assignment(); c.assignment = lambda: dict(state)
        c.apply_demonstrated_change.side_effect = lambda selected: state.update(egg='counter')
        c.observe = Mock(side_effect=AssertionError('No hidden revisit'))
        c.transfer = Mock(side_effect=AssertionError('No duplicate work'))
        c.intervene({'egg': 'counter'})
        c.observe.assert_not_called(); c.transfer.assert_not_called()
        c.check_change_occupancy.assert_called_once()
        self.assertFalse(c.report['intervention_eligibility'][0]['rehearsal_executed'])
        self.assertEqual(c.report['intervention_eligibility'][0]['placements']['egg']['witness_index'], 0)

    def test_witness_records_both_real_transfer_endpoints(self):
        c = self.change_controller(); c.transfer_source_pose = np.array([1.,2.,3.,1.,0.,0.,0.])
        c.remember_successful_transfer(Transfer('potato', 'counter', 'fridge'))
        self.assertEqual(c.demonstrated_placements['potato']['counter']['pose'], c.transfer_source_pose.tolist())
        self.assertEqual(c.demonstrated_placements['potato']['fridge']['pose'], [0.]*7)
        self.assertEqual(len(c.report['successful_transfer_witnesses']), 1)

if __name__ == '__main__': unittest.main()
