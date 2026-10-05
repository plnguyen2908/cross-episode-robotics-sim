"""Protocol tests: stale state, false success and extra receptacles are rejected."""
import unittest
from contextlib import contextmanager
from cross_episode_sim.controller.transfer_chain import TwoReceptacleChain

class Executor:
    def __init__(self):
        self.state = {'egg': 'counter', 'potato': 'counter'}
        self.transfers = []
        self.calls = []
        self.groups = []
        self.group_sizes = []
    @contextmanager
    def transfer_group(self, count):
        self.group_sizes.append(count)
        before = len(self.transfers)
        yield
        self.groups.append(self.transfers[before:])

    def assignment(self):
        return dict(self.state)
    def observe(self, receptacles):
        self.calls.append('observe')
        return self.assignment()
    def transfer(self, move):
        self.calls.append('transfer')
        self.transfers.append(move)
        self.state[move.object_name] = move.destination
    def intervene(self, moves):
        self.calls.append('change')
        self.state.update(moves)

class ChainTest(unittest.TestCase):
    def make(self):
        executor = Executor()
        return executor, TwoReceptacleChain(('counter', 'fridge'), ('egg', 'potato'), executor)
    def test_three_native_objects_initially_split_across_tables(self):
        executor = Executor()
        executor.state = {'remote_a': 'table_a', 'remote_b': 'table_b', 'phone': 'table_a'}
        chain = TwoReceptacleChain(('table_a', 'table_b'), tuple(executor.state), executor)
        target = chain.run_history(2)
        self.assertEqual(executor.state, target)
        self.assertEqual(chain.manipulated, set(executor.state))
        self.assertEqual(len([e for e in chain.events if e['kind'] == 'intervention']), 2)
        self.assertEqual({(t.source, t.destination) for t in executor.transfers},
                         {('table_a', 'table_b'), ('table_b', 'table_a')})
        self.assertEqual(chain.snapshots[0]['assignment'], target)
        self.assertEqual(executor.calls[:4], ['transfer', 'transfer', 'change', 'observe'])

    def test_full_protocol(self):
        executor, chain = self.make()
        target = chain.run()
        self.assertEqual(executor.state, target)
        self.assertEqual(len(executor.transfers), 4)
        self.assertEqual(len(chain.snapshots), 3)
        self.assertEqual(chain.snapshots[1]['assignment'], {'egg': 'fridge', 'potato': 'fridge'})
        self.assertEqual(chain.snapshots[2]['assignment'], {'egg': 'counter', 'potato': 'counter'})
    def test_history_restores_older_post_change_snapshot(self):
        executor, chain = self.make()
        target = chain.run_history(2)
        self.assertEqual(target, {'egg': 'fridge', 'potato': 'counter'})
        changes = [i for i,e in enumerate(chain.events) if e['kind'] == 'intervention']
        self.assertEqual(len(changes), 2)
        previous = 0
        for i in changes:
            self.assertEqual(sum(e['kind'] == 'work' for e in chain.events[previous:i]), 2)
            previous = i + 1
        request = next(e for e in chain.events if e['kind'] == 'restore_request')
        self.assertEqual(request['source_label'], 'after change 1')
        self.assertNotEqual(request['source_snapshot'], len(chain.snapshots)-1)
        self.assertEqual(chain.events[-1]['n_restored'], 1)
        self.assertEqual(executor.transfers[-1].object_name, 'egg')
        self.assertEqual(executor.transfers[-1].source, 'counter')
        self.assertEqual(executor.transfers[-1].destination, 'fridge')
        self.assertEqual(executor.state, target)

    def test_ordinary_work_includes_both_physical_directions(self):
        executor, chain = self.make()
        chain.run_history(2)
        self.assertEqual([(m.object_name,m.source,m.destination) for m in executor.transfers[:4]], [
            ('egg','counter','fridge'), ('potato','counter','fridge'),
            ('egg','fridge','counter'), ('potato','counter','fridge')])
        self.assertEqual(chain.snapshots[0]['assignment'], {'egg':'fridge','potato':'counter'})
        self.assertEqual(chain.snapshots[1]['assignment'], {'egg':'counter','potato':'counter'})
        changes = [e for e in chain.events if e['kind']=='intervention']
        self.assertTrue(all(e['moves']==[{'object_name':'potato','source':'fridge','destination':'counter'}]
                            for e in changes))

    def test_history_visits_only_after_dynamic_changes(self):
        executor, chain = self.make()
        chain.run_history(2)
        self.assertEqual(executor.calls, [
            'transfer', 'transfer', 'change', 'observe',
            'transfer', 'transfer', 'change', 'observe', 'transfer'])
        self.assertEqual([s['label'] for s in chain.snapshots],
                         ['after change 1', 'after change 2'])
        request = next(e for e in chain.events if e['kind'] == 'restore_request')
        self.assertEqual(request['source_snapshot'], 0)
        self.assertEqual(chain.events[0]['kind'], 'work')

    def test_history_groups_only_consecutive_work_and_restoration(self):
        executor, chain = self.make()
        chain.run_history(2)
        self.assertEqual(executor.group_sizes, [2, 2, 1])
        self.assertEqual([[m.object_name for m in group] for group in executor.groups],
                         [['egg', 'potato'], ['egg', 'potato'], ['egg']])

    def test_three_change_cycles_keep_two_transfers_per_change(self):
        executor, chain = self.make()
        chain.run_history(3)
        self.assertEqual(sum(e['kind']=='intervention' for e in chain.events), 3)
        self.assertEqual(len(executor.transfers), 7)
        self.assertEqual(chain.events[-1]['source_label'], 'after change 1')

    def test_failed_grasp_does_not_advance(self):
        executor, chain = self.make()
        executor.transfer = lambda move: None
        with self.assertRaises(RuntimeError):
            chain.work('egg', 'fridge')
        self.assertEqual(chain.events, [])
    def test_intervention_cannot_leave_pair(self):
        executor, chain = self.make()
        with self.assertRaises(ValueError):
            chain.intervene({'egg': 'oven'})
        self.assertEqual(executor.state['egg'], 'counter')
    def test_incomplete_observation_rejected(self):
        executor, chain = self.make()
        executor.observe = lambda pair: {'egg': 'counter'}
        with self.assertRaises(ValueError):
            chain.explore()
    def test_trivial_restoration_rejected(self):
        _, chain = self.make()
        snapshot = chain.explore()
        with self.assertRaises(ValueError):
            chain.restore(snapshot)
    def test_bidirectional_protocol(self):
        executor, chain = self.make()
        chain.work('egg', 'fridge')
        chain.work('egg', 'counter')
        self.assertEqual(executor.state['egg'], 'counter')

    def test_three_objects_are_physically_used_before_becoming_exchangeable(self):
        executor = Executor(); executor.state['shaker'] = 'counter'
        chain = TwoReceptacleChain(('counter', 'fridge'), tuple(executor.state), executor)
        target = chain.run_history(2)
        self.assertEqual([(m.object_name,m.source,m.destination) for m in executor.transfers[:4]], [
            ('egg','counter','fridge'), ('potato','counter','fridge'),
            ('egg','fridge','counter'), ('shaker','counter','fridge')])
        self.assertEqual(executor.group_sizes, [2, 2, 2])
        self.assertEqual(target, {'egg':'fridge','potato':'counter','shaker':'counter'})
        self.assertEqual(executor.state, target)
        changes = [e for e in chain.events if e['kind'] == 'intervention']
        self.assertEqual({e['object_name'] for e in changes[1]['moves']}, {'potato','shaker'})
        seen = set()
        for event in chain.events:
            if event['kind'] == 'work': seen.add(event['object_name'])
            if event['kind'] == 'intervention':
                self.assertTrue({e['object_name'] for e in event['moves']} <= seen)

    def test_four_objects_use_the_four_primary_transfers(self):
        executor = Executor(); executor.state.update(salt='counter', pepper='counter')
        chain = TwoReceptacleChain(('counter', 'fridge'), tuple(executor.state), executor)
        target = chain.run_history(2)
        self.assertEqual([move.object_name for move in executor.transfers[:4]],
                         ['egg', 'potato', 'salt', 'pepper'])
        self.assertEqual(chain.manipulated, set(executor.state))
        self.assertEqual(executor.state, target)

    def test_five_objects_require_three_cycles(self):
        executor = Executor(); executor.state.update(salt='counter', pepper='counter', mug='counter')
        chain = TwoReceptacleChain(('counter', 'fridge'), tuple(executor.state), executor)
        with self.assertRaisesRegex(ValueError, 'Two transfers per cycle'):
            chain.run_history(2)
        self.assertEqual(executor.calls, [])
        target = chain.run_history(3)
        self.assertEqual(chain.manipulated, set(executor.state))
        self.assertEqual(executor.state, target)

    def test_failed_transfer_never_qualifies_for_dynamic_change(self):
        executor, chain = self.make()
        executor.transfer = lambda move: None
        with self.assertRaises(RuntimeError): chain.work('egg', 'fridge')
        with self.assertRaisesRegex(ValueError, 'successful physical transfer'):
            chain.intervene({'egg': 'fridge'})
        self.assertEqual(chain.manipulated, set())
        self.assertNotIn('change', executor.calls)

if __name__ == '__main__':
    unittest.main()
