"""Regression checks for unsafe/invalid transitions in the simulated cycle."""
import unittest

from cross_episode_sim.tasks.coffee.brew import MachineCycle


class MachineCycleTest(unittest.TestCase):
    def test_start_requires_each_prerequisite_and_warmup(self):
        machine=MachineCycle()
        with self.assertRaises(RuntimeError):machine.start(0,True,True,True)
        machine.power_on(0)
        with self.assertRaises(RuntimeError):machine.start(1.9,True,True,True)
        for physical in [(False,True,True),(True,False,True),(True,True,False)]:
            with self.assertRaises(RuntimeError):machine.start(2,*physical)
        self.assertIsNone(machine.started_at)
        machine.start(2,True,True,True)
        self.assertEqual(machine.started_at,2)

    def test_completion_waits_for_elapsed_time_and_button_release(self):
        machine=MachineCycle();machine.power_on(0);machine.start(2,True,True,True)
        self.assertIsNone(machine.update(9.99,True,True,True,True))
        self.assertIsNone(machine.update(10,True,True,True,False))
        self.assertEqual(machine.update(10.1,True,True,True,True),'brew_complete')
        self.assertTrue(machine.completed)
        with self.assertRaises(RuntimeError):machine.start(12,True,True,True)

    def test_losing_cup_lock_or_grounds_aborts_without_later_completion(self):
        for physical in [(False,True,True),(True,False,True),(True,True,False)]:
            machine=MachineCycle();machine.power_on(0);machine.start(2,True,True,True)
            self.assertEqual(machine.update(4,*physical,True),'brew_aborted')
            self.assertIsNone(machine.update(20,True,True,True,True))
            self.assertFalse(machine.completed)
            self.assertTrue(machine.aborted)


if __name__=='__main__':unittest.main()
