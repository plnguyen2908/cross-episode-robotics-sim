"""Check latch parity with the vendored native fixture implementation."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from cross_episode_sim.fixtures.microwave_button import next_state
from cross_episode_sim.paths import robocasa_dir

class MicrowaveStateTest(unittest.TestCase):
    def test_native_state_parity(self):
        source = (robocasa_dir()/'models/fixtures/microwave.py').read_text()
        tree = ast.parse(source)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update_state')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<native microwave>', 'exec'), namespace)
        for on, start, stop, opened in itertools.product((False, True), repeat=4):
            fixture = SimpleNamespace(_turned_on=on, name='test', is_open=lambda env: opened)
            env = SimpleNamespace(robots=[SimpleNamespace(gripper={'right': object()})],
                                  check_contact=lambda gripper, name: start if name.endswith('start_button') else stop)
            namespace['update_state'](fixture, env)
            self.assertEqual(next_state(on, start, stop, opened), fixture._turned_on)

    def test_release_preserves_state(self):
        self.assertTrue(next_state(True, False, False, False))
        self.assertFalse(next_state(False, False, False, False))

if __name__ == '__main__':
    unittest.main()
