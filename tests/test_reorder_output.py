"""Full-task completion, success/failure videos, and encoder cleanup regressions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck

class OutputTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        c = PhysicalReorderCheck.__new__(PhysicalReorderCheck)
        c.output = Path(self.folder.name)
        c.args = SimpleNamespace(resume_dir=None)
        c.report = {'success': True, 'error': 'planner failed'}
        c.trace = [{'stage': 'partial work'}]
        c.writer, c.head_writer, c.renderer = Mock(), Mock(), Mock()
        c.finalize_report = Mock()
        c.render_deferred_video = Mock()
        self.c = c

    def test_failure_renders_saved_trace_without_claiming_success(self):
        c = self.c; c.finish_run_outputs(False)
        c.render_deferred_video.assert_called_once()
        self.assertFalse(json.loads((c.output/'report.json').read_text())['success'])
        self.assertEqual(json.loads((c.output/'trace.json').read_text()), c.trace)
        self.assertEqual(c.report['video_status'], 'complete')
        self.assertEqual(c.report['video_outcome'], 'failure')
        self.assertFalse(json.loads((c.output/'physics_result.json').read_text())['physics_success'])
        for resource in (c.writer,c.head_writer,c.renderer): resource.close.assert_called()

    def test_success_publishes_physics_result_before_rendering(self):
        c = self.c
        def render():
            result = json.loads((c.output/'physics_result.json').read_text())
            self.assertTrue(result['physics_success'])
            self.assertEqual(result['video_status'], 'pending')
            self.assertTrue((c.output/'trace.json').exists())
        c.render_deferred_video.side_effect = render
        c.finish_run_outputs(True)
        c.render_deferred_video.assert_called_once()
        self.assertEqual(json.loads((c.output/'report.json').read_text())['video_status'], 'complete')

    def test_failure_signal_survives_trace_serialization_error(self):
        c = self.c; c.trace = object()
        with self.assertRaises(TypeError): c.finish_run_outputs(False)
        self.assertFalse(json.loads((c.output/'physics_result.json').read_text())['physics_success'])
        c.render_deferred_video.assert_not_called()
        for resource in (c.writer,c.head_writer,c.renderer): resource.close.assert_called()

    def test_render_exception_preserves_physics_result_and_closes_resources(self):
        c = self.c; c.render_deferred_video.side_effect = RuntimeError('encoder failed')
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'): c.finish_run_outputs(True)
        self.assertTrue(json.loads((c.output/'physics_result.json').read_text())['physics_success'])
        self.assertEqual(c.report['video_status'], 'failed')
        for resource in (c.writer,c.head_writer,c.renderer): resource.close.assert_called()

    def test_interruption_labels_recorded_video_as_failure(self):
        c = self.c; c.initialize_pair = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt): c.execute()
        c.render_deferred_video.assert_called_once()
        self.assertEqual(c.report['video_outcome'], 'failure')
        self.assertFalse(c.report['success'])

    def test_empty_failed_run_skips_video(self):
        c = self.c; c.trace = []
        c.finish_run_outputs(False)
        c.render_deferred_video.assert_not_called()
        self.assertEqual(c.report['video_status'], 'skipped_empty_trace')

if __name__ == '__main__': unittest.main()
