import json
import pytest
from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline


@pytest.mark.parametrize('task,keep_going,report_exists,continues', [
    ('prepare_and_serve_meal', True, True, True),
    ('prepare_and_serve_meal', True, False, False),
    ('prepare_and_serve_meal', False, True, False),
    ('set_table', True, True, False),
])
def test_stage_failure_preserved_without_fabricating_output(
        tmp_path, monkeypatch, task, keep_going, report_exists, continues):
    pipeline = object.__new__(SensorHouseholdPipeline)
    pipeline.config = {'task': task}
    pipeline.continue_on_failure = keep_going
    pipeline.stages = []
    report = tmp_path/'report.json'
    if report_exists:
        report.write_text(json.dumps({'execution_success': False, 'reason': 'release_unconfirmed'}))
    def failed(self, name, command, report, success_key):
        self.stages.append({'name': name, 'success': False, 'exit_code': 1})
        raise RuntimeError('stage failed')
    monkeypatch.setattr(SensorSetTablePipeline, 'stage', failed)
    if continues:
        result = pipeline.stage('place', ['python', 'test_place.py'], report, 'execution_success')
        assert result['execution_success'] is False
    else:
        with pytest.raises(RuntimeError, match='stage failed'):
            pipeline.stage('place', ['python', 'test_place.py'], report, 'execution_success')
    assert pipeline.stages == [{'name': 'place', 'success': False, 'exit_code': 1}]
