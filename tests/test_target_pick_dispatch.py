import json
from pathlib import Path
from cross_episode_sim.evaluation.target_conditioned_household import TargetConditionedHouseholdPipeline
from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline


def test_bowl_specific_proposals_do_not_replace_cup_path(monkeypatch,tmp_path):
    monkeypatch.setattr(SensorHouseholdPipeline,'_pick_object',lambda *args:'original')
    p=TargetConditionedHouseholdPipeline.__new__(TargetConditionedHouseholdPipeline)
    p.config={'task':'set_table'}
    p.stage=lambda name,command,*rest:command[1]
    dock=tmp_path/'dock';dock.mkdir()
    for label,expected in [('cup','original'),('bowl','scripts/execute_target_pick.py')]:
        (dock/'report.json').write_text(json.dumps({'selected_object':{'label':label}}))
        assert p._pick_object(1,dock,tmp_path/'pick')==expected
