import json
from pathlib import Path
import pytest
from cross_episode_sim.evaluation.test_sources import test_sources


def test_manifest_excludes_unlisted_successful_folder(tmp_path):
    root=tmp_path/'test_two_rooms';root.mkdir()
    for name in ('listed','unlisted'):
        (root/name).mkdir()
        (root/name/'physics_result.json').write_text('{"physics_success": true}')
    (root/'manifest.json').write_text(json.dumps({'episodes':[{'task':'set_table','folder':'listed'}]}))
    assert test_sources(tmp_path,['test_two_rooms'],'set_table')==[(root/'listed').resolve()]


def test_missing_manifest_fails_instead_of_scanning(tmp_path):
    with pytest.raises(FileNotFoundError):test_sources(tmp_path,['test_two_rooms'],'set_table')


def test_manifest_escape_rejected(tmp_path):
    root=tmp_path/'test_two_rooms';root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'episodes':[{'task':'set_table','folder':'../outside'}]}))
    with pytest.raises(ValueError):test_sources(tmp_path,['test_two_rooms'],'set_table')


def test_task_filter(tmp_path):
    root=tmp_path/'test_two_rooms';root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'episodes':[{'task':'warm_two_pots','folder':'pot'}]}))
    assert test_sources(tmp_path,['test_two_rooms'],'set_table')==[]
