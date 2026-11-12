import json
import pytest
from cross_episode_sim.code_policy.fixture_profiles import HouseholdFixtureControls
from cross_episode_sim.runtime.fixture_controls import LiveFixtureControls


def test_drawer_retains_only_completed_withdrawal_pose(monkeypatch,tmp_path):
    controls=object.__new__(HouseholdFixtureControls)
    controls.drawer_withdrawn=False
    monkeypatch.setattr(LiveFixtureControls,'drawer_contact',lambda *args:None)
    def unwanted(*args):raise AssertionError('Unconditional tuck')
    monkeypatch.setattr(LiveFixtureControls,'navigation_posture',unwanted)
    controls.drawer_contact(tmp_path,{},True)
    events=[]
    controls.navigation_posture(tmp_path,lambda *args:events.append(args))
    report=json.loads((tmp_path/'posture_report.json').read_text())
    assert report['retained_posture'] and report['simulated_seconds']==0
    assert events==[('retained_checked_withdrawal_pose','completed')]


def test_failed_withdrawal_does_not_authorize_retention(monkeypatch,tmp_path):
    controls=object.__new__(HouseholdFixtureControls);controls.drawer_withdrawn=False
    def blocked(*args):raise RuntimeError('withdraw_path_blocked')
    monkeypatch.setattr(LiveFixtureControls,'drawer_contact',blocked)
    with pytest.raises(RuntimeError,match='withdraw_path_blocked'):
        controls.drawer_contact(tmp_path,{},True)
    assert not controls.drawer_withdrawn


def test_other_fixture_posture_keeps_existing_behavior(monkeypatch,tmp_path):
    controls=object.__new__(HouseholdFixtureControls);controls.drawer_withdrawn=False
    calls=[]
    monkeypatch.setattr(LiveFixtureControls,'navigation_posture',lambda *args:calls.append(args))
    controls.navigation_posture(tmp_path,lambda *args:None)
    assert len(calls)==1
