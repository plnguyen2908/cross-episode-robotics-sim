from pathlib import Path

import pytest

from cross_episode_sim.manipulation.molmo_objects import assets_root


def installation(path):
    (path / "grasps/droid/Egg_1").mkdir(parents=True)
    (path / "objects/thor").mkdir(parents=True)
    return path


def test_stale_configured_root_falls_back_to_local_complete_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    local = installation(tmp_path / ".cache/molmospaces/assets/local")
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(tmp_path / "stale"))
    with pytest.warns(RuntimeWarning, match="incomplete"):
        assert assets_root() == local


def test_valid_configured_root_has_priority_and_trims_whitespace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    configured = installation(tmp_path / "custom")
    installation(tmp_path / ".cache/molmospaces/assets/local")
    monkeypatch.setenv("MLSPACES_ASSETS_DIR", str(configured) + "\n\n")
    assert assets_root() == configured


def test_missing_installation_reports_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("MLSPACES_ASSETS_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="Set MLSPACES_ASSETS_DIR"):
        assets_root()
