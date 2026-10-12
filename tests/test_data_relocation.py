import pytest

from scripts.install_data import relocate


def test_relocate_preserves_backups_spaces_binary_and_other_roots(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cache = tmp_path / "local cache"
    cache.mkdir()
    original = '<mesh file="/old/cache/Bedroom Objects/Book.obj"/><mesh file="/old/cache_other/x.obj"/>'
    scene = data / "scene.xml"
    scene.write_text(original)
    binary = data / "grasps.npz"
    binary.write_bytes(b"/old/cache/\x00\xff")
    backup = tmp_path / "backup"
    assert relocate(data, [("/old/cache", cache)], backup) == 1
    assert (backup / "scene.xml").read_text() == original
    assert scene.read_text() == original.replace("/old/cache/", str(cache) + "/")
    assert binary.read_bytes() == b"/old/cache/\x00\xff"
    assert relocate(data, [("/old/cache", cache)], tmp_path / "backup2") == 0


def test_relocate_validates_all_roots_before_changing_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    scene = data / "scene.xml"
    scene.write_text("/old/cache/object.obj")
    with pytest.raises(ValueError, match="Replacement directory does not exist"):
        relocate(data, [("/old/cache", tmp_path), ("/other", tmp_path / "missing")], tmp_path / "backup")
    assert scene.read_text() == "/old/cache/object.obj"
    assert not (tmp_path / "backup").exists()


def test_model_digest_preserves_qualification_across_default_cache_relocation(tmp_path,monkeypatch):
    from cross_episode_sim import paths
    home=tmp_path/'home'
    old=tmp_path/'old_cache'
    new=home/'.cache/molmo-spaces-resources'
    monkeypatch.setattr(paths.Path,'home',classmethod(lambda cls:home))
    monkeypatch.setattr(paths,'local_roots',lambda:{old:'${MLSPACES_CACHE_DIR}'})
    model=tmp_path/'model.xml'
    model.write_text(f'<mesh file="{old}/objects/egg.obj" scale="1 1 1"/>')
    original=paths.model_digest(model)
    model.write_text(f'<mesh file="{new}/objects/egg.obj" scale="1 1 1"/>')
    assert paths.model_digest(model)==original
    model.write_text(f'<mesh file="{new}/objects/egg.obj" scale="2 2 2"/>')
    assert paths.model_digest(model)!=original
