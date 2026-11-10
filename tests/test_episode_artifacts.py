import numpy as np
from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.runtime import artifacts


def test_handoff_stays_in_memory_and_has_no_aliasing(tmp_path):
    runtime=EpisodeRuntime(artifact_root=tmp_path)
    with runtime.activate():
        x=np.arange(4.)
        artifacts.save_archive(tmp_path/'navigation_map.npz',evidence=x)
        artifacts.save_array(tmp_path/'private_checkpoint/state.npy',x)
        artifacts.write_text(tmp_path/'summary.json','{"config": {}}')
        artifacts.write_text(tmp_path/'report.json','{"success": true}')
        x[:]=99
        assert artifacts.exists(tmp_path/'navigation_map.npz')
        assert artifacts.load_array(tmp_path/'navigation_map.npz')['evidence'][0]==0
        copied=artifacts.load_array(tmp_path/'navigation_map.npz');copied['evidence'][:]=88
        assert artifacts.load_array(tmp_path/'navigation_map.npz')['evidence'][0]==0
        assert not list(tmp_path.iterdir())
    assert [p.name for p in tmp_path.iterdir()]==['report.json']


def test_legacy_reads_are_preserved_outside_active_output(tmp_path):
    source=tmp_path/'source.npy';np.save(source,np.arange(3))
    with EpisodeRuntime(artifact_root=tmp_path/'new_run').activate():
        assert np.array_equal(artifacts.load_array(source),np.arange(3))
