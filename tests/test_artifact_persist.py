import numpy as np
from types import SimpleNamespace
from cross_episode_sim.runtime import artifacts


def test_failed_stage_inputs_are_written_to_disk(tmp_path):
    root=tmp_path/'place';other=tmp_path/'elsewhere'
    runtime=SimpleNamespace(artifacts={
        str((root/'input.npz').resolve()):dict(points=np.ones((2,3))),
        str((root/'payload.npy').resolve()):np.zeros(3),
        str((root/'report.json').resolve()):'{"x": 1}',
        str((other/'skip.npy').resolve()):np.zeros(1)})
    assert artifacts.persist_under(runtime,[root])==3
    assert np.load(root/'input.npz')['points'].shape==(2,3)
    assert np.load(root/'payload.npy').shape==(3,)
    assert (root/'report.json').read_text()=='{"x": 1}'
    assert not (other/'skip.npy').exists()
