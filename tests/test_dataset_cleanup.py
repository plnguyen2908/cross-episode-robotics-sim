import json
from cross_episode_sim.generation.dataset_cleanup import prune_failed_episodes


def test_prunes_finished_failures_preserves_active_and_success(tmp_path):
    for name in ('failed','active','success','render_failed'):
        folder=tmp_path/name;folder.mkdir();(folder/'run.log').write_text('detail')
    (tmp_path/'render_failed'/'report.json').write_text('{"success": true}')
    (tmp_path/'results.json').write_text(json.dumps({'failed':{'success':False},'success':{'success':True},'render_failed':{'success':False}}))
    assert prune_failed_episodes(tmp_path)==1
    assert not (tmp_path/'failed').exists()
    assert all((tmp_path/n).exists() for n in ('active','success','render_failed'))
    assert json.loads((tmp_path/'failures/failed.json').read_text())['log_tail']=='detail'
    assert prune_failed_episodes(tmp_path)==0
