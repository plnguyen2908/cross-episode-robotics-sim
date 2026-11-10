import os
import signal
import sys
import time
from cross_episode_sim.runtime.service_supervisor import ServiceSupervisor


def test_worker_restart_preserves_sibling_and_cleanup(tmp_path):
    specs=[dict(kind=k,command=[sys.executable,'-c','import time; time.sleep(60)'],env=os.environ.copy()) for k in ('molmo','qwen')]
    supervisor=ServiceSupervisor(specs,tmp_path,restart_delay=0.)
    try:
        supervisor.tick()
        old=supervisor.workers['molmo']['process']
        sibling=supervisor.workers['qwen']['process']
        os.kill(old.pid,signal.SIGKILL);old.wait(timeout=5)
        supervisor.tick()
        new=supervisor.workers['molmo']['process']
        assert new.pid!=old.pid and new.poll() is None
        assert supervisor.workers['molmo']['restarts']==1
        assert supervisor.workers['qwen']['process'] is sibling
        assert sibling.poll() is None
    finally:supervisor.close()
    assert all(w['process'].poll() is not None for w in supervisor.workers.values())


def test_onering_worker_restarts_keep_existing_recordings(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.serve_sensor_model import build
    import cross_episode_sim.navigation.onering as onering
    outputs=[]
    class FakePolicy:
        def __init__(self,checkpoint,output,device):
            output.mkdir(parents=True,exist_ok=False);outputs.append(output)
    monkeypatch.setattr(onering,'OneRingProcess',FakePolicy)
    args=SimpleNamespace(checkpoint='test',output=tmp_path/'onering',device='cpu')
    build('onering',args);build('onering',args)
    assert len(set(outputs))==2 and all(p.is_dir() for p in outputs)
