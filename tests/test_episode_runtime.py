import sys
from pathlib import Path
import numpy as np
import mujoco
import pytest
from cross_episode_sim.runtime.episode import EpisodeRuntime,load_checkpoint,episode_sensors


def checkpoint(path,model,data):
    p=path/'private_checkpoint';p.mkdir(parents=True)
    mujoco.mj_saveModel(model,str(p/'model.mjb'),None)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,state,spec)
    np.save(p/'state.npy',state)


def test_live_handoff_does_not_restore_or_replace_simulator(tmp_path):
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);checkpoint(tmp_path/'first',m,d)
    runtime=EpisodeRuntime()
    with runtime.activate():
        m,d=load_checkpoint(tmp_path/'first');d.time=1.;d.qpos[0]=.4
        checkpoint(tmp_path/'next',m,d)
        m2,d2=load_checkpoint(tmp_path/'next')
        assert m2 is m and d2 is d and d2.qpos[0]==.4
        with pytest.raises(RuntimeError,match='current live state'):load_checkpoint(tmp_path/'first')
        assert runtime.metrics['checkpoint_restores']==1


def test_camera_borrowed_across_stages_closed_only_at_episode_end(monkeypatch):
    import cross_episode_sim.sensors as sensors
    events=[]
    class Fake:
        def __init__(self,*args,**kw):events.append('created')
        def close(self):events.append('closed')
    monkeypatch.setattr(sensors,'OnboardSensorInterface',Fake)
    runtime=EpisodeRuntime();m=object();d=object()
    with runtime.activate():
        with episode_sensors(m,d) as a:pass
        with episode_sensors(m,d) as b:assert a is b
        assert events==['created']
    assert events==['created','closed']


def test_stage_runner_restores_argv_and_captures_exit(tmp_path):
    script=tmp_path/'stage.py';script.write_text('import sys\nprint(sys.argv[1])\nraise SystemExit(2)\n')
    original=sys.argv
    with (tmp_path/'log').open('w') as log:
        result=EpisodeRuntime().run([sys.executable,str(script),'hello'],log)
    assert result.returncode==2 and sys.argv is original
    assert (tmp_path/'log').read_text().strip()=='hello'


def test_stage_failure_keeps_traceback_in_live_log(tmp_path):
    script=tmp_path/'broken.py';script.write_text("raise ValueError('bad stage input')\n")
    with (tmp_path/'log').open('w') as log:
        result=EpisodeRuntime().run([sys.executable,str(script)],log)
    assert result.returncode==1
    assert 'ValueError: bad stage input' in (tmp_path/'log').read_text()
