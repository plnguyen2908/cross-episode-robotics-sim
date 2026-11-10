import json
import h5py
import mujoco
import numpy as np
from cross_episode_sim.recording.deferred import StateVideoRecorder


def test_records_actual_states_without_stepping_or_rendering(tmp_path):
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    with StateVideoRecorder(model,data,tmp_path/'recording') as recorder:
        recorder.record('initial')
        recorder.record('duplicate')
        data.time=.1;data.qpos[0]=.42
        before=data.qpos.copy();recorder.record('moving')
        assert np.array_equal(data.qpos,before) and data.time==.1
    with h5py.File(tmp_path/'recording/states.h5') as f:
        assert len(f['states'])==2
        replay=mujoco.MjData(model)
        mujoco.mj_setState(model,replay,f['states'][1],mujoco.mjtState.mjSTATE_INTEGRATION)
        assert replay.time==.1 and replay.qpos[0]==.42
        assert list(f['phases'].asstr()[:])==['initial','moving']


def test_failure_still_closes_recording(tmp_path):
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    try:
        with StateVideoRecorder(model,mujoco.MjData(model),tmp_path/'failed') as recorder:
            recorder.record('failure');raise RuntimeError('simulated failure')
    except RuntimeError:pass
    assert json.loads((tmp_path/'failed/metadata.json').read_text())['frames']==1


def test_map_snapshots_only_follow_observed_map_changes(tmp_path):
    from types import SimpleNamespace as NS
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    data=mujoco.MjData(model)
    grid=NS(frames=1,origin=np.zeros(2),floor_views=np.ones((2,2)),
            evidence=np.zeros((2,2,1)),config=NS(resolution=.05,minimum_floor_views=1))
    tracking=NS(accepted=True,status='tracked',T_map_base=np.eye(4))
    with StateVideoRecorder(model,data,tmp_path/'recording') as recorder:
        recorder.record_map(grid,tracking,'start')
        data.time=1.;recorder.record_map(grid,tracking,'frozen')
        grid.frames=2;grid.evidence[1,1,0]=2
        data.time=2.;recorder.record_map(grid,tracking,'observed_update')
    events=[json.loads(line) for line in (tmp_path/'recording/diagnostics.jsonl').read_text().splitlines()]
    assert events[0]['map']==events[1]['map']!=events[2]['map']
    first=np.load(tmp_path/'recording'/events[0]['map'])
    last=np.load(tmp_path/'recording'/events[-1]['map'])
    assert not first['occupied'].any() and last['occupied'][1,1]
