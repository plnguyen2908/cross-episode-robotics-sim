"""Private reset to a recorded dock; test live handle acquisition without oracle poses."""
import argparse,json
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.runtime.episode import EpisodeRuntime
from cross_episode_sim.runtime.fixture_controls import LiveFixtureControls


def main():
    p=argparse.ArgumentParser();p.add_argument('--navigation',type=Path,required=True)
    p.add_argument('--target',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rec=a.navigation/'state_recording'
    model=mujoco.MjModel.from_binary_path(str(rec/'model.mjb'));data=mujoco.MjData(model)
    with h5py.File(rec/'states.h5') as f:mujoco.mj_setState(model,data,f['states'][-1],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    selected=json.loads((a.navigation/'report.json').read_text())['selected_object']
    T=np.array(json.loads((rec/'diagnostics.jsonl').read_text().splitlines()[-1])['pose'])
    runtime=EpisodeRuntime(a.output);runtime.bind(model,data)
    success=False;error=None
    with runtime.activate():
        backend=LiveFixtureControls(runtime,a.output)
        backend.target_observation=(a.target,T[:3,:3].T@(np.asarray(selected['position'])-T[:3,3]))
        try:
            grasp,_=backend.acquire_grasp(a.target,a.output/'acquire')
            private=json.loads((grasp/'report.json').read_text());success=private.get('success',False)
        except Exception as exc:
            import traceback
            traceback.print_exc();error=str(exc)
        result=dict(success=success,error=error,scope='recorded dock handle-grasp component; no opening or full-task qualification')
        (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
