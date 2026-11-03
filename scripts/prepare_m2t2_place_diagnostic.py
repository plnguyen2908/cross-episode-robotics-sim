"""Private isolated placement reset. No claim of navigation or carrying success."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
call=next(v for v in json.loads((a.source/'skill_calls.json').read_text()) if v['skill']=='place')
m=mujoco.MjModel.from_binary_path(str(a.source/'action_recording/model.mjb'));d=mujoco.MjData(m)
with np.load(a.source/'action_recording/initial_state.npz') as z:mujoco.mj_setState(m,d,z['state'],int(z['state_type']))
with np.load(a.source/'action_recording/actions.npz') as z:
 i=np.searchsorted(z['time'],call['start_time']);d.qpos[:]=z['qpos'][i];d.qvel[:]=0;d.ctrl[:]=z['ctrl'][i];d.time=float(z['time'][i])
mujoco.mj_forward(m,d);cp=a.output/'private_checkpoint';cp.mkdir();mujoco.mj_saveModel(m,str(cp/'model.mjb'),None);spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(cp/'state.npy',state)
(a.output/'report.json').write_text(json.dumps(dict(docking_success=False,isolated_placement_reset=True,source=str(a.source),call_index=call['index'],scope='Recorded pre-place qpos/ctrl reset, zero velocity. No navigation/carry success claim.'),indent=2))
