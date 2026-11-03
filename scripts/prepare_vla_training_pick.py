"""Private reset to an exported fine-tuning pick's first frame, not policy input."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--episode',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
audit=json.loads((a.dataset/'meta/arm_skills.json').read_text())
if audit['split']!='fine_tuning':raise ValueError('Expected fine-tuning data')
e=next(x for x in audit['episodes'] if x['episode_index']==a.episode)
if e['skill']!='pick' or e['category']!='bowl':raise ValueError('Current scorer supports bowl picks only')
source=Path(e['source']);m=mujoco.MjModel.from_binary_path(str(source/'action_recording/model.mjb'));d=mujoco.MjData(m)
with np.load(source/'action_recording/initial_state.npz') as z:mujoco.mj_setState(m,d,z['state'],int(z['state_type']))
i=e['source_sample_indices'][0]
with np.load(source/'action_recording/actions.npz') as z:
 d.qpos[:]=z['qpos'][i];d.qvel[:]=0;d.ctrl[:]=z['ctrl'][i];d.time=float(z['time'][i])
mujoco.mj_forward(m,d)
checkpoint=a.output/'private_checkpoint';checkpoint.mkdir(parents=True,exist_ok=False)
mujoco.mj_saveModel(m,str(checkpoint/'model.mjb'),None)
spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);np.save(checkpoint/'state.npy',state)
(a.output/'configuration.json').write_text(json.dumps(dict(source=str(source),training_episode=e['episode_index'],call_index=e['call_index'],prompt=e['prompt'],reset='Matches exported first-frame qpos, zero qvel, ctrl and time. Remaining integration state inherited from source initial state; not a replay-qualified dynamic checkpoint.',scope='training-example diagnostic, not held-out evaluation'),indent=2))
print(json.dumps(dict(output=str(a.output),prompt=e['prompt'],expert_duration=e['end_time']-e['start_time']),indent=2))
