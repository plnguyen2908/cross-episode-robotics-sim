"""Private post-run physical support diagnostics; never imported by execution.

Does not certify a full task or reconstruct missing appliance heating history.
"""
import argparse,json
from pathlib import Path
import h5py,mujoco
from cross_episode_sim.finetuning.replay_scoring import CompositeReplayScorer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));d=mujoco.MjData(m)
    with h5py.File(a.recording/'states.h5') as f:
        states=f['states'][-10:]
    mujoco.mj_setState(m,d,states[-1],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    scorer=CompositeReplayScorer(m,d,a.source);final=scorer.snapshot(d);recent=[]
    for state in states:
        mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        recent.append(scorer.snapshot(d))
    result=dict(snapshot=final,last_recorded_frames=len(recent),
        recent_relations={role:[dict(support=s['objects'][role]['support'],region=s['objects'][role]['region']) for s in recent] for role in final['objects']},
        scope='private post-run snapshot; not full task/heating qualification')
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(held=final['held'],objects={k:dict(support=v['support'],region=v['region']) for k,v in final['objects'].items()})))


if __name__=='__main__':main()
