"""Navigation replay while CARRYING an object.

Start = a history episode's saved state at the moment its carry navigate begins (object in the
gripper, arm in its carry posture, its own arm/gripper commands held). The robot localizes with the
360-degree turn, then replays a DIFFERENT history episode's carry route to the same destination
(same house + task), exactly as test_nav_replay.py does. Base motion in carry posture needs a motion
guard; this test passes an always-allow guard (the replayed routes were carried along by a past
robot) - a limitation, the pipeline's carry uses its swept check.
Private score: end error, handover to the destination, and whether the object stayed in the gripper.
"""
import argparse, json, sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import ACTIONS, HISTORY_ROOT, run  # noqa: E402


def carry_starts(per_house_task=1):
    plan = {p['id']: p for p in json.loads((HISTORY_ROOT/'plan.json').read_text())['episodes']}
    tracked = [e for e in plan if (HISTORY_ROOT/'episodes'/e/'trajectory.npz').exists() and (ACTIONS/e/'memory.json').exists()]
    groups = {}
    for e in tracked:groups.setdefault((plan[e]['house_id'][:12], plan[e]['task']), []).append(e)
    out = []
    for (house, task), eps in sorted(groups.items()):
        if len(eps) < 2:continue
        n = 0
        for e in sorted(eps):
            mem = json.loads((ACTIONS/e/'memory.json').read_text())
            seg = next((s for s in mem['skills'] if s['skill'] == 'navigate' and s['arguments'].get('destination') and s['recorded_success']), None)
            if seg is None:continue
            dest = seg['arguments']['destination']
            others = [o for o in eps if o != e and any(s['skill'] == 'navigate' and s['arguments'].get('destination') == dest
                      for s in json.loads((ACTIONS/o/'memory.json').read_text())['skills'])]
            if not others:continue
            out.append(dict(house=house, task=task, episode=e, segment=seg, source=plan[e]['source'])); n += 1
            if n >= per_house_task:break
    return out


def build_start(c):
    rec = Path(c['source'])/'action_recording'
    model = mujoco.MjModel.from_binary_path(str(rec/'model.mjb')); data = mujoco.MjData(model)
    with np.load(rec/'initial_state.npz') as s:mujoco.mj_setState(model, data, s['state'], int(s['state_type']))
    with np.load(rec/'actions.npz') as a:
        i = int(np.searchsorted(a['time'], c['segment']['start_seconds'])); data.qpos[:] = a['qpos'][i]; data.ctrl[:] = a['ctrl'][i]
    data.qvel[:] = 0; mujoco.mj_forward(model, data)
    for _ in range(200):mujoco.mj_step(model, data)  # settle with the recorded commands held
    carried = c['segment']['arguments']['target']
    bind = None; stack = [json.loads((Path(c['source'])/'task_manifest.json').read_text())]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if k == carried and isinstance(v, str):
                    try:model.body(v); bind = v
                    except KeyError:pass
                stack.append(v)
        elif isinstance(x, list):stack.extend(x)
    for name in (f'{carried}_test_object_main', f'{carried}_prop'):  # scene objects: <label>_test_object_main, grocery props <label>_prop
        if bind is not None:break
        try:model.body(name); bind = name
        except KeyError:pass
    obj = model.body(bind).id if bind else None; grip = model.body('robot_0/gripper/base').id
    held = (lambda: float(np.linalg.norm(data.xpos[obj]-data.xpos[grip]))) if obj is not None else (lambda: float('nan'))
    return dict(house=c['house'], task=c['task'], model=model, data=data, guard=lambda f, w, t: True, manifest_source=c['source'],
                exclude=c['episode'], carried=dict(label=carried, body=bind), held=held)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--only', nargs='*', help='history episode ids to run (default all)')
    a = p.parse_args()
    starts = [c for c in carry_starts() if not a.only or c['episode'] in a.only or c['source'] in a.only][a.shard::a.shards]
    print(json.dumps(dict(carry_starts=[(c['house'], c['task'], c['episode'], c['segment']['arguments']) for c in starts])), flush=True)
    for c in starts:
        try:
            st = build_start(c); dest = c['segment']['arguments']['destination']
            print(json.dumps(run(Path(c['source']), dest, a.output/f"{c['episode']}", 3, start=st)), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(json.dumps(dict(source=c['episode'], error=f'{type(e).__name__}: {e}')), flush=True)
