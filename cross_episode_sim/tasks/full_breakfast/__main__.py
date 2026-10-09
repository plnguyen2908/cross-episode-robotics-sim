"""Generate one cross-episode example: two given history episodes and the episode to solve.

    python -m cross_episode_sim.tasks.full_breakfast --seed 7 --output runs/full_breakfast/7

writes, in --output:

    history_1.mp4, history_2.mp4   the given episodes, in the order a policy sees them
    current.mp4                    a demonstration of the episode to solve
    prompt.txt                     the instruction for the episode to solve
    episode.json                   which history is which, every episode's layout and outcome
    runs/                          each episode's full run directory (report, trace, videos)

History episodes: breakfast for two (gather two cups and two bowls, a person fills
them, serve them at the dining table) and coffee (brew two cups through the
machine). They appear in a seeded random order, each with its own randomized
layout: furniture offsets, tabletop and counter objects, objects on the floor, the
coffee machine at a random kitchen spot. The episode to solve combines them in a
new layout: brew coffee in both cups, a person fills the bowls, and serve every cup
beside a bowl at the dining table. Its prompt does not repeat the number of place
settings or how to make coffee: those are what the history shows.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROMPT = ('Make breakfast with coffee at the dining table. Set the same place settings as in the '
          'breakfast you saw: each person gets a bowl of food and, beside it, a cup of coffee made '
          'with the coffee machine the way you saw it done. The cups and bowls are in the kitchen; '
          'a person will fill the bowls. Finish with your gripper empty.\n')
VIDEO = {'breakfast': 'breakfast_gather_fill_serve.mp4', 'coffee': 'moonlake_full_workflow.mp4',
         'full_breakfast': 'moonlake_full_workflow.mp4'}


def sample(seed):
    """Every episode's randomized settings, from one seed."""
    rng = np.random.default_rng(seed)
    order = [str(t) for t in rng.permutation(['breakfast', 'coffee'])]
    def coffee_layout():
        # The main counter only: carrying a cup back to a machine on the right
        # counter, beside the fridge, has no room to turn yet.
        rng.choice(['main_counter', 'right_counter'])  # keeps each seed's other draws unchanged
        return dict(machine_slot='main_counter',
                    placement_seed=int(rng.integers(0, 10_000)), clutter=int(rng.integers(2, 5)),
                    floor_objects=int(rng.integers(6, 11)))
    episodes = {
        'breakfast': dict(seed=int(rng.integers(0, 10_000)), table_objects=int(rng.integers(2, 5)),
                          floor_objects=int(rng.integers(6, 11))),
        'coffee': coffee_layout(),
        'full_breakfast': coffee_layout(),
    }
    return order, episodes


def command(task, settings, output, label):
    python = [sys.executable, '-u', '-m']
    if task == 'breakfast':
        return python+['cross_episode_sim.tasks.breakfast.gather', '--output', str(output),
                       '--seed', str(settings['seed']), '--randomize', '--table-setting', 'dining',
                       '--table-objects', str(settings['table_objects']),
                       '--floor-objects', str(settings['floor_objects']), '--episode-label', label]
    extra = ['--serve-breakfast'] if task == 'full_breakfast' else []
    return python+['cross_episode_sim.tasks.coffee.workflow', '--output', str(output), '--cups', '2',
                   '--machine-slot', settings['machine_slot'], '--placement-seed', str(settings['placement_seed']),
                   '--table-setting', 'dining', '--clutter', str(settings['clutter']),
                   '--floor-objects', str(settings['floor_objects']), '--episode-label', label]+extra


def summary(run):
    report = json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {}
    manifest = json.loads((run/'task_manifest.json').read_text()) if (run/'task_manifest.json').exists() else {}
    return dict(success=bool(report.get('success')), error=report.get('error'),
                goal=report.get('composite_goal_evidence'),
                instruction=manifest.get('instruction'),
                machine_pose=manifest.get('coffee', {}).get('machine_pose'),
                floor_objects=[o['key'] for o in manifest.get('floor_objects', [])
                               + [s for s in manifest.get('scattered_objects', []) if s['kind'] == 'floor_object']],
                table_objects=[o['key'] for o in manifest.get('table_objects', [])
                               + [s for s in manifest.get('scattered_objects', []) if s['kind'] == 'surface_clutter']])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2'],
                        help='One GPU per episode; the three episodes run in parallel')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'runs').mkdir()
    order, settings = sample(args.seed)
    labels = {task: f'History {i+1} (given): {task}' for i, task in enumerate(order)}
    labels['full_breakfast'] = 'Current (to solve): breakfast with coffee'
    names = {order[0]: 'history_1', order[1]: 'history_2', 'full_breakfast': 'current'}
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    repo = Path(__file__).resolve().parents[3]
    if (repo/'external/robocasa/robocasa').exists():
        env.setdefault('ROBOCASA_DIR', str(repo/'external/robocasa/robocasa'))
    processes = {}
    for gpu, task in zip(args.gpus*3, [*order, 'full_breakfast']):
        run = out/'runs'/names[task]
        log = open(out/'runs'/f'{names[task]}.log', 'w')
        processes[task] = (subprocess.Popen(command(task, settings[task], run, labels[task]),
                                            env={**env, 'CUDA_VISIBLE_DEVICES': gpu},
                                            stdout=log, stderr=subprocess.STDOUT), log)
        print(f'{names[task]}: {task} on GPU {gpu}, log {log.name}', flush=True)
    for task, (process, log) in processes.items():
        process.wait(); log.close()
        print(f'{names[task]}: {task} exit {process.returncode}', flush=True)
    episodes = []
    for task in [*order, 'full_breakfast']:
        run = out/'runs'/names[task]
        video = run/VIDEO[task]
        if video.exists():
            shutil.copy2(video, out/f'{names[task]}.mp4')
        episodes.append(dict(name=names[task], task=task, role='current' if task == 'full_breakfast' else 'history',
                             label=labels[task], video=f'{names[task]}.mp4' if video.exists() else None,
                             run=str(run.relative_to(out)), settings=settings[task], **summary(run)))
    (out/'prompt.txt').write_text(PROMPT)
    record = dict(
        seed=args.seed,
        history_order=[f'{names[t]}: {t}' for t in order],
        episodes=episodes,
        current=dict(prompt=PROMPT.strip(), video='current.mp4',
                     goal='Both cups brewed and every cup and bowl at the dining table, each cup '
                          'beside a bowl, the bowls filled, the gripper empty.'),
        carried_from_history=dict(
            place_settings='2 people: 2 cups and 2 bowls (from the breakfast history)',
            coffee='remove the portafilter, dose it from the grounds box, reinstall it, put a cup under '
                   'the spout, power on, brew, take the cup out; repeat for the second cup (from the coffee history)'))
    (out/'episode.json').write_text(json.dumps(record, indent=2))
    ok = all(e['success'] for e in episodes)
    print(json.dumps(dict(output=str(out), success=ok, order=order)), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
