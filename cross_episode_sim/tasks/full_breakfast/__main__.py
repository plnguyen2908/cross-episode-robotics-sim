"""Generate one cross-episode example: two given history episodes and the episode to solve.

    python -m cross_episode_sim.tasks.full_breakfast --seed 7 --output runs/full_breakfast/7

writes, in --output:

    history_1.mp4, history_2.mp4   the given episodes, in the order a policy sees them
    current.mp4                    a demonstration of the episode to solve
    prompt.txt                     the instruction for the episode to solve
    episode.json                   which history is which, every episode's layout and outcome
    runs/                          each episode's full run directory (report, trace, videos)

History episodes: breakfast for a seeded 1–3 people (gather one cup and bowl per
person, wait for filling, then serve them) and coffee (brew the same number of
cups through the machine). They appear in a seeded random order, each with its own randomized
layout: furniture offsets, tabletop and counter objects, objects on the floor, the
coffee machine in a seeded kitchen, living, or dining location. Food preparation
and vessel homes remain in the kitchen. The episode to solve combines them in a
new layout: brew coffee in every cup, a person fills the bowls, and serve every cup
beside a bowl at the dining table. Its prompt does not repeat the number of place
settings or how to make coffee: those are what the history shows.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from cross_episode_sim.paths import RUNS_DIR


def create_output(requested, seed, root=RUNS_DIR):
    """Every invocation gets a new directory; never reuse an earlier run."""
    if requested is None:
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f'full_breakfast_seed{seed}_', dir=root))
    output = Path(requested).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError:
        output = Path(tempfile.mkdtemp(prefix=output.name + '_', dir=output.parent))
    return output

PROMPT = ('Make breakfast with coffee at the dining table. Set the same place settings as in the '
          'breakfast you saw: each person gets a bowl of food and, beside it, a cup of coffee made '
          'with the coffee machine the way you saw it done. The cups and bowls are in the kitchen; '
          'a person will fill the bowls. Use any available cups and bowls, but serve only the number '
          'shown in the breakfast history. Finish with your gripper empty.\n')
VIDEO = {'breakfast': 'breakfast_gather_fill_serve.mp4', 'coffee': 'moonlake_full_workflow.mp4',
         'full_breakfast': 'moonlake_full_workflow.mp4'}


def sample(seed, profile="baseline", people=None, extra_furniture=None):
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
    if profile not in ('baseline', 'varied'):
        raise ValueError('Unknown layout profile')
    variation = np.random.default_rng(np.random.SeedSequence([seed, 104729]))
    sampled_people = int(variation.integers(1, 4))
    if people is None:
        people = sampled_people if profile == 'varied' else 2
    from cross_episode_sim.tasks.servings import validate_people
    validate_people(people)
    if extra_furniture is not None and (type(extra_furniture) is not int or not 0 <= extra_furniture <= 8):
        raise ValueError('extra furniture count must be in [0, 8]')
    for settings in episodes.values():
        settings['people'] = people
        settings['randomize_scene'] = profile == 'varied'
        sampled_furniture = int(variation.integers(1, 4))
        settings['extra_furniture'] = extra_furniture if extra_furniture is not None else sampled_furniture if profile == 'varied' else 0
        if profile == 'varied' and 'machine_slot' in settings:
            from cross_episode_sim.tasks.coffee.placement import sample_offset
            pose_rng = np.random.default_rng(np.random.SeedSequence([settings['placement_seed'], 65537]))
            _, offset = sample_offset(pose_rng, settings['machine_slot'])
            offset[1] = pose_rng.uniform(-.015, .015)
            settings['machine_offset'] = offset.tolist()
            room_rng = np.random.default_rng(np.random.SeedSequence([settings['placement_seed'], 131071]))
            settings['machine_room'] = str(room_rng.choice(['kitchen', 'living', 'dining']))
    return order, episodes


def command(task, settings, output, label, pour_step_deg=30, portafilter_step_deg=25):
    python = [sys.executable, '-u', '-m']
    variation = ['--extra-furniture', str(settings.get('extra_furniture', 0)),
                 '--nav-speed', str(settings.get('nav_speed', .24)),
                 '--turn-speed', str(settings.get('turn_speed', .6))]
    if settings.get('prepare_only'):
        variation += ['--prepare-only']
    if task == 'breakfast':
        return python+['cross_episode_sim.tasks.breakfast.gather', '--output', str(output),
                       '--seed', str(settings['seed']), '--randomize', '--table-setting', 'dining',
                       '--table-objects', str(settings['table_objects']),
                       '--floor-objects', str(settings['floor_objects']), '--episode-label', label,
                       '--people', str(settings.get('people', 2))]+variation
    extra = ['--serve-breakfast', '--extra-vessels', str(settings.get('extra_vessels', 0))] if task == 'full_breakfast' else []
    if settings.get('randomize_scene'):
        extra += ['--randomize-scene']
    extra += ['--machine-room', settings.get('machine_room', 'kitchen')]
    if 'machine_offset' in settings and settings.get('machine_room', 'kitchen') == 'kitchen':
        extra += ['--machine-offset', *map(str, settings['machine_offset'])]
    return python+['cross_episode_sim.tasks.coffee.workflow', '--output', str(output), '--cups', str(settings.get('people', 2)),
                   '--machine-slot', settings['machine_slot'], '--placement-seed', str(settings['placement_seed']),
                   '--table-setting', 'dining', '--clutter', str(settings['clutter']),
                   '--floor-objects', str(settings['floor_objects']), '--episode-label', label,
                   '--pour-step-deg', str(pour_step_deg),
                   '--portafilter-step-deg', str(portafilter_step_deg)]+extra+variation


def summary(run):
    report = json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {}
    manifest = json.loads((run/'task_manifest.json').read_text()) if (run/'task_manifest.json').exists() else {}
    return dict(success=bool(report.get('success')), error=report.get('error'),
                planner_execution=report.get('planner_execution'),
                goal=report.get('composite_goal_evidence'),
                instruction=manifest.get('instruction'),
                serving_count=manifest.get('serving_count', manifest.get('gathering_config', {}).get('people')),
                added_furniture=manifest.get('added_furniture', []),
                furniture_variation=manifest.get('furniture_variation', manifest.get('sampled_variation', {}).get('furniture')),
                object_poses={i['role']: i.get('position') for i in manifest.get('bindings', [])},
                machine_pose=manifest.get('coffee', {}).get('machine_pose'),
                machine_room=manifest.get('coffee', {}).get('machine_room', 'kitchen'),
                machine_support=manifest.get('supports', {}).get('coffee_station'),
                food_preparation_room=manifest.get('food_preparation_room', 'kitchen'),
                floor_objects=[o['key'] for o in manifest.get('floor_objects', [])
                               + [s for s in manifest.get('scattered_objects', []) if s['kind'] == 'floor_object']],
                table_objects=[o['key'] for o in manifest.get('table_objects', [])
                               + [s for s in manifest.get('scattered_objects', []) if s['kind'] == 'surface_clutter']])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extra-vessels', type=int, choices=range(7), default=3, help='Current pool has 3 + this many vessels per type; 0 disables the surplus pool')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path,
                        help='Output directory; default is a unique directory under CES_RUNS_DIR. Existing paths get a unique suffix.')
    parser.add_argument('--layout-profile', choices=('baseline', 'varied'), default='varied',
                        help='varied: 1–3 settings, extra furniture and independent mug homes; baseline: prior seed layouts')
    parser.add_argument('--people', type=int, choices=(1,2,3), help='Override the shared serving count in all three episodes')
    parser.add_argument('--extra-furniture', type=int, choices=range(9), help='Override added furniture count per episode (varied default: 1–3)')
    parser.add_argument('--nav-speed', type=float, default=.24, help='Mean base travel speed in m/s')
    parser.add_argument('--turn-speed', type=float, default=.6, help='Mean base turning speed in rad/s')
    parser.add_argument('--prepare-only', action='store_true', help='Check scene generation only; no robot execution or videos')
    parser.add_argument('--fail-fast', action='store_true', help='Stop all episode workers on a definitive failure')
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2'],
                        help='Physical nvidia-smi GPU indices or UUIDs for planning and EGL rendering; episodes run in parallel')
    parser.add_argument('--progress-interval', type=float, default=10.,
                        help='Seconds between progress summaries for all three episodes')
    parser.add_argument('--pour-step-deg', type=int, choices=(5, 10, 15, 20, 30), default=30,
                        help='Coffee tilt increment (default 30), with 5-degree collision screening retained')
    parser.add_argument('--portafilter-step-deg', type=int, choices=(5, 10, 15, 25), default=25,
                        help='Portafilter twist target increment (default 25) for both coffee episodes')
    args = parser.parse_args()
    if args.progress_interval <= 0:
        parser.error('--progress-interval must be positive')
    if not all(np.isfinite(v) and v > 0 for v in (args.nav_speed, args.turn_speed)):
        parser.error('Navigation speeds must be positive and finite')
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    bindings = (resolve_gpus(args.gpus)
                if not args.prepare_only and os.environ.get('MUJOCO_GL', 'egl').lower() == 'egl' else {})
    out = create_output(args.output, args.seed)
    print(f'Output: {out}', flush=True)
    (out/'runs').mkdir()
    order, settings = sample(args.seed, args.layout_profile, args.people, args.extra_furniture)
    settings['full_breakfast']['extra_vessels'] = args.extra_vessels
    for item in settings.values():
        item['prepare_only'] = args.prepare_only
        item['nav_speed'] = args.nav_speed
        item['turn_speed'] = args.turn_speed
    labels = {task: f'History {i+1} (given): {task}' for i, task in enumerate(order)}
    labels['full_breakfast'] = 'Current (to solve): breakfast with coffee'
    names = {order[0]: 'history_1', order[1]: 'history_2', 'full_breakfast': 'current'}
    (out/'run_manifest.json').write_text(json.dumps(dict(seed=args.seed, layout_profile=args.layout_profile, history_video_layout='wall_corners_plus_robot_2x2_v3', prepare_only=args.prepare_only, pour_step_deg=args.pour_step_deg,
                                                       portafilter_step_deg=args.portafilter_step_deg,
                                                       fail_fast=args.fail_fast,
                                                       skill_interface_required=True, episodes={
        names[task]: dict(task=task, label=labels[task], settings=settings[task],
                         gpu=gpu, gpu_binding=bindings.get(gpu))
        for gpu, task in zip(args.gpus*3, [*order, 'full_breakfast'])}), indent=2))
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    repo = Path(__file__).resolve().parents[3]
    if (repo/'external/robocasa/robocasa').exists():
        env.setdefault('ROBOCASA_DIR', str(repo/'external/robocasa/robocasa'))
    from cross_episode_sim.tasks.full_breakfast.fail_fast import definitive_failure, save_failure, stop_workers
    processes = {}
    try:
        for gpu, task in zip(args.gpus*3, [*order, 'full_breakfast']):
            run = out/'runs'/names[task]
            log = open(out/'runs'/f'{names[task]}.log', 'w')
            try:
                processes[task] = (subprocess.Popen(command(task, settings[task], run, labels[task], args.pour_step_deg,
                                                        args.portafilter_step_deg),
                                                env=worker_environment(env, gpu, bindings.get(gpu)),
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log)
            except BaseException:
                log.close()
                raise
            rendering = f', EGL device {bindings[gpu]["egl_device"]}' if gpu in bindings else ''
            print(f'{names[task]}: {task} on GPU {gpu}{rendering}, log {log.name}', flush=True)
    except BaseException:
        stop_workers(processes)
        for _, log in processes.values():
            log.close()
        raise
    from cross_episode_sim.tasks.full_breakfast.progress import print_progress
    next_progress = 0.
    try:
        while True:
            exit_codes = {names[task]: process.poll() for task, (process, _) in processes.items()}
            failure = definitive_failure(out, exit_codes) if args.fail_fast else None
            if failure:
                print(f"Fail-fast: {failure['episode']}: {failure['error']}; stopping this seed", flush=True)
                stop_workers(processes)
                save_failure(out, failure)
                return 1
            if time.monotonic() >= next_progress:
                if args.prepare_only:
                    print('Preparation: '+', '.join(f'{name}={"running" if code is None else "passed" if code == 0 else "failed"}' for name, code in exit_codes.items()), flush=True)
                else:
                    print_progress(out, exit_codes)
                next_progress = time.monotonic() + args.progress_interval
            if all(code is not None for code in exit_codes.values()):
                break
            time.sleep(min(args.progress_interval, 1.) if args.fail_fast else args.progress_interval)
    finally:
        if any(process.poll() is None for process, _ in processes.values()):
            stop_workers(processes)
        for _, log in processes.values():
            log.close()
    for task, (process, log) in processes.items():
        log.close()
        print(f'{names[task]}: {task} exit {process.returncode}', flush=True)
    if args.prepare_only:
        prepared = {names[t]: dict(exit_code=proc.returncode, settings=settings[t]) for t, (proc, _) in processes.items()}
        ok = all(v['exit_code'] == 0 for v in prepared.values())
        (out/'preparation.json').write_text(json.dumps(dict(success=ok, physical_validation='pending', episodes=prepared), indent=2))
        print(json.dumps(dict(output=str(out), preparation_success=ok, physical_validation='pending')), flush=True)
        return 0 if ok else 1
    people = settings['full_breakfast']['people']
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
                     goal=f'All {people} cups brewed and every cup and bowl at the dining table, each cup '
                          'beside a bowl, the bowls filled, the gripper empty.'),
        carried_from_history=dict(
            place_settings=f'{people} people: {people} cups and {people} bowls (from the breakfast history)',
            coffee='remove the portafilter, dose it from the grounds box, reinstall it, put a cup under '
                   'the spout, power on, brew, take the cup out; repeat for each remaining cup (from the coffee history)'))
    (out/'episode.json').write_text(json.dumps(record, indent=2))
    ok = all(e['success'] for e in episodes)
    print(json.dumps(dict(output=str(out), success=ok, order=order)), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
