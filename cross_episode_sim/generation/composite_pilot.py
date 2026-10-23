"""Independent composite pilot specifications and fail-fast physical execution.

Datagen owns scenes, policies issue atomic calls, evaluators own goal checks.
No legacy history pairing or standalone fixture-demo concatenation is used.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path

TASKS = {
    'set_table': dict(
        instruction='Set the dining table for three people. Give each place setting one bowl and one cup. Leave extra items where they are.',
        objects={'bowl': 3, 'cup': 3},
        procedures=['by_place_setting', 'by_object_category'],
        goal_keys=['three_complete_settings', 'extras_unchanged', 'empty_hand'],
        requirements=['multi_object_transfer', 'setting_regions']),
    'clear_and_store': dict(
        instruction='Clear the bowls and cups from the dining table. Put the bowls in the kitchen cabinet and the cups in the drawer. Leave both storage compartments closed.',
        objects={'bowl': 3, 'cup': 3},
        procedures=['finish_one_storage_at_a_time', 'open_both_then_store_then_close'],
        goal_keys=['bowls_in_cabinet', 'cups_in_drawer', 'cabinet_closed', 'drawer_closed', 'empty_hand'],
        requirements=['multi_object_transfer', 'cabinet_interior_placement', 'drawer_interior_placement', 'shared_fixture_manipulation']),
    'prepare_and_serve_meal': dict(
        instruction='Warm the food container in the microwave for at least 30 seconds. Bring it and the cold drink from the fridge to the dining table. Leave the microwave off and both appliance doors closed.',
        objects={'food_container': 1, 'drink_bottle': 1},
        procedures=['heat_while_retrieving', 'retrieve_before_heating'],
        goal_keys=['food_heated_long_enough', 'all_meal_items_served', 'microwave_off', 'microwave_closed', 'fridge_closed', 'empty_hand'],
        requirements=['multi_object_transfer', 'fridge_retrieval', 'microwave_loading', 'microwave_controls', 'private_duration_tracking']),
    'unpack_and_sort_groceries': dict(
        instruction='Put all packaged drinks in the purple basket and all packaged snacks in the orange basket. Leave both baskets in their initial locations. Leave unrelated objects where they are.',
        objects={'drink_bottle': 2, 'snack_box': 2, 'basket': 2},
        procedures=['drinks_first', 'snacks_first'],
        goal_keys=['drinks_in_purple_basket', 'snacks_in_orange_basket', 'baskets_unchanged', 'unrelated_unchanged', 'empty_hand'],
        requirements=['multi_object_transfer', 'basket_containment']),
    'warm_two_pots': dict(
        instruction='Warm both cookware items on separate stove burners for at least 30 seconds each. Finish with both burners off and both items remaining on the stove.',
        objects={'pot': 2},
        procedures=['start_first_before_placing_second', 'place_both_then_start'],
        goal_keys=['both_pots_heated_long_enough', 'both_burners_off', 'pots_on_separate_burners', 'empty_hand'],
        requirements=['multi_object_transfer', 'pot_grasps', 'independent_burner_controls', 'private_duration_tracking']),
}
VARIANTS = ('original', 'alternative_procedure', 'kitchen_layout',
            'furniture_translation', 'furniture_rotation', 'object_instances', 'combined')


def scene_hash(config):
    return hashlib.sha256(json.dumps(config['scene'], sort_keys=True).encode()).hexdigest()


def configurations():
    """35 independent episodes; variants 1/2 have identical scene specifications."""
    result = []
    for task, definition in TASKS.items():
        for i, variant in enumerate(VARIANTS):
            scene = dict(seed=7, kitchen_layout=1, kitchen_style=1, people=3,
                         object_counts=deepcopy(definition['objects']),
                         asset_instance_set='baseline', furniture={},
                         forbidden_object_categories=['plate', 'flat_object'],
                         minimum_heating_seconds=30. if task in ('prepare_and_serve_meal', 'warm_two_pots') else None)
            if task == 'unpack_and_sort_groceries':
                placements = {
                    'original': ('kitchen_counter', 'kitchen_counter'),
                    'alternative_procedure': ('kitchen_counter', 'kitchen_counter'),
                    'kitchen_layout': ('dining_table', 'dining_table'),
                    'furniture_translation': ('kitchen_counter', 'dining_table'),
                    'furniture_rotation': ('dining_table', 'kitchen_counter'),
                    'object_instances': ('dining_table', 'dining_table'),
                    'combined': ('dining_table', 'kitchen_counter'),
                }
                scene['basket_supports'] = dict(zip(('purple_basket', 'orange_basket'), placements[variant]))
            if task == 'clear_and_store':
                # Tall cups can fit an open drawer but jam against its cabinet
                # during closure. Use the previously qualified compact mugs.
                scene['asset_categories'] = {'cup': 'mug'}
            if variant in ('kitchen_layout', 'combined'):
                scene['kitchen_layout'] = 3
            if variant in ('furniture_translation', 'combined'):
                scene['furniture'].update({
                    'dining_set': dict(dx=.35, dy=.05, yaw_deg=0),
                    'sofa': dict(dx=.7, dy=.6, yaw_deg=0),
                    'coffee_table': dict(dx=.55, dy=.6, yaw_deg=0),
                })
            if variant in ('furniture_rotation', 'combined'):
                for name, angle in ([('armchair_left', 8), ('armchair_right', -8)]
                                    if variant == 'furniture_rotation' else [('sofa', 8), ('coffee_table', 8)]):
                    scene['furniture'].setdefault(name, dict(dx=0, dy=0))['yaw_deg'] = angle
            if variant in ('object_instances', 'combined'):
                scene['asset_instance_set'] = 'same_category_alternatives'
            c = dict(schema_version=1, episode_id=f'{task}_{i+1:02d}_{variant}',
                     task=task, variant=variant, instruction=definition['instruction'],
                     procedure=definition['procedures'][int(variant in ('alternative_procedure', 'combined'))],
                     scene=scene, goal_keys=definition['goal_keys'],
                     requirements=definition['requirements'], catalog_version='fixed-atomic-v1',
                     video_speedup=1.0, dataset_role='pilot_unassigned',
                     motion=dict(optimize_motion_components=True, smooth_action=True, record_action_replay=True),
                     initial_state_policy='deterministic_setup_from_scene_config',
                     validation_status='not_executed')
            from cross_episode_sim.tasks.composites import create_task
            c['instruction'] = create_task(c).make_prompt()
            c['scene_spec_sha256'] = scene_hash(c)
            result.append(c)
    return result



def variation_configurations():
    """Seven coverage demos per task, including a matched closed-door route pair."""
    result = []
    for task in TASKS:
        group = {c['variant']: c for c in configurations() if c['task'] == task}
        if task == 'warm_two_pots':
            result.extend(group.values())
            continue
        chosen = [deepcopy(group[k]) for k in
                  ('original', 'alternative_procedure', 'kitchen_layout',
                   'furniture_translation', 'object_instances')]
        chosen[3]['scene']['furniture'] = deepcopy(group['combined']['scene']['furniture'])
        chosen[3]['variation_axis'] = 'furniture_translation_and_rotation'
        for route in ('open_shortcut', 'living_room_detour'):
            c = deepcopy(group['original'])
            c['room_route'] = route
            if task == 'unpack_and_sort_groceries':
                c['scene']['basket_supports'] = dict(purple_basket='dining_table', orange_basket='dining_table')
            c['scene']['room_doors'] = {
                'room_door_0': 'closed', 'room_door_1': 'open', 'room_door_2': 'open'}
            c['variation_axis'] = route
            chosen.append(c)
        for i, c in enumerate(chosen, 1):
            c['episode_id'] = f"{task}_{i:02d}_{c.get('variation_axis', c['variant'])}"
            c['scene_spec_sha256'] = scene_hash(c)
        result.extend(chosen)
    # Seven coverage cases remain; appearance is mixed into existing axes.
    # Keep the matched shortcut/detour pair visually and physically identical.
    themes=(None,'warm_home','cool_daylight','sage_neutral','warm_home','cool_daylight','cool_daylight')
    counts={}
    for c in result:
        index=counts.get(c['task'],0);counts[c['task']]=index+1
        theme=themes[index]
        if theme:
            c['scene']['appearance']={'preset':theme}
            c['episode_id']+='_'+theme
        c['observation']={'physical_cameras':['robot_0/navigation_cam','robot_0/wrist_cam'],
                          'video_panels':['navigation_rgb','navigation_depth','wrist_rgb','wrist_depth'],
                          'depth_units':'metres','depth_display_range_m':[0,8]}
        c['scene_spec_sha256']=scene_hash(c)
    return result

def validate_config(c):
    from cross_episode_sim.generation.room_routes import room_door_states
    room_door_states(c['scene'])
    if c['task'] not in TASKS or c['variant'] not in (*VARIANTS,'sampled'):
        raise ValueError('Unknown task or variant')
    d = TASKS[c['task']]
    if c['procedure'] not in d['procedures'] or c['goal_keys'] != d['goal_keys']:
        raise ValueError('Task procedure/goal contract changed')
    if c['requirements'] != d['requirements']:
        raise ValueError('Task requirements cannot be removed')
    if c['scene_spec_sha256'] != scene_hash(c):
        raise ValueError('Scene hash mismatch; regenerate after changing scene configuration')
    if c.get('video_speedup') != 1.0:
        raise ValueError('Pilot videos must preserve normal playback')
    if any(k in c for k in ('histories', 'history_ids', 'history_paths')):
        raise ValueError('Pilot episodes must not bundle selected histories')


def describe(c):
    from cross_episode_sim.generation.composite_scenes import scene_specification
    from cross_episode_sim.tasks.composites import create_task
    task = create_task(c)
    return dict(task_class=type(task).__name__, prompt=task.make_prompt(),
                scene_specification=scene_specification(c),
                plan=task.make_plan(), physical_execution=False)


def preflight(c):
    from cross_episode_sim.generation.environment_factory import preflight as check_scene
    validate_config(c)
    describe(c)  # Validate task-specific parameters before any GPU allocation.
    return check_scene(c)


def execute(c, output):
    from cross_episode_sim.generation.episode_setup import prepare_composite_episode
    status = preflight(c)
    if not status['ready']:
        raise RuntimeError('Physical preflight failed: ' + json.dumps(status))
    steps=len(describe(c)['plan'])
    summary = ""
    if c["scene"].get("base_layout"):
        house=c["scene"]["base_layout"]
        summary=f" | rooms={house['room_count']} house_seed={house['seed']} controls={c.get('sampling',{}).get('active_controls',[])}"
    print(f"{c['task']}: [0/{steps}] scene setup{summary}",flush=True)
    env, task, evaluator, evidence = prepare_composite_episode(c, output)
    success = False
    failure = None
    try:
        (output / 'instruction.txt').write_text(task.make_prompt() + '\n')
        plan=task.make_plan()
        steps=len(plan)
        print(f"{c['task']}: [0/{steps}] scene ready; resolved atomic plan",flush=True)
        (output / 'oracle_plan.json').write_text(json.dumps(plan, indent=2))
        result = task.execute()
        final_snapshot=env.evaluation_snapshot()
        (output/'final_snapshot.json').write_text(json.dumps(final_snapshot,indent=2))
        goal = evaluator.evaluate(final_snapshot)
        result.update(success=bool(goal) and all(v is True for v in goal.values()), goal_evidence=goal)
        if not result['success']:
            raise RuntimeError('Composite goal failed: ' + str(goal))
        if set(result['goal_evidence']) != set(c['goal_keys']):
            raise RuntimeError('Task/config goal contract mismatch')
        print(f"{c['task']}: [{steps}/{steps}] goal passed; rendering video",flush=True)
        video = Path(env.render(output, speedup=1.0)).resolve()
        if output.resolve() not in video.parents or not video.is_file():
            raise RuntimeError('Episode-local rendered video missing')
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
            '-show_entries', 'format=duration', '-of', 'json', str(video)]))
        if float(probe['format']['duration']) <= 0:
            raise RuntimeError('Video is empty')
        (output / 'skill_calls.json').write_text(json.dumps(env.interface.history, indent=2))
        result.update(setup_evidence=evidence, video=str(video.relative_to(output.resolve())),
                      video_speedup=1.0, scene_spec_sha256=c['scene_spec_sha256'])
        (output / 'report.json').write_text(json.dumps(result, indent=2))
        success = True
    except Exception as exc:
        failure = dict(kind=type(exc).__name__, message=str(exc))
        raise
    finally:
        try:
            env.close()
        finally:
            if not success:
                (output / 'report.json').write_text(json.dumps(dict(
                    success=False, error=failure, completed_calls=len(env.interface.history)), indent=2))


def run_worker(cmd, env, folder, timeout, progress=print, episode_id=None):
    """Stream stage changes; terminate the worker and its children on timeout."""
    with (folder / 'run.log').open('w') as log:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        def read_output():
            previous = None
            for line in process.stdout:
                log.write(line)
                log.flush()
                # Controller heartbeats repeat. Expose only actual stage changes.
                if re.match(r'^[a-z_]+: \[\d+/\d+\]',line):
                    stage = line.strip()
                    if stage != previous:
                        progress(f"{episode_id}: {stage.split(': ', 1)[1]}" if episode_id else stage)
                        previous = stage
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            reader.join(timeout=5)



def run_episode_queue(configs, run, workers):
    """Fill every worker slot from one queue, including same-task episodes."""
    pending=iter(configs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        def submit_next():
            config=next(pending,None)
            if config is not None:active[pool.submit(run,config)]=config
        for _ in range(workers):submit_next()
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for job in finished:
                config=active.pop(job)
                try:result=job.result()
                except Exception as exc:result=dict(success=False,error=str(exc))
                submit_next()
                yield config,result


def run_task_lanes(configs, run, workers):
    """Parallelize tasks while executing each task's variants sequentially."""
    from collections import deque
    lanes = {}
    for config in configs:
        lanes.setdefault(config['task'], deque()).append(config)
    ready = deque(lanes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while ready or active:
            while ready and len(active) < workers:
                task = ready.popleft()
                config = lanes[task].popleft()
                active[pool.submit(run, config)] = config
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for job in finished:
                config = active.pop(job)
                try:
                    result = job.result()
                except Exception as exc:
                    result = dict(success=False, error=str(exc))
                yield config, result
                if lanes[config['task']]:
                    ready.append(config['task'])

def load_resume_configs(root):
    """Read saved specifications rather than regenerate randomized parameters."""
    manifest = root / 'batch_configs.json'
    if manifest.exists():
        return json.loads(manifest.read_text())
    configs = [json.loads(p.read_text()) for p in sorted(root.glob('*/config.json'))]
    if not configs:
        raise ValueError(f'No saved configurations in {root}')
    return configs


def completed_result(folder):
    try:
        report = json.loads((folder / 'report.json').read_text())
        video = folder / report['video']
        if report.get('success') is True and video.is_file() and video.stat().st_size:
            return dict(success=True, output=str(folder), skipped=True)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def prepare_retry_folder(root, episode_id):
    folder = root / episode_id
    if folder.exists():
        archive = root / '_previous_attempts' / episode_id
        archive.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (archive / str(attempt)).exists():
            attempt += 1
        folder.rename(archive / str(attempt))
    folder.mkdir()
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('pilot', 'variations', 'randomized'), default='pilot')
    p.add_argument('--seed', type=int, help='Override randomized/YAML master seed')
    p.add_argument('--count-per-task', type=int, help='Override randomized/YAML sample count')
    p.add_argument('--write-configs', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--sampling-yaml', type=Path, nargs='+', help='Per-task YAML variation samplers')
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--variant', choices=(*VARIANTS,'sampled'))
    p.add_argument('--people', type=int, choices=(1, 2, 3),
                   help='Override paired bowl/cup counts for set_table and clear_and_store')
    p.add_argument('--room-route', choices=('open_shortcut', 'living_room_detour'),
                   help='Closed kitchen/dining door with an open living-room detour; physical qualification pending')
    p.add_argument('--describe', action='store_true', help='Print built-in task scene, prompt and plans without simulation')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', type=Path, help='Keep successes and retry failed/unfinished episodes in a saved batch')
    p.add_argument('--workers', type=int, default=8, help='Concurrent independent episodes, each leasing a separate idle GPU')
    p.add_argument('--timeout', type=float, default=7200, help='Maximum seconds per physical episode')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if (a.seed is not None or a.count_per_task is not None) and (a.resume or a.config or a.worker or (a.suite!='randomized' and not a.sampling_yaml)):
        p.error('Seed/count overrides require a new randomized or YAML batch')
    if a.count_per_task is not None and a.count_per_task<1:p.error('count-per-task must be positive')
    if a.workers < 1 or a.timeout <= 0:
        p.error('workers and timeout must be positive')
    if a.sampling_yaml and (a.resume or a.config or a.worker or a.people is not None or a.room_route or a.variant):
        p.error("--sampling-yaml cannot be combined with resume/config/worker/people/room-route/variant")
    if a.resume and (a.output or a.config or a.worker or a.room_route):
        p.error('--resume cannot be combined with --output, --config, --worker or --room-route')
    if a.resume and not a.resume.is_dir():
        p.error(f'Resume folder does not exist: {a.resume}')
    configs = load_resume_configs(a.resume) if a.resume else [json.loads(a.config.read_text())] if a.config else (variation_configurations() if a.suite == 'variations' else configurations())
    if a.suite == 'randomized' and not (a.resume or a.config or a.sampling_yaml):
        from cross_episode_sim.generation.variation_sampling import sample_file
        configs=[c for i,name in enumerate(TASKS) for c in sample_file(Path(__file__).resolve().parents[2]/'configs/composite_sampling/randomized'/f'{name}.yaml',seed=None if a.seed is None else a.seed+i,count=a.count_per_task)]
    if a.sampling_yaml:
        from cross_episode_sim.generation.variation_sampling import sample_file
        configs = [c for i,path in enumerate(a.sampling_yaml) for c in sample_file(path,seed=None if a.seed is None else a.seed+i,count=a.count_per_task)]
        if len({c['episode_id'] for c in configs}) != len(configs):
            p.error('Provide only one sampling YAML per task per batch')
    from cross_episode_sim.control.smooth_action import composite_motion_settings
    for c in configs:
        c['motion'] = composite_motion_settings(c.get('motion'))
    configs = [c for c in configs if (not a.task or c['task'] == a.task)
               and (not a.variant or c['variant'] == a.variant)]
    if not configs:
        p.error('No matching configurations')
    if a.people is not None and not a.resume:
        from cross_episode_sim.tasks.composites import create_task
        for c in configs:
            if c['task'] in ('set_table', 'clear_and_store'):
                c['scene']['people'] = a.people
                c['scene']['object_counts'].update(bowl=a.people, cup=a.people)
                c['instruction'] = create_task(c).make_prompt()
                c['scene_spec_sha256'] = scene_hash(c)
    if a.room_route:
        if any(c['task'] == 'warm_two_pots' for c in configs):
            p.error('Select a cross-room task: warm_two_pots currently stays in the kitchen')
        for c in configs:
            c['scene']['room_doors'] = {'room_door_0': 'closed', 'room_door_1': 'open', 'room_door_2': 'open'}
            c['room_route'] = a.room_route
            c['episode_id'] += '_' + a.room_route
            c['scene_spec_sha256'] = scene_hash(c)
    for c in configs:
        validate_config(c)
    if a.write_configs:
        a.write_configs.mkdir(parents=True, exist_ok=True)
        for c in configs:
            f = a.write_configs / (c['episode_id'] + '.json')
            if f.exists() and json.loads(f.read_text()) != c:
                p.error(f'Refusing to overwrite edited config: {f}')
            f.write_text(json.dumps(c, indent=2) + '\n')
        print(f'Wrote {len(configs)} independent specifications; no simulations executed.')
        return 0
    if a.describe:
        print(json.dumps({c['episode_id']: describe(c) for c in configs}, indent=2))
        return 0
    checks = {c['episode_id']: preflight(c) for c in configs}
    if a.preflight or not all(row['ready'] for row in checks.values()):
        print(json.dumps(checks, indent=2))
        return 0 if all(row['ready'] for row in checks.values()) else 2
    if a.worker:
        if len(configs) != 1 or not a.output:
            p.error('Worker requires one config and output')
        try:
            execute(configs[0], a.output.resolve())
        except Exception as exc:
            report=a.output/'report.json'
            if not report.exists():
                report.write_text(json.dumps(dict(success=False,error=dict(
                    kind=type(exc).__name__,message=str(exc))),indent=2))
            raise
        return 0
    root = a.resume or a.output
    if a.resume:
        pass
    elif root:
        root.mkdir(parents=True, exist_ok=False)
    else:
        Path('runs').mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='composite_pilot_', dir='runs'))
    root = root.resolve()
    import fcntl
    batch_lock = (root / '.generation.lock').open('a')
    try:
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error(f'Another generator is already running in {root}')
    if not a.resume:
        (root / 'batch_configs.json').write_text(json.dumps(configs, indent=2))
    print(f'OUTPUT={root}', flush=True)
    def run(c):
        from cross_episode_sim.generation.gpu_pool import lease_gpu
        from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
        folder = prepare_retry_folder(root, c['episode_id'])
        config = folder / 'config.json'; config.write_text(json.dumps(c, indent=2))
        with lease_gpu('auto', waiting=lambda: print(c['episode_id'] + ': waiting for idle GPU', flush=True)) as gpu:
            binding = resolve_gpus([gpu])[gpu]
            tqdm.write(f"{c['episode_id']}: GPU {gpu}")
            (folder/'gpu_assignment.json').write_text(json.dumps(dict(gpu=gpu),indent=2))
            env = worker_environment(dict(os.environ, MUJOCO_GL='egl'), gpu, binding)
            cmd = [sys.executable, '-u', '-m', 'cross_episode_sim.generation.composite_pilot', '--worker', '--config', str(config),
                   '--output', str(folder)]
            try:
                returncode = run_worker(cmd, env, folder, a.timeout, tqdm.write, episode_id=c['episode_id'])
                report = json.loads((folder / 'report.json').read_text()) if (folder / 'report.json').exists() else {}
                return dict(success=returncode == 0 and report.get('success') is True,
                            exit_code=returncode, error=report.get('error'), output=str(folder))
            except subprocess.TimeoutExpired:
                return dict(success=False, error='episode_timeout', output=str(folder))
    from tqdm.auto import tqdm
    results_path = root / 'results.json'
    results = json.loads(results_path.read_text()) if a.resume and results_path.exists() else {}
    pending = []
    for c in configs:
        completed = completed_result(root / c['episode_id']) if a.resume else None
        if completed:
            results[c['episode_id']] = completed
            print(c['episode_id'] + ': KEEP successful example', flush=True)
        else:
            pending.append(c)
    print(f'{len(configs) - len(pending)} successful examples kept; {len(pending)} to run', flush=True)
    results_path.write_text(json.dumps(results, indent=2))
    for c, result in tqdm(run_episode_queue(pending, run, a.workers),
                          initial=len(configs) - len(pending), total=len(configs), desc='Composite demos', unit='demo'):
        results[c['episode_id']] = result
        temporary_results = root / 'results.json.tmp'
        temporary_results.write_text(json.dumps(results, indent=2))
        temporary_results.replace(root / 'results.json')
        print(c['episode_id'] + ': ' + ('PASS' if result['success'] else 'FAIL'), flush=True)
    return 0 if all(results[c['episode_id']]['success'] for c in configs) else 1


if __name__ == '__main__':
    raise SystemExit(main())
