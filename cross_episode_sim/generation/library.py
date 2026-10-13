"""History library API. Records are generator metadata, never planner inputs."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from cross_episode_sim.evaluation.dataset import inspect_example, write_json
from cross_episode_sim.tasks.full_breakfast.__main__ import (
    PROMPT,
    VIDEO,
    command,
    create_output,
    sample,
    summary,
)

CONTRACT = 'breakfast-coffee-v1'
CATEGORIES = ('counting', 'imitation', 'temporal', 'preference')


@dataclass(frozen=True)
class HistoryConfig:
    task: str
    seed: int
    people: int | None = None  # Legacy constructor/record compatibility.
    layout_profile: str = 'varied'
    scene: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self):
        if self.task not in ('coffee', 'breakfast'):
            raise ValueError('Implemented histories: coffee (imitation), breakfast (counting). '
                             'Temporal and preference tasks are not implemented yet.')
        if not isinstance(self.parameters, dict) or not isinstance(self.scene, dict):
            raise ValueError('parameters and scene must be objects')
        key = 'cups' if self.task == 'coffee' else 'people'
        if set(self.parameters) - {key}:
            raise ValueError(f'{self.task} history parameters: {key}')
        count = self.parameters.get(key, self.people if self.people is not None else 1)
        if self.people is not None and (type(self.people) is not int or self.people != count):
            raise ValueError('Legacy people conflicts with parameters')
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError('name must be a nonempty display name')
        object.__setattr__(self, 'parameters', {key: count})
        object.__setattr__(self, 'people', count)
        _validate(self.seed, count, self.layout_profile, self.scene)
        _task_options(self.task, self.scene)
        if self.task == 'coffee' and self.people != 1:
            raise ValueError('Coffee histories demonstrate exactly one cup')


@dataclass(frozen=True)
class CurrentPlan:
    """Task-owned composition result; settings remain private generator metadata."""
    worker_task: str
    settings: dict
    prompt: str
    video: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CurrentTaskDefinition:
    """Callbacks: validate_config(config), compose(config, history_records),
    execute(plan, output, gpu), verify(plan, run). Execution writes runs/current
    through public atomic skills; verification raises on unmet postconditions.
    Generic physical-goal and video validation always runs too.
    """
    validate_config: Callable
    compose: Callable
    execute: Callable
    verify: Callable
    version: str = '1'
    history_requirements: Callable | None = None


_CURRENT_TASKS = {}


def register_current_task(name, definition):
    if not isinstance(name, str) or not name or name in _CURRENT_TASKS:
        raise ValueError(f'Invalid or duplicate current task name: {name!r}')
    if not isinstance(definition, CurrentTaskDefinition):
        raise TypeError('Expected CurrentTaskDefinition')
    _CURRENT_TASKS[name] = definition


def current_task(name):
    try:
        return _CURRENT_TASKS[name]
    except KeyError:
        raise ValueError(f'Unknown current task {name!r}; registered: {sorted(_CURRENT_TASKS)}') from None


@dataclass(frozen=True)
class CurrentConfig:
    seed: int
    histories: tuple[str, ...]
    task: str = 'breakfast_with_coffee'
    layout_profile: str = 'varied'
    scene: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if not self.histories or isinstance(self.histories, str):
            raise ValueError('Provide a nonempty list of history directories')
        if not isinstance(self.parameters, dict) or not isinstance(self.scene, dict):
            raise ValueError('parameters and scene must be objects')
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError('name must be a nonempty display name')
        current_task(self.task).validate_config(self)


def _validate(seed, people, profile, scene):
    from cross_episode_sim.tasks.servings import validate_people
    validate_people(people)
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if profile not in ('baseline', 'varied'):
        raise ValueError('layout_profile must be baseline or varied')
    allowed = {'extra_furniture', 'floor_objects', 'table_objects', 'clutter',
               'machine_room', 'extra_vessels', 'nav_speed', 'turn_speed'}
    if set(scene) - allowed:
        raise ValueError(f'Unknown scene options: {sorted(set(scene) - allowed)}')
    for key, value in scene.items():
        if key == 'machine_room':
            if value not in ('kitchen', 'living', 'dining'):
                raise ValueError('machine_room must be kitchen, living, or dining')
        elif key in ('nav_speed', 'turn_speed'):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        elif type(value) is not int or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if scene.get('extra_furniture', 0) > 8 or scene.get('extra_vessels', 0) > 6:
        raise ValueError('extra_furniture maximum is 8; extra_vessels maximum is 6')



def _task_options(task, scene):
    unsupported = ({'clutter', 'machine_room', 'extra_vessels'} if task == 'breakfast'
                   else {'table_objects', 'extra_vessels'} if task == 'coffee'
                   else {'table_objects'}) & set(scene)
    if unsupported:
        raise ValueError(f'Scene options not used by {task}: {sorted(unsupported)}')


def _validate_count(run, expected):
    if summary(run)['serving_count'] != expected:
        raise ValueError(f'Generated serving count does not match requested count {expected}: {run}')

def _settings(config, people):
    worker_task = 'full_breakfast' if config.task == 'breakfast_with_coffee' else config.task
    settings = sample(config.seed, config.layout_profile, people)[1][worker_task]
    settings.update(config.scene)
    if worker_task == 'full_breakfast':
        settings.setdefault('extra_vessels', 3)
        if settings['extra_vessels'] < 2:
            raise ValueError('Current requires a fixed surplus pool: extra_vessels >= 2 (at least 5 per type)')
    return settings


def run_episode(task, settings, output, name, gpu='0', *, progress=None):
    """Run one existing physical worker, with bounded failure shutdown and progress."""
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    from cross_episode_sim.tasks.full_breakfast.fail_fast import definitive_failure, stop_workers
    from cross_episode_sim.tasks.full_breakfast.progress import format_progress, snapshot
    env = dict(os.environ)
    if settings.get('motion_source'):
        env['CES_MOTION_SOURCE'] = str(Path(settings['motion_source']).resolve())
    env.setdefault('MUJOCO_GL', 'egl')
    repo = Path(__file__).resolve().parents[2]
    if (repo / 'external/robocasa/robocasa').exists():
        env.setdefault('ROBOCASA_DIR', str(repo / 'external/robocasa/robocasa'))
    gpu = str(gpu)
    binding = resolve_gpus([gpu])[gpu] if env['MUJOCO_GL'] == 'egl' else None
    log_path = output / 'runs' / f'{name}.log'
    print(f'{output.name}: {name} ({task}), GPU {gpu}, log {log_path}', flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(command(task, settings, output / 'runs' / name, name),
                                   env=worker_environment(env, gpu, binding), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        workers = {name: (process, log)}
        try:
            next_update = 0.
            while True:
                code = process.poll()
                failure = definitive_failure(output, {name: code})
                if failure:
                    if progress:
                        progress(snapshot(output, name, code))
                    write_json(output / 'generation_failure.json', failure)
                    raise RuntimeError(f"{name}: {failure['error']}")
                if time.monotonic() >= next_update or code is not None:
                    row = snapshot(output, name, code)
                    print(format_progress(row), flush=True)
                    if progress:
                        progress(row)
                    next_update = time.monotonic() + 10
                if code is not None:
                    break
                time.sleep(1)
        except BaseException:
            # The worker may have exited while a renderer in its group survives.
            stop_workers(workers)
            raise


def _validate_run(run, task=None, *, video_name=None):
    report = json.loads((run / 'report.json').read_text())
    goals = report.get('composite_goal_evidence')
    if report.get('success') is not True or not goals or any(v is not True for v in goals.values()):
        raise ValueError(f'Unsuccessful physical goals: {run}')
    if report.get('video_status') != 'complete':
        raise ValueError(f'Incomplete rendering: {run}')
    video = run / (video_name or VIDEO[task])
    if not video.resolve().is_relative_to(run.resolve()):
        raise ValueError('Video must be inside the run directory')
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        first = capture.read()[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
        if frames < 1 or not first or not capture.read()[0]:
            raise ValueError(f'Unreadable video: {video}')
    finally:
        capture.release()
    return dict(goals=goals, frames=frames)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def history_config_dict(config):
    """Write the shared schema while continuing to read legacy people records."""
    record = asdict(config)
    record.pop('people')
    return record


def create_history(config: HistoryConfig, library, *, gpu='0', progress=None):
    """Generate one validated history in a unique library directory; return its path."""
    settings = _settings(config, config.people)
    root = create_output(None, config.seed, Path(library))
    (root / 'runs').mkdir()
    write_json(root / 'config.json', history_config_dict(config))
    write_json(root / 'run_manifest.json', dict(skill_interface_required=True,
               episodes={'history_1': dict(task=config.task, settings=settings)}))
    if progress:
        run_episode(config.task, settings, root, 'history_1', gpu, progress=progress)
        progress(dict(status='validating', step=None, total=None, operation='video and physical goals', stage=''))
    else:
        run_episode(config.task, settings, root, 'history_1', gpu)
    run = root / 'runs/history_1'
    evidence = _validate_run(run, config.task)
    _validate_count(run, config.people)
    files = ['report.json', 'task_manifest.json', VIDEO[config.task]]
    record = dict(version=1, id=root.name, contract=CONTRACT, task=config.task,
                  category='imitation' if config.task == 'coffee' else 'counting',
                  config=history_config_dict(config), settings=settings, run='runs/history_1',
                  validation=evidence, sha256={f: _digest(run / f) for f in files})
    # Publish only after physical goals and video validation succeed.
    write_json(root / 'history.json', record)
    return root


def load_history(path):
    root = Path(path).resolve()
    record = json.loads((root / 'history.json').read_text())
    if record.get('version') != 1 or record.get('contract') != CONTRACT:
        raise ValueError(f'Incompatible history contract: {root}')
    config = HistoryConfig(**record['config'])
    if record['task'] != config.task or record['settings']['people'] != config.people:
        raise ValueError('History metadata disagrees with its configuration')
    run = (root / record['run']).resolve()
    if not run.is_relative_to(root):
        raise ValueError('History run must be inside its library directory')
    for filename in ('report.json', 'task_manifest.json', VIDEO[config.task]):
        if record['sha256'].get(filename) != _digest(run / filename):
            raise ValueError(f'History changed since validation: {root}/{filename}')
    _validate_run(run, config.task)
    _validate_count(run, config.people)
    return root, run, record


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def create_current(config: CurrentConfig, output=None, *, gpu='0'):
    """Compose validated histories, execute only current, emit legacy-compatible example."""
    histories = [load_history(path) for path in config.histories]
    definition = current_task(config.task)
    plan = definition.compose(config, [record for _, _, record in histories])
    if not isinstance(plan, CurrentPlan) or not plan.prompt.strip():
        raise ValueError('Task compose must return a CurrentPlan with a nonempty prompt')
    settings = plan.settings
    root = create_output(output, config.seed)
    (root / 'runs').mkdir()
    write_json(root / 'config.json', asdict(config))
    episodes, manifest = [], {}
    for index, (source, run, record) in enumerate(histories, 1):
        name = f'history_{index}'
        # Materialize compatible run paths; no history physics is rerun.
        shutil.copytree(run, root / 'runs' / name, copy_function=_link_or_copy)
        _link_or_copy(run / VIDEO[record['task']], root / f'{name}.mp4')
        manifest[name] = dict(task=record['task'], settings=record['settings'],
                              history_id=record['id'], history_source=str(source))
        episodes.append(dict(name=name, task=record['task'], role='history',
                             video=f'{name}.mp4', run=f'runs/{name}',
                             settings=record['settings'], history_id=record['id'], **summary(run)))
    manifest['current'] = dict(task=config.task, worker_task=plan.worker_task, settings=settings,
                               label=config.name or config.task, task_version=definition.version)
    write_json(root / 'run_manifest.json', dict(seed=config.seed, skill_interface_required=True,
               history_contract=CONTRACT, prepare_only=False, episodes=manifest))
    definition.execute(plan, root, gpu)
    run = root / 'runs/current'
    _validate_run(run, video_name=plan.video)
    definition.verify(plan, run)
    _link_or_copy(run / plan.video, root / 'current.mp4')
    episodes.append(dict(name='current', task=config.task, role='current', video='current.mp4',
                         run='runs/current', settings=settings, **summary(run)))
    (root / 'prompt.txt').write_text(plan.prompt)
    write_json(root / 'episode.json', dict(seed=config.seed, episodes=episodes,
               history_order=[f"{e['name']}: {e['task']}" for e in episodes[:-1]],
               current=dict(task=config.task, name=config.name or config.task,
                            prompt=plan.prompt.strip(), video='current.mp4'),
               history_ids=[r['id'] for _, _, r in histories]))
    write_json(root / 'validation.json', inspect_example(root))
    return root


def _validate_breakfast_current(config):
    _validate(config.seed, 1, config.layout_profile, config.scene)
    _task_options('full_breakfast', config.scene)
    if set(config.parameters) - {'people'}:
        raise ValueError('breakfast_with_coffee parameters: people (optional, must match history)')
    if 'people' in config.parameters and (type(config.parameters['people']) is not int or config.parameters['people'] not in (1, 2, 3)):
        raise ValueError('Current sampling supports 1–3 people')
    if len(config.histories) != 2:
        raise ValueError('Provide exactly one coffee and one breakfast history directory')


def _compose_breakfast_current(config, histories):
    by_task = {record['task']: record for record in histories}
    if len(histories) != 2 or set(by_task) != {'coffee', 'breakfast'}:
        raise ValueError('Current needs one coffee and one breakfast history')
    people = HistoryConfig(**by_task['breakfast']['config']).people
    if config.parameters.get('people', people) != people:
        raise ValueError('Current people must match the breakfast history')
    settings = _settings(config, people)
    return CurrentPlan('full_breakfast', settings, PROMPT, VIDEO['full_breakfast'], parameters={'people': people})


def _execute_breakfast_current(plan, output, gpu):
    run_episode(plan.worker_task, plan.settings, output, 'current', gpu)


def _verify_breakfast_current(plan, run):
    _validate_count(run, plan.settings['people'])


def _breakfast_history_requirements(request):
    return {
        'coffee': dict(task='coffee', category='imitation', contract=CONTRACT, parameters={'cups': 1}),
        'breakfast': dict(task='breakfast', category='counting', contract=CONTRACT,
                          parameters={'people': request.get('parameters', {}).get('people', [1, 2, 3])}),
    }


register_current_task('full_breakfast', CurrentTaskDefinition(
    validate_config=_validate_breakfast_current, compose=_compose_breakfast_current,
    execute=_execute_breakfast_current, verify=_verify_breakfast_current,
    history_requirements=_breakfast_history_requirements,
))

# Keep previously written configs readable.
register_current_task('breakfast_with_coffee', current_task('full_breakfast'))
