"""Oracle-skill evaluation of a generated coffee/current episode's exact layout."""
import json
from pathlib import Path

from cross_episode_sim.evaluation.dataset import load_initial_scene, scene_layout
from cross_episode_sim.evaluation.sessions import ControllerSession


def worker_task(entry):
    """Resolve public composition names to their physical execution task."""
    task = entry.get('worker_task', entry['task'])
    if task == 'breakfast_with_coffee':
        task = 'full_breakfast'
    if task not in ('coffee', 'full_breakfast'):
        raise ValueError(f"Unsupported coffee evaluation task: {entry['task']!r} (worker {task!r})")
    return task


def from_example(*, example, output_root, seed=None, episode='current',
                 cameras=('robot_0/wrist_cam',), image_size=(224, 224), visual_grounding=False, any_vessel=True):
    """The model never receives run_manifest, the witness plan, or current video.

    The example fixes the scene seed; the evaluator seed controls the model trial.
    VLA/direct coffee execution is deliberately unsupported until its machine
    events and success predicates are independent of demonstrator phase flags.
    """
    from cross_episode_sim.tasks.coffee.workflow import main
    from cross_episode_sim.tasks.full_breakfast.__main__ import command
    root = Path(example)
    entry = json.loads((root / 'run_manifest.json').read_text())['episodes'][episode]
    settings = dict(entry['settings'], prepare_only=False)
    invocation = command(worker_task(entry), settings, Path(output_root), 'Model evaluation')
    arguments = invocation[invocation.index('-m')+2:]
    if any_vessel:
        arguments.append('--dynamic-clutter')
    captured = []

    def handle(controller, interface, goal, initialize):
        layouts = root / 'scene_layouts.json'
        original = root / 'runs' / episode / 'task_manifest.json'
        expected = (json.loads(layouts.read_text())[episode] if layouts.exists() else
                    scene_layout(json.loads(original.read_text())) if original.exists() else None)
        if expected is not None and scene_layout(controller.manifest) != expected:
            raise ValueError('Recreated scene differs from the recorded layout; regenerate this example with the current scene code')
        initialize()
        captured.append(ControllerSession(controller, interface, goal, cameras=cameras, image_size=image_size, visual_grounding=visual_grounding, any_vessel=any_vessel))
        return 0

    snapshot = root / 'initial_scenes' / episode
    loader = (lambda output: load_initial_scene(snapshot, output)) if (snapshot / 'snapshot.json').exists() else None
    main(arguments, execution_handler=handle, scene_loader=loader)
    if len(captured) != 1:
        raise RuntimeError('Coffee evaluation did not receive an initialized skill interface')
    return captured[0]
