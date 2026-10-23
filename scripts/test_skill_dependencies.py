"""Physical dependency smoke check on a frozen scene; no Qwen call."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example', type=Path, default=Path('datasets/test_set_v1/examples/example_0002'))
    p.add_argument('--case', choices=['box', 'attach'], default='box')
    p.add_argument('--gpu', default='7')
    p.add_argument('--output', type=Path)
    p.add_argument('--render-video', action='store_true')
    args = p.parse_args()
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    os.environ.update(worker_environment(os.environ, args.gpu, resolve_gpus([args.gpu])[args.gpu]))
    os.environ['MUJOCO_GL'] = 'egl'
    from cross_episode_sim.evaluation.coffee import from_example
    from cross_episode_sim.evaluation.dataset import write_json
    root = args.output or Path(tempfile.mkdtemp(prefix='skill_dependencies_', dir='runs'))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f'SMOKE_OUTPUT={root}', flush=True)
    session = from_example(example=args.example, output_root=str(root/'scene'), any_vessel=True)
    result = dict(success=False, scope='physical dependency checks; not full-task success', calls=[])
    try:
        calls = ([('press_power', {}), ('navigate', dict(asset_type='CoffeeDosingBox')),
                  ('pick_dosing_box', {}), ('wait_until_ready', {}),
                  ('navigate', dict(asset_type='Mug'))] if args.case == 'box'
                 else [('detach_portafilter', {}), ('attach_portafilter', {})])
        result['case'] = args.case
        result['scope'] = 'physical dependency checks; not full-task success'
        result['initial_visible_bodies'] = sorted(session.interface.visible_bodies())
        for name, arguments in calls:
            session.validate_skill(name, arguments)
            outcome = session.oracle_skill(name, arguments)
            result['calls'].append(dict(skill=name, arguments=arguments, result=outcome))
            write_json(root/'result.json', result)
            if not outcome['success']:
                break
        else:
            result['success'] = True  # Each successful skill has already verified its physical postcondition.
        if args.render_video:
            result['video'] = session.finalize_video(dict(success=False, termination='smoke_test', goals=session.score()), root)
    finally:
        write_json(root/'result.json', result)
        session.close()
    print(json.dumps(dict(success=result['success'], output=str(root))))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
