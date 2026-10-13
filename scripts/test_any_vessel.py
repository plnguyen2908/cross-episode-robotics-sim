"""Physical oracle type-selection check on a frozen current scene; no Qwen call."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example', type=Path, default=Path('datasets/test_set_v1/examples/example_0002'))
    p.add_argument('--asset-type', default='Mug')
    p.add_argument('--gpu', default='7')
    p.add_argument('--output', type=Path)
    p.add_argument('--render-video', action='store_true')
    args = p.parse_args()
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    os.environ.update(worker_environment(os.environ, args.gpu, resolve_gpus([args.gpu])[args.gpu]))
    os.environ['MUJOCO_GL'] = 'egl'
    from cross_episode_sim.evaluation.coffee import from_example
    from cross_episode_sim.evaluation.dataset import write_json
    root = args.output or Path(tempfile.mkdtemp(prefix='any_vessel_smoke_', dir='runs'))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f'SMOKE_OUTPUT={root}', flush=True)
    session = from_example(example=args.example, output_root=str(root/'scene'), any_vessel=True)
    result = dict(success=False, scope='type-selected navigation and pick only; not full-task success', calls=[])
    try:
        for name, params in [('navigate', dict()), ('pick', dict(source='surface'))]:
            arguments = dict(asset_type=args.asset_type, **params)
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
