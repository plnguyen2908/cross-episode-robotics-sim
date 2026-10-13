"""Evaluate a VLM planner or coding agent with oracle, VLA, or direct actions."""
import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation import Budgets, evaluate
from cross_episode_sim.evaluation.history import history_context
from cross_episode_sim.evaluation.sessions import TaskSession


def load(spec, options):
    module, name = spec.split(':', 1)
    return getattr(importlib.import_module(module), name)(**json.loads(options))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planner', choices=('vlm', 'agent'), default='vlm')
    p.add_argument('--backend', choices=('oracle', 'vla', 'direct'), default='oracle')
    p.add_argument('--model', required=True, help='module:Class implementing the planner or agent protocol')
    p.add_argument('--model-options', default='{}', help='JSON keyword arguments')
    p.add_argument('--vla', help='module:Class implementing start_skill and act')
    p.add_argument('--vla-options', default='{}')
    p.add_argument('--session', help='Custom module:factory implementing the evaluation session protocol')
    p.add_argument('--session-options', default='{}')
    p.add_argument('--task', choices=('breakfast', 'tidy_up'), default='breakfast')
    p.add_argument('--task-options', default='{}')
    p.add_argument('--history-example', type=Path, help='Read public history videos and prompt only; does not select the current scene')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--gpu', help='Physical NVIDIA GPU index or UUID for both CUDA and EGL')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--output', type=Path, help='Output directory; existing paths get a unique suffix, default unique under runs')
    p.add_argument('--budgets', default='{}', help='JSON overrides for Budgets')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    if (args.backend == 'vla') != bool(args.vla):
        p.error('--vla is required exactly when --backend vla')
    if args.gpu is not None:
        from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
        binding = resolve_gpus([args.gpu])[args.gpu]
        os.environ.update(worker_environment(os.environ, args.gpu, binding))
        os.environ['MUJOCO_GL'] = 'egl'
    budget = Budgets(**json.loads(args.budgets))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.mkdir(exist_ok=False)
        except FileExistsError:
            output = Path(tempfile.mkdtemp(prefix=output.name+'_', dir=output.parent))
    else:
        Path('runs').mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='model_evaluation_', dir='runs')).resolve()
    print(f'OUTPUT={output}', flush=True)
    records = []
    for index in tqdm(range(args.episodes), desc='Evaluation episodes', unit='episode', dynamic_ncols=True):
        session = None
        result = None
        episode_output = output / f'episode_{index:03d}'
        tqdm.write(f'episode_{index:03d}: preparing (seed={args.seed+index})')
        try:
            # New instances prevent hidden model/agent memory leaking between trials.
            model = load(args.model, args.model_options)
            vla = load(args.vla, args.vla_options) if args.vla else None
            if args.session:
                options = json.loads(args.session_options)
                options.update(output_root=str(output / f'scene_{index:03d}'), seed=args.seed+index)
                session = load(args.session, json.dumps(options))
            else:
                session = TaskSession(args.task, output_root=output / f'scene_{index:03d}', seed=args.seed+index,
                                      options=json.loads(args.task_options), max_steps=budget.actions)
            context = (history_context(args.history_example) if args.history_example else
                       dict(instruction=session.instruction, histories=[]))
            context['seed'] = args.seed+index
            tqdm.write(f'episode_{index:03d}: evaluating ({args.planner} + {args.backend})')
            result = evaluate(model, session, episode_output, planner=args.planner,
                              backend=args.backend, vla=vla, budgets=budget, context=context, progress=True)
        except (Exception, SystemExit) as exc:
            result = dict(success=False, termination='evaluation_error', error=f'{type(exc).__name__}: {exc}',
                          planner=args.planner, skill_backend=args.backend)
        finally:
            if session is not None:
                if result is not None and hasattr(session, 'finalize_video'):
                    tqdm.write(f'episode_{index:03d}: rendering model rollout')
                    try:
                        result['video'] = session.finalize_video(result, episode_output)
                    except Exception as exc:
                        # Rendering failure must not change the measured task outcome.
                        result['video'] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
                try:
                    session.close()
                except Exception as exc:
                    result = dict(success=False, termination='cleanup_error',
                                  error=f'{type(exc).__name__}: {exc}',
                                  planner=args.planner, skill_backend=args.backend,
                                  evaluation_result=result)
        result.update(seed=args.seed+index, episode=index)
        episode_output.mkdir(parents=True, exist_ok=True)
        (episode_output / 'result.json').write_text(json.dumps(result, indent=2))
        records.append(result)
        summary = dict(success_rate=sum(r['success'] for r in records)/len(records), episodes=records,
                       model=args.model, vla=args.vla, planner=args.planner, backend=args.backend)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        tqdm.write(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
