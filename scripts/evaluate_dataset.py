"""Evaluate Qwen + oracle skills on collected examples, recreating each exact current scene."""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation.dataset import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('datasets/successful_examples'))
    p.add_argument('--output', type=Path)
    p.add_argument('--ids', nargs='+', help='Select exact IDs from dataset.json')
    p.add_argument('--limit', type=int)
    p.add_argument('--unique-layouts', action='store_true', help='Evaluate one example per layout fingerprint')
    p.add_argument('--gpu', default='3', help='Simulator GPU; inference server uses its own GPU')
    p.add_argument('--request-timeout', type=float, default=1800., help='Seconds to wait for each model response')
    p.add_argument('--model', default='Qwen/Qwen3.5-9B')
    p.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument('--history-frames', type=int, help='Legacy image-only ablation')
    sampling.add_argument('--video-fps', type=float, help='Override Qwen processor video sampling rate')
    p.add_argument('--max-planner-steps', type=int, help='Short interface test; an incomplete task remains unsuccessful')
    p.add_argument('--wall-seconds', type=float, default=10800.)
    args = p.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if args.limit is not None and args.limit < 1 or args.max_planner_steps is not None and args.max_planner_steps < 1:
        p.error('limit and max-planner-steps must be positive')
    dataset = args.dataset.resolve()
    examples = json.loads((dataset / 'dataset.json').read_text())['examples']
    if args.ids:
        if set(args.ids) - {e['id'] for e in examples}:
            p.error('Unknown dataset example ID')
        examples = [e for e in examples if e['id'] in args.ids]
    if args.unique_layouts:
        seen = set(); selected = []
        for entry in examples:
            if entry['layout_fingerprint'] not in seen:
                selected.append(entry); seen.add(entry['layout_fingerprint'])
        examples = selected
    examples = examples[:args.limit]
    if not examples:
        p.error('Dataset selection is empty')
    if args.output:
        root = args.output.resolve(); root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists():
            root = Path(tempfile.mkdtemp(prefix=root.name + '_', dir=root.parent))
        else:
            root.mkdir()
    else:
        Path('runs').mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='qwen_oracle_evaluation_', dir='runs')).resolve()
    print(f'EVALUATION={root}', flush=True)
    records = []
    for i, entry in enumerate(tqdm(examples, desc='Evaluation examples', unit='example', dynamic_ncols=True)):
        example = (dataset / entry['path']).resolve()
        if not example.is_relative_to(dataset):
            raise ValueError('Dataset example must be inside dataset root')
        trial = root / f'{i:03d}_{entry["id"]}'
        budgets = dict(wall_seconds=args.wall_seconds)
        if args.max_planner_steps:
            budgets['tool_calls'] = 2 * args.max_planner_steps
        command = [sys.executable, '-m', 'scripts.evaluate_models', '--planner', 'vlm', '--backend', 'oracle',
                   '--model', 'cross_episode_sim.evaluation.qwen:QwenPlanner',
                   '--model-options', json.dumps(dict(model=args.model, endpoint=args.endpoint, history_frames=args.history_frames, video_fps=args.video_fps, timeout=args.request_timeout)),
                   '--session', 'cross_episode_sim.evaluation.coffee:from_example',
                   '--session-options', json.dumps(dict(example=str(example), any_vessel=True, visual_grounding=False, cameras=['robot_0/wrist_cam'], image_size=[336, 448])),
                   '--history-example', str(example),
                   '--gpu', args.gpu, '--seed', str(entry['seed']), '--output', str(trial), '--budgets', json.dumps(budgets)]
        write_json(root / f'{i:03d}_command.json', dict(command=command, example=entry))
        tqdm.write(f'Example {i+1}/{len(examples)}: {entry["id"]}')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            with (root / f'{i:03d}.log').open('w') as log:
                for line in process.stdout:
                    log.write(line); log.flush(); tqdm.write(line.rstrip('\n'))
            code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
        result_path = trial / 'episode_000' / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(success=False, error=f'Missing result, exit {code}')
        journal = trial / 'episode_000' / 'events.jsonl'
        calls = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        oracle = [e for e in calls if e['kind'] == 'oracle_skill']
        records.append(dict(example_id=entry['id'], result=result, worker_exit=code,
                            oracle_calls=len(oracle), oracle_successes=sum(e['result']['success'] for e in oracle),
                            output=str(trial.relative_to(root))))
        write_json(root / 'summary.json', dict(model=args.model, dataset=str(dataset),
                   scope='limited_skill_smoke_test' if args.max_planner_steps else 'full_task_evaluation',
                   examples=records, success_rate=sum(r['result']['success'] for r in records)/len(records)))
    return 0 if all(r['worker_exit'] == 0 and not r['result'].get('error') for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
