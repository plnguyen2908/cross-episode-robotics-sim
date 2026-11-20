#!/usr/bin/env python3
"""Run several household episodes in parallel from ONE terminal, one per GPU.

    python3 scripts/eval_households.py --suite all --gpus 2,5,6,7
    python3 scripts/eval_households.py <episode folder> <episode folder> ... --gpus 2,5

Each episode runs through scripts/eval_household.sh (our Qwen sensor pipeline from
this checkout, no wall-time or sim-time limit). Live lines from every run are shown
with a [task-rooms] tag: stages, public calls and the private task progress after
every call. When a run ends its GPU takes the next queued episode. A summary table
(success, progress, calls, wall time) is printed at the end and written to
<output>/summary.json. Ctrl-C stops every run.

    python3 scripts/eval_households.py --split test --gpus 2,5,6,7

--split runs every episode of v4 splits (from their manifests) for --tasks (default:
set_table, warm_two_pots, unpack_and_sort_groceries); --split test = test_same_map and
test_new_map. Success rate and mean progress rate are reported overall, by room
count, by task and by task x room count (live after each run, and summary_by_rooms.json).

Suites (episodes under runs/): two_room = the three 2-room episodes that passed
(fix186 pots, fix187 set table, fix190 grocery); multi_room = one v4 test_same_map
episode per task with 3, 4 and 5 rooms; all = both (12 runs).
"""
import argparse, datetime, json, os, queue, signal, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = 'runs/all_open_fixed_v3/{split}/batches/round_continuous/{name}'
V4 = 'runs/all_open_fixed_v4/test_same_map/batches/round_continuous/{name}'
SUITES = {
    'two_room': [V3.format(split='fine_tuning', name='warm_two_pots_000000'),
                 V3.format(split='test_same_map', name='set_table_000017'),
                 V3.format(split='test_same_map', name='unpack_and_sort_groceries_000017')],
    'multi_room': [V4.format(name=n) for n in (
        'warm_two_pots_000024', 'set_table_000028', 'unpack_and_sort_groceries_000022',     # 3 rooms
        'warm_two_pots_000023', 'set_table_000029', 'unpack_and_sort_groceries_000028',     # 4 rooms
        'warm_two_pots_000022', 'set_table_000032', 'unpack_and_sort_groceries_000021')]}   # 5 rooms
SUITES['all'] = SUITES['two_room'] + SUITES['multi_room']
SHORT = {'warm_two_pots': 'pots', 'set_table': 'table', 'unpack_and_sort_groceries': 'grocery',
         'clear_and_store': 'store', 'prepare_and_serve_meal': 'meal'}


def episode_meta(source):
    config = json.loads((source / 'config.json').read_text())
    return config['task'], (config.get('dataset_sampling') or {}).get('requested_room_count')


def label(source):
    task, rooms = episode_meta(source)
    return f"{SHORT.get(task, task)}-{rooms}r-{source.name.rsplit('_', 1)[-1]}"


def rates(rows):
    """Success rate and mean progress rate; a run that ended without a report counts as 0 for both."""
    progress = [r.get('progress_rate') or 0. for r in rows]
    return dict(runs=len(rows), successes=sum(r.get('success') is True for r in rows),
                success_rate=round(sum(r.get('success') is True for r in rows) / len(rows), 3) if rows else None,
                mean_progress_rate=round(sum(progress) / len(progress), 3) if progress else None,
                runs_without_report=sum(r.get('progress_rate') is None for r in rows))


def by_rooms(results):
    rows = list(results.values())
    rooms = sorted({r['rooms'] for r in rows}, key=lambda x: (x is None, x))
    tasks = sorted({r['task'] for r in rows})
    return dict(overall=rates(rows),
                by_rooms={str(k): rates([r for r in rows if r['rooms'] == k]) for k in rooms},
                by_task={k: rates([r for r in rows if r['task'] == k]) for k in tasks},
                by_task_and_rooms={f'{k}/{n}r': rates([r for r in rows if r['task'] == k and r['rooms'] == n])
                                   for k in tasks for n in rooms if any(r['task'] == k and r['rooms'] == n for r in rows)})


def fmt(x):
    return f"{x['successes']}/{x['runs']} ok ({x['success_rate']:.0%}), progress {x['mean_progress_rate'] if x['mean_progress_rate'] is not None else '-'}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episodes', nargs='*', type=Path)
    p.add_argument('--suite', choices=sorted(SUITES))
    p.add_argument('--split', help='every episode of these dataset splits (comma list) from their manifests: '
                   'test (= test_same_map,test_new_map), test_same_map, test_new_map, dev_same_map, dev_new_map')
    p.add_argument('--dataset', type=Path, default=ROOT / 'runs/all_open_fixed_v4', help='dataset for --split')
    p.add_argument('--tasks', default='set_table,warm_two_pots,unpack_and_sort_groceries', help='tasks taken from --split')
    p.add_argument('--gpus', default='2,5,6,7', help='comma-separated GPU ids, one run per GPU at a time')
    p.add_argument('--output', type=Path, help='default runs/household_eval/batch_<time>')
    p.add_argument('--raw', action='store_true', help='show every line of every run')
    p.add_argument('--list', action='store_true', help='print the selected episodes and exit')
    a = p.parse_args()
    sources = [ROOT / s for s in SUITES[a.suite]] if a.suite else []
    if a.split:
        sys.path.insert(0, str(ROOT))
        from cross_episode_sim.evaluation.test_sources import test_sources
        splits = [s for x in a.split.split(',') for s in (['test_same_map', 'test_new_map'] if x == 'test' else [x])]
        for split in splits:
            for task in a.tasks.split(','):sources += test_sources(a.dataset, [split], task.strip())
    sources += [e.resolve() for e in a.episodes]
    if not sources:p.error('give episode folders or --suite')
    for s in sources:
        if not (s / 'config.json').exists():p.error(f'not an episode folder: {s}')
    if a.list:
        for s in sources:print(label(s), s)
        print(len(sources), 'episodes');return
    gpus = [g.strip() for g in a.gpus.split(',') if g.strip()]
    out = a.output or ROOT / 'runs/household_eval' / ('batch_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    jobs = queue.Queue()
    names = {}
    for s in sources:
        name = label(s)
        while name in names.values():name += '+'
        names[s] = name;jobs.put(s)
    width = max(len(n) for n in names.values())
    lock = threading.Lock();procs = {};results = {}
    started = time.monotonic()

    def say(tag, text):
        with lock:print(f'[{tag:<{width}}] {text}', flush=True)

    def worker(gpu):
        while True:
            try:source = jobs.get_nowait()
            except queue.Empty:return
            name = names[source];folder = out / name
            command = ['bash', str(ROOT / 'scripts/eval_household.sh'), str(source), '--gpu', gpu, '--output', str(folder)]
            if a.raw:command.append('--raw')
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, start_new_session=True)
            procs[name] = proc;t0 = time.monotonic()
            say(name, f'started on GPU {gpu}')
            for line in proc.stdout:
                line = line.rstrip('\n')
                # Drop the runner's timestamp prefix; keep the message.
                if ' | ' in line[:40]:line = line.split(' | ', 1)[1]
                say(name, line)
            code = proc.wait();procs.pop(name, None)
            try:report = json.loads((folder / 'episode/report.json').read_text())
            except Exception:report = {}
            prog = report.get('progress') or {}
            task, rooms = episode_meta(source)
            results[name] = dict(source=str(source), task=task, rooms=rooms, gpu=gpu, exit_code=code, success=report.get('success'),
                                 progress_done=prog.get('done'), progress_total=prog.get('total'), progress_rate=prog.get('rate'),
                                 completed_calls=report.get('completed_calls'), planned_calls=report.get('planned_calls'),
                                 wall_minutes=round((time.monotonic() - t0) / 60, 1), output=str(folder))
            (out / 'summary.json').write_text(json.dumps(results, indent=2))
            stats = by_rooms(results);(out / 'summary_by_rooms.json').write_text(json.dumps(stats, indent=2))
            say(name, f"finished: success={report.get('success')} progress={prog.get('done')}/{prog.get('total')} "
                      f"in {results[name]['wall_minutes']} min")
            say('eval', f"so far {len(results)}/{len(sources)} done | " +
                ' | '.join(f"{k} rooms: {fmt(v)}" for k, v in stats['by_rooms'].items()))

    def stop(*_):
        say('eval', 'stopping all runs')
        running = list(procs.values())
        for proc in running:
            try:os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:pass
        deadline = time.monotonic() + 20
        for proc in running:
            try:proc.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:pass
        say('eval', f'stopped; partial logs in {out}')
        os._exit(130)
    signal.signal(signal.SIGINT, stop)
    say('eval', f'{len(sources)} episodes on GPUs {",".join(gpus)} -> {out}')
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for i, t in enumerate(threads):
        t.start()
        if i < len(threads) - 1:time.sleep(20)   # stagger startup (model loads, SLAM init)
    while any(t.is_alive() for t in threads):time.sleep(1)
    print(f'\n== {len(results)} runs in {round((time.monotonic() - started) / 60, 1)} min -> {out}/summary.json')
    print(f"{'run':<{width}}  success  progress  calls   wall")
    for name in names.values():
        r = results.get(name, {})
        print(f"{name:<{width}}  {str(r.get('success')):<7}  {r.get('progress_done')}/{r.get('progress_total'):<6}  "
              f"{r.get('completed_calls')}/{r.get('planned_calls'):<4}  {r.get('wall_minutes')} min")
    if results:
        stats = by_rooms(results)
        print(f"\n== success rate and mean progress rate -> {out}/summary_by_rooms.json")
        print(f"{'group':<34} runs  success   progress")
        def row(name, x):
            print(f"{name:<34} {x['runs']:<5} {x['successes']}/{x['runs']} {x['success_rate']:>5.0%}   "
                  f"{x['mean_progress_rate'] if x['mean_progress_rate'] is not None else '-'}")
        row('all', stats['overall'])
        for k, x in stats['by_rooms'].items():row(f'{k} rooms', x)
        for k, x in stats['by_task'].items():row(k, x)
        for k, x in stats['by_task_and_rooms'].items():row(k, x)


if __name__ == '__main__':
    main()
