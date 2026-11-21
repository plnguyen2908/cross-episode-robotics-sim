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
                runs_without_report=sum(r.get('progress_rate') is None for r in rows),
                terminations={t: sum((r.get('termination') or 'no_report') == t for r in rows)
                              for t in sorted({r.get('termination') or 'no_report' for r in rows})})


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


def rescore(batch):
    """Summaries of a batch from the reports on disk (reruns included, renamed failed starts excluded)."""
    batch = batch.resolve();results = {}
    for folder in sorted(p for p in batch.iterdir() if p.is_dir() and '_failedstart_' not in p.name):
        try:source = Path(json.loads((folder / 'run_info.json').read_text())['source'])
        except Exception:continue
        try:report = json.loads((folder / 'episode/report.json').read_text())
        except Exception:report = {}
        prog = report.get('progress') or {};task, rooms = episode_meta(source)
        results[folder.name] = dict(source=str(source), task=task, rooms=rooms, success=report.get('success'),
                                    progress_done=prog.get('done'), progress_total=prog.get('total'), progress_rate=prog.get('rate'),
                                    completed_calls=report.get('completed_calls'), planned_calls=report.get('planned_calls'),
                                 termination=report.get('termination'), simulated_minutes=round((report.get('simulated_seconds') or 0) / 60, 1),
                                    wall_minutes=round((report.get('wall_seconds') or 0) / 60, 1), output=str(folder),
                                    has_report=bool(report))
    (batch / 'summary.json').write_text(json.dumps(results, indent=2))
    stats = by_rooms(results);(batch / 'summary_by_rooms.json').write_text(json.dumps(stats, indent=2))
    missing = [k for k, v in results.items() if not v['has_report']]
    print(f"== {batch.name}: {len(results)} episodes, {len(missing)} without a report (counted as 0){': ' + ', '.join(missing) if missing else ''}")
    print(f"{'group':<34} runs  success   progress")
    for name, x in [('all', stats['overall'])] + [(f'{k} rooms', v) for k, v in stats['by_rooms'].items()] + \
            list(stats['by_task'].items()) + list(stats['by_task_and_rooms'].items()):
        print(f"{name:<34} {x['runs']:<5} {x['successes']}/{x['runs']} {x['success_rate']:>5.0%}   "
              f"{x['mean_progress_rate'] if x['mean_progress_rate'] is not None else '-'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episodes', nargs='*', type=Path)
    p.add_argument('--suite', choices=sorted(SUITES))
    p.add_argument('--split', help='every episode of these dataset splits (comma list) from their manifests: '
                   'test (= test_same_map,test_new_map), test_same_map, test_new_map, dev_same_map, dev_new_map')
    p.add_argument('--dataset', type=Path, default=ROOT / 'runs/all_open_fixed_v4', help='dataset for --split')
    p.add_argument('--tasks', default='set_table,warm_two_pots,unpack_and_sort_groceries', help='tasks taken from --split')
    p.add_argument('--gpus', default='2,5,6,7', help='comma-separated GPU ids, one run per GPU at a time')
    p.add_argument('--endpoints', help='comma-separated service endpoints files (scripts/start_eval_services.sh '
                   '--sets): the worker of --gpus entry i uses file i (cycled), so each episode slot has its own '
                   'model services; default: SERVICE_ENDPOINTS_FILE or the built-in addresses for every worker')
    p.add_argument('--output', type=Path, help='default runs/household_eval/batch_<time>')
    p.add_argument('--raw', action='store_true', help='show every line of every run')
    p.add_argument('--list', action='store_true', help='print the selected episodes and exit')
    p.add_argument('--gpt-limits', action='store_true', help='the GPT agent-as-policy eval budgets: 45 wall min, 15 simulated min, 100 calls')
    p.add_argument('--wall-minutes', type=float, help='per-episode wall limit; the live state is scored when reached')
    p.add_argument('--sim-minutes', type=float, help='per-episode simulated-time limit')
    p.add_argument('--max-calls', type=int, help='per-episode limit on plan calls')
    p.add_argument('--planner-endpoint', default='127.0.0.1:51510', help='Qwen worker for zero-shot planning (not the perception worker)')
    p.add_argument('--planner', choices=('recipe', 'qwen_zero_shot', 'qwen_zero_shot_pure', 'qwen_replan', 'gpt_zero_shot', 'gpt_replan'), default='recipe',
                   help='recipe (fixed plan, default); qwen_zero_shot (Qwen writes the whole plan with the contract rules and '
                        'up to 2 repair requests, executed open loop); qwen_zero_shot_pure (one request, no rules); '
                        'qwen_replan (qwen_zero_shot plus a new rest-of-plan from Qwen after each failed call); '
                        'gpt_zero_shot / gpt_replan (the same with GPT through the OpenAI API: OPENAI_API_KEY, '
                        'PLANNER_MODEL default gpt-6.1-sol, PLANNER_EFFORT default high)')
    p.add_argument('--into', type=Path, help='rerun into this existing batch folder: an episode folder there '
                   'without a report is renamed <name>_failedstart_<n> (kept) and replaced; one with a report is skipped')
    p.add_argument('--rescore', type=Path, help='rebuild summary.json and summary_by_rooms.json of a batch folder '
                   'from its episode reports (renamed *_failedstart_* folders excluded), print the tables and exit')
    a = p.parse_args()
    if a.planner.startswith('gpt_') and not a.list and not a.rescore:
        # Fail in seconds, before any simulator starts, on a missing key or a wrong model ID.
        if not os.environ.get('OPENAI_API_KEY'):p.error('--planner gpt_* needs OPENAI_API_KEY in the environment')
        sys.path.insert(0, str(ROOT))
        from cross_episode_sim.evaluation.gpt_planner import check_planner_model, PlannerServiceError
        try:print('planner model:', check_planner_model() or os.environ.get('PLANNER_MODEL', 'gpt-6.1-sol') + ' (key cannot list models; not checked)', flush=True)
        except PlannerServiceError as exc:p.error(str(exc))
    if a.gpt_limits:a.wall_minutes, a.sim_minutes, a.max_calls = a.wall_minutes or 45., a.sim_minutes or 15., a.max_calls or 100
    if a.rescore:
        rescore(a.rescore)
        sys.path.insert(0, str(ROOT / 'scripts'))
        from collect_results import collect
        print(f'all_results.jsonl: {collect(a.rescore)} episodes');return
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
    summary_name = 'summary.json'
    if a.into:
        out = a.into.resolve()
        if not out.is_dir():p.error(f'--into needs an existing batch folder: {out}')
        used = {json.loads(i.read_text()).get('planner', 'recipe') for i in out.glob('*/run_info.json')}
        if used - {a.planner}:p.error(f'{out.name} was run with planner {sorted(used)}; pass the same --planner')
        want = dict(wall_minutes=a.wall_minutes, sim_minutes=a.sim_minutes, max_calls=a.max_calls)
        for info in out.glob('*/run_info.json'):
            had = json.loads(info.read_text()).get('limits')
            had = had if isinstance(had, dict) else dict(wall_minutes=None, sim_minutes=None, max_calls=None)
            if {k: (float(v) if v is not None else None) for k, v in had.items()} != {k: (float(v) if v is not None else None) for k, v in want.items()}:
                p.error(f'{info.parent.name} ran with limits {had}; pass the same limits')
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S');summary_name = f'summary_rerun_{stamp}.json'
        kept = []
        for s in sources:
            old = out / label(s)
            if (old / 'episode/report.json').exists():
                print(f'skip {old.name}: it already has a report');continue
            if old.exists():
                n = 1
                while (out / f'{old.name}_failedstart_{n}').exists():n += 1
                old.rename(out / f'{old.name}_failedstart_{n}');print(f'kept the failed start as {old.name}_failedstart_{n}')
            kept.append(s)
        sources = kept
        if not sources:print('nothing to rerun');return
    else:
        out = a.output or ROOT / 'runs/household_eval' / ('batch_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                                          + ('' if a.planner == 'recipe' else '_' + a.planner)
                                                          + ('_limited' if a.wall_minutes or a.sim_minutes or a.max_calls else ''))
        out.mkdir(parents=True, exist_ok=False)
    jobs = queue.Queue()
    names = {}
    for s in sources:
        name = label(s)
        while name in names.values():name += '+'
        names[s] = name;jobs.put(s)
    width = max(len(n) for n in names.values())
    lock = threading.Lock();procs = {};results = {};stopping = threading.Event();last = {};starts = {}
    started = time.monotonic()

    def say(tag, text):
        try:
            with lock:print(f'[{tag:<{width}}] {text}', flush=True)
        except (BrokenPipeError, OSError):
            # The terminal or pipe is gone: stop every run rather than leave
            # them running without a launcher.
            if not stopping.is_set():threading.Thread(target=stop, daemon=False).start()

    endpoint_files = [f.strip() for f in (a.endpoints or '').split(',') if f.strip()]
    for f in endpoint_files:
        if not Path(f).exists():p.error(f'no endpoints file {f}')

    def worker(gpu, slot=0):
        env = dict(os.environ)
        if endpoint_files:env['SERVICE_ENDPOINTS_FILE'] = str(Path(endpoint_files[slot % len(endpoint_files)]).resolve())
        while True:
            if stopping.is_set():return
            try:source = jobs.get_nowait()
            except queue.Empty:return
            # /nobackup is shared and nearly full: a full disk kills runs and
            # the model services. Wait (do not start) under 40 GB free.
            while not stopping.is_set():
                free = __import__('shutil').disk_usage(out).free / 1e9
                if free >= 40:break
                say('eval', f'only {free:.0f} GB free on {out.anchor or "/"}{str(out).split("/")[1]}; waiting before starting {names[source]}')
                time.sleep(300)
            if stopping.is_set():return
            name = names[source];folder = out / name
            command = ['bash', str(ROOT / 'scripts/eval_household.sh'), str(source), '--gpu', gpu, '--output', str(folder)]
            command += ['--planner', a.planner, '--planner-endpoint', a.planner_endpoint]
            for flag, value in (('--wall-minutes', a.wall_minutes), ('--sim-minutes', a.sim_minutes), ('--max-calls', a.max_calls)):
                if value is not None:command += [flag, str(value)]
            if a.raw:command.append('--raw')
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, start_new_session=True)
            procs[name] = proc;t0 = time.monotonic();last[name] = [t0, 'starting']
            say(name, f'started on GPU {gpu}')
            if stopping.is_set():
                try:os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:pass
            for line in proc.stdout:
                line = line.rstrip('\n')
                # Drop the runner's timestamp prefix; keep the message.
                if ' | ' in line[:40]:line = line.split(' | ', 1)[1]
                if line.startswith(('STAGE', 'PUBLIC CALL')):last[name][1] = line.split(' — ')[0][:70]
                say(name, line)
            code = proc.wait();procs.pop(name, None);last.pop(name, None)
            if stopping.is_set():return
            try:report = json.loads((folder / 'episode/report.json').read_text())
            except Exception:report = {}
            if not report and time.monotonic() - t0 < 600 and starts.get(source, 0) < 1:
                # Ended without a report within 10 min: a startup failure, not a
                # policy result. Keep its folder (renamed) and run it once more.
                starts[source] = starts.get(source, 0) + 1
                n = 1
                while (out / f'{name}_failedstart_{n}').exists():n += 1
                folder.rename(out / f'{name}_failedstart_{n}')
                cause = [l for l in (out / f'{name}_failedstart_{n}' / 'live.log').read_text(errors='replace').splitlines()
                         if 'Error' in l or 'not ready' in l][-1:] if (out / f'{name}_failedstart_{n}' / 'live.log').exists() else []
                say(name, f"ended without a report after {round(time.monotonic() - t0)} s "
                          f"({cause[0].split(' | ', 1)[-1][:120] if cause else 'no error line'}); kept as {name}_failedstart_{n}, retrying in 60 s")
                time.sleep(60);jobs.put(source);continue
            # Finished run: hard-link its byte-identical model copies (same
            # paths and contents; about 0.7 GB per run in batch 20261120).
            subprocess.run([sys.executable, str(ROOT / 'scripts/dedupe_run_mjb.py'), str(folder)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            prog = report.get('progress') or {}
            task, rooms = episode_meta(source)
            results[name] = dict(source=str(source), task=task, rooms=rooms, gpu=gpu, exit_code=code, success=report.get('success'),
                                 progress_done=prog.get('done'), progress_total=prog.get('total'), progress_rate=prog.get('rate'),
                                 completed_calls=report.get('completed_calls'), planned_calls=report.get('planned_calls'),
                                 termination=report.get('termination'), simulated_minutes=round((report.get('simulated_seconds') or 0) / 60, 1),
                                 wall_minutes=round((time.monotonic() - t0) / 60, 1), output=str(folder))
            (out / summary_name).write_text(json.dumps(results, indent=2))
            try:
                # Everything about this episode in one line of <batch>/all_results.jsonl.
                sys.path.insert(0, str(ROOT / 'scripts'))
                from collect_results import append_episode
                append_episode(out, folder)
            except Exception as exc:say(name, f'all_results.jsonl not updated: {exc}')
            stats = by_rooms(results)
            if not a.into:(out / 'summary_by_rooms.json').write_text(json.dumps(stats, indent=2))
            say(name, f"finished: success={report.get('success')} progress={prog.get('done')}/{prog.get('total')} "
                      f"termination={report.get('termination')} "
                      f"in {results[name]['wall_minutes']} min")
            say('eval', f"so far {len(results)}/{len(sources)} done | " +
                ' | '.join(f"{k} rooms: {fmt(v)}" for k, v in stats['by_rooms'].items()))

    def stop(*_):
        # No new episode starts after this; every running one is stopped (SIGINT,
        # then SIGKILL after 20 s), including one a worker launched meanwhile.
        if stopping.is_set():return
        stopping.set();say('eval', 'stopping all runs (no new episodes will start)')
        deadline = time.monotonic() + 20
        while procs and time.monotonic() < deadline:
            for proc in list(procs.values()):
                try:os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:pass
            time.sleep(2)
        for proc in list(procs.values()):
            try:os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:pass
        say('eval', f'stopped; partial logs in {out}')
        os._exit(130)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):signal.signal(sig, stop)

    def heartbeat():
        # Long stages (exploration, carries) print nothing for minutes.
        while not stopping.is_set():
            time.sleep(120)
            for name, (t0, where) in list(last.items()):
                say(name, f'... running {round((time.monotonic() - t0) / 60)} min, at {where}')
    threading.Thread(target=heartbeat, daemon=True).start()
    say('eval', f'{len(sources)} episodes on GPUs {",".join(gpus)} -> {out} | planner {a.planner} | limits: wall '
               f'{a.wall_minutes or "none"} min, simulated {a.sim_minutes or "none"} min, calls {a.max_calls or "none"}')
    if endpoint_files:say('eval', 'service sets: ' + ', '.join(f'worker {i} (GPU {g}) -> {Path(endpoint_files[i % len(endpoint_files)]).parent.name}'
                                                           for i, g in enumerate(gpus)))
    threads = [threading.Thread(target=worker, args=(g, i), daemon=True) for i, g in enumerate(gpus)]
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
