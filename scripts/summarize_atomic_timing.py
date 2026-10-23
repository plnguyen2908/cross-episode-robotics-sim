"""Report every public atomic call's simulated duration, including slow calls."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root):
    rows = []
    for path in sorted(root.glob('*/skill_calls.json')):
        folder = path.parent
        trace_path = folder / 'trace.json'
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
        for call in json.loads(path.read_text()):
            start, end = call['start_time'], call.get('end_time')
            if end is None:
                continue
            duration = end-start
            stages = Counter()
            for a,b in zip(trace,trace[1:]):
                dt = min(end,b['time'])-max(start,a['time'])
                if 0 < dt <= .2:
                    stages[b['stage']] += dt
            skill = call['skill']
            rows.append(dict(demo=folder.name, skill=skill, arguments=call.get('arguments',{}),
                success=call.get('success') is True, seconds=round(duration,3),
                timing=('explicit_wait' if skill=='wait' else 'faster_than_target' if duration<5
                        else 'within_target' if duration<=15 else 'over_target'),
                stage_seconds={k:round(v,3) for k,v in stages.items()},
                largest_stages=[dict(stage=k,seconds=round(v,3)) for k,v in stages.most_common(5)]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',type=Path)
    args=parser.parse_args()
    rows=summarize(args.suite)
    (args.suite/'atomic_timing.json').write_text(json.dumps(rows,indent=2))
    lines=['# Atomic execution timing','',
           'Measured simulated time per call. Calls faster than five seconds are not padded.',
           'Over-target calls require investigation or a documented exception; flags alone are not validation.',
           '', '| Demo | Skill / arguments | Passed | Seconds | Timing |',
           '| --- | --- | --- | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['demo']} | {r['skill']} {json.dumps(r['arguments'])} | {r['success']} | {r['seconds']:.2f} | {r['timing']} |")
    (args.suite/'atomic_timing.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
