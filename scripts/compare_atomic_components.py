"""Compare live component-only and smoothed atomic suites, including failures."""
import argparse
import json
import hashlib
from pathlib import Path


def results(root):
    rows = {}
    for path in sorted(root.glob('*/report.json')):
        report = json.loads(path.read_text())
        calls_path = path.parent / 'skill_calls.json'
        calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
        complete = [c for c in calls if c.get('status') == 'completed']
        stages = {}
        trace_path = path.parent / 'trace.json'
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            for a, b in zip(trace, trace[1:]):
                dt = b['time'] - a['time']
                if dt >= 0:
                    label = a.get('stage', 'unknown')
                    stages[label] = stages.get(label, 0.) + dt
        scene = path.parent / 'robocasa_scene.xml'
        scene_hash = hashlib.sha256(scene.read_bytes()).hexdigest() if scene.exists() else None
        rows[path.parent.name] = dict(motion_configuration=report.get('motion_configuration'), scene_sha256=scene_hash, success=report.get('success') is True,
            error=report.get('error'), video_status=report.get('video_status'),
            skill_seconds=sum(c['end_time']-c['start_time'] for c in complete),
            calls=calls, stage_seconds=stages, source=str(path.parent.resolve()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('components', type=Path)
    parser.add_argument('smooth', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = results(args.components), results(args.smooth)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(dict(components=before, smooth=after), indent=2))
    lines = ['# Atomic component and smoothing comparison', '',
        'Live simulation durations summed over completed public calls. Both runs use component optimizations; only the second enables trajectory retiming/interpolation. Failed or incomplete runs are not speedup evidence.', '',
        '| Demo family | Component-only | Smoothed | Change |', '| --- | ---: | ---: | ---: |']
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name, {}), after.get(name, {})
        def cell(row):
            return f"{row['skill_seconds']:.2f} s" if row.get('success') else 'Failed/pending'
        comparable = (a.get('success') and b.get('success') and a['skill_seconds']
                      and a.get('scene_sha256') and a.get('scene_sha256') == b.get('scene_sha256'))
        if comparable:
            percent = 100*(1-b['skill_seconds']/a['skill_seconds'])
            change = f"{abs(percent):.1f}% {'shorter' if percent >= 0 else 'longer'}"
        else:
            change = '—'
        lines.append(f'| {name} | {cell(a)} | {cell(b)} | {change} |')
    lines += ['', 'Detailed call durations, stage timings, failures, and source paths are in [comparison.json](comparison.json).',
              'Stage totals are attributed from consecutive simulation trace samples; public-call durations use exact call boundaries.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
