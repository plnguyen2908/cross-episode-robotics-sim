"""Export one successful skill call per video from completed atomic suites."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cross_episode_sim.skills.catalog import SKILL_NAMES  # noqa: E402


def rendered_times(trace, fps, speed):
    """Match the deferred renderer's frame scheduling, preserving source time."""
    deadline = 0.
    times = []
    for row in trace:
        if row['time'] + 1e-9 < deadline:
            continue
        times.append(row.get('source_time', row['time']))
        deadline += speed / fps
    return times


def export(suites, output):
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    for suite in suites:
        for report_path in sorted(suite.glob('*/report.json')):
            folder = report_path.parent
            report = json.loads(report_path.read_text())
            # Failed/rolled-back runs need separate diagnostic exports, not demo clips.
            if not report.get('success') or report.get('video_status') != 'complete':
                continue
            calls_path = folder / 'skill_calls.json'
            if not calls_path.exists():
                continue
            videos = [p for p in folder.glob('*.mp4')
                      if p.name not in ('head_camera.mp4', 'grasp_closeup.mp4')]
            if len(videos) != 1:
                continue
            trace_path = folder / 'trace_trimmed.json'
            if not trace_path.exists():
                trace_path = folder / 'trace.json'
            trace = json.loads(trace_path.read_text())
            fps = float(report['video_fps'])
            times = rendered_times(trace, fps, float(report['video_speedup']))
            calls = json.loads(calls_path.read_text())
            # Simulation rollback makes time-only attribution ambiguous.
            if any(not c.get('success') for c in calls):
                continue
            for call in calls:
                name = call['skill']
                if name not in SKILL_NAMES or name in entries:
                    continue
                selected = [i for i, t in enumerate(times)
                            if call['start_time'] <= t < call['end_time']]
                if not selected:
                    continue
                first, end = selected[0], selected[-1] + 1
                target = output / f'{name}.mp4'
                subprocess.run([
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(videos[0]), '-vf',
                    f'trim=start_frame={first}:end_frame={end},setpts=PTS-STARTPTS',
                    '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                    '-threads', '2', str(target)], check=True)
                entries[name] = dict(video=str(target.resolve()), source=str(folder.resolve()),
                                     call=call, start_frame=first, end_frame=end,
                                     duration_seconds=(end-first)/fps)
    missing = sorted(set(SKILL_NAMES) - set(entries))
    (output / 'manifest.json').write_text(json.dumps(
        dict(skills=entries, missing=missing), indent=2))
    lines = ['# Individual atomic skill videos', '',
             'Each clip contains one successful public skill call. Playback speed is preserved.',
             '', '| Skill | Video |', '| --- | --- |']
    lines += [f'| {name} | [{name}.mp4]({name}.mp4) |' for name in sorted(entries)]
    lines += ['', 'Missing successful clips: ' + (', '.join(missing) or 'none')]
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suites', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    entries = export(args.suites, args.output)
    print(f'Exported {len(entries)}/{len(SKILL_NAMES)} skills to {args.output}')


if __name__ == '__main__':
    main()
