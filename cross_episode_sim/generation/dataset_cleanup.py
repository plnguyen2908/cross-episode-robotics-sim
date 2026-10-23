"""Remove failed dataset payloads only after the batch parent records worker exit."""
import json
import shutil
from pathlib import Path


def prune_failed_episodes(batch):
    batch = Path(batch)
    results = batch / 'results.json'
    if not results.exists():
        return 0
    rows = json.loads(results.read_text())
    removed = 0
    for episode_id, result in rows.items():
        if result.get('success') is not False:
            continue
        # Only direct children named by this batch's results may be deleted.
        if Path(episode_id).name != episode_id or episode_id in ('.', '..'):
            raise ValueError('Invalid episode identifier in batch results')
        folder = batch / episode_id
        if folder.is_symlink():
            raise ValueError('Refusing to prune symlink episode')
        if not folder.is_dir():
            continue
        report = folder / 'report.json'
        if report.exists() and json.loads(report.read_text()).get('success') is True:
            continue  # Preserve physical successes, even if rendering failed.
        record = batch / 'failures' / (episode_id + '.json')
        record.parent.mkdir(exist_ok=True)
        evidence = {'episode_id': episode_id, 'result': result}
        for name in ('config.json', 'report.json'):
            if (folder/name).exists():
                evidence[name] = json.loads((folder/name).read_text())
        log = folder / 'run.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 8000))
                evidence['log_tail'] = stream.read().decode(errors='replace')
        record.write_text(json.dumps(evidence, indent=2) + '\n')
        shutil.rmtree(folder)
        removed += 1
    return removed
