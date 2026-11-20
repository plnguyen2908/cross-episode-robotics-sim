"""Shared-output bookkeeping so several evaluation workers can split one episode list."""
import json
import os
import socket
import time
from collections import Counter
from pathlib import Path

# An episode the model service could not serve: never scored, rerun on resume.
INVALID='model_service_error'


def write_json(path,value):
    path=Path(path);tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:pass
    return True


def claim(folder,resume=False):
    """Reserve one episode folder. False: it is finished, or another live worker holds it."""
    folder=Path(folder)
    try:folder.mkdir()
    except FileExistsError:
        if not resume:raise
        result=folder/'result.json'
        if result.is_file():
            if json.loads(result.read_text())['termination']!=INVALID:return False
        else:
            try:owner=json.loads((folder/'claim.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):owner=None
            if owner is None:
                # Another worker may be between mkdir and writing its claim.
                if time.time()-folder.stat().st_mtime<60:return False
            elif owner['host']!=socket.gethostname() or alive(owner['pid']):return False
        stale=folder.parent/'_stale'/f'{folder.name}.{time.time_ns()}'
        stale.parent.mkdir(exist_ok=True)
        try:folder.rename(stale);folder.mkdir()
        except (FileNotFoundError,FileExistsError):return False
    write_json(folder/'claim.json',{'pid':os.getpid(),'host':socket.gethostname(),'started':time.time()})
    return True


def room_counts(dataset,splits):
    """Room count per episode from the split manifests (absent in older datasets)."""
    counts={}
    for split in splits:
        root=(Path(dataset)/split).resolve()
        for row in json.loads((root/'manifest.json').read_text())['episodes']:
            if 'room_count' in row:counts[(root/row['folder']).resolve()]=row['room_count']
    return counts


def rate(rows):
    result={'episodes':len(rows),'successes':sum(r['success'] for r in rows),
            'success_rate':sum(r['success'] for r in rows)/len(rows) if rows else None}
    values=[r['task_progress']['rate'] for r in rows
            if r.get('task_progress') and r['task_progress'].get('rate') is not None]
    if values:result.update(progress_episodes=len(values),mean_task_progress=sum(values)/len(values))
    return result


def summarize(output,planned,mock=False):
    """Rebuild summary.json from every result on disk, whichever worker wrote it."""
    output=Path(output)
    results=[json.loads(p.read_text()) for p in sorted(output.glob('[0-9][0-9][0-9][0-9]_*/result.json'))]
    scored=[r for r in results if r['termination']!=INVALID]
    def groups(key):
        return {str(value):rate([r for r in scored if r.get(key)==value])
                for value in sorted({r[key] for r in scored if r.get(key) is not None})}
    usage=Counter()
    for r in results:usage.update(r.get('usage') or {})
    summary={**rate(scored),'planned':planned,'pending':planned-len(scored),'service_errors':len(results)-len(scored),
             'mock':mock,'by_task':groups('task'),'by_room_count':groups('room_count'),'by_split':groups('split'),
             'terminations':dict(Counter(r['termination'] for r in results)),'usage':dict(usage),'results':results}
    write_json(output/'summary.json',summary)
    return summary
