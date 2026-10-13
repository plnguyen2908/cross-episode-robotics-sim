"""Lease an idle visible GPU for an entire history, including deferred rendering."""
import csv
import fcntl
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def available_gpus():
    listing = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
        text=True, timeout=20)
    busy = set(processes.split())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed = None if visible is None else [v.strip() for v in visible.split(',') if v.strip()]
    result = []
    for index, uuid, free, utilization in csv.reader(listing.splitlines(), skipinitialspace=True):
        if allowed is not None and not any(v == index or (v.startswith('GPU-') and uuid.startswith(v)) for v in allowed):
            continue
        try:
            free, utilization = int(free), int(utilization)
        except ValueError:
            continue  # Do not schedule devices whose availability cannot be measured.
        if uuid not in busy and free >= 8192 and utilization <= 10:
            result.append(dict(index=index, uuid=uuid, free=free, utilization=utilization))
    return sorted(result, key=lambda row: (row['utilization'], -row['free'], int(row['index'])))


@contextmanager
def lease_gpu(requested='auto', *, waiting=None):
    """Auto assignments cooperate across our workers; external jobs remain unreserved."""
    if requested != 'auto':
        yield str(requested)
        return
    directory = Path(tempfile.gettempdir()) / f'ces-gpu-leases-{os.getuid()}'
    directory.mkdir(exist_ok=True)
    last_notice = -float('inf')
    while True:
        for row in available_gpus():
            lock = (directory / f"{row['uuid']}.lock").open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                # Another application may have started since enumeration.
                if not any(g['uuid'] == row['uuid'] for g in available_gpus()):
                    continue
                yield row['uuid']
                return
            finally:
                lock.close()
        if time.monotonic() - last_notice >= 30:
            if waiting:
                waiting()
            last_notice = time.monotonic()
        time.sleep(5)
