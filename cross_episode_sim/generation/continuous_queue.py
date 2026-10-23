"""Continuously refill a shared worker pool; no task lanes or batch barriers."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_continuous_queue(next_job, run, completed, workers):
    """Callbacks run on the coordinator except `run`, which runs in worker threads."""
    if workers < 1:
        raise ValueError('workers must be positive')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while True:
            while len(active) < workers:
                job = next_job()
                if job is None:
                    break
                active[pool.submit(run, job)] = job
            if not active:
                return
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                # Infrastructure exceptions stop scheduling; ordinary rollout
                # failures must be returned as explicit result dictionaries.
                completed(job, future.result())
