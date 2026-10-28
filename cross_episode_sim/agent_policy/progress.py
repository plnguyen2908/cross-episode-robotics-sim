"""Human-readable progress, including heartbeats during blocking model calls."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import threading
import time


class Progress:
    def __init__(self, label, path=None, interval=15):
        self.label = label
        self.path = Path(path) if path else None
        self.interval = interval
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def log(self, message):
        from tqdm import tqdm
        elapsed = time.monotonic() - self.started
        line = f'{datetime.now(ZoneInfo('America/New_York')).isoformat(timespec="seconds")} [{self.label}] +{elapsed:.1f}s | {message}'
        with self.lock:
            tqdm.write(line)
            sys.stdout.flush()
            if self.path:
                with self.path.open('a') as stream:
                    stream.write(line + '\n')

    @contextmanager
    def stage(self, name):
        start = time.monotonic()
        self.log(name + ' — started')
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(self.interval):
                self.log(f'{name} — still running ({time.monotonic()-start:.1f}s)')
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            yield
        except BaseException:
            self.log(f'{name} — failed after {time.monotonic()-start:.1f}s')
            raise
        else:
            self.log(f'{name} — done in {time.monotonic()-start:.1f}s')
        finally:
            stop.set()
            worker.join()
