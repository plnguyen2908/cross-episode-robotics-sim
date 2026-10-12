"""Persist setup-stage timings before a controller and its progress writer exist."""
import json
from pathlib import Path
from time import perf_counter


class SetupTiming:
    def __init__(self, output):
        self.path = Path(output) / 'setup_timings.json'
        self.stages = []

    def __call__(self, name, function, *args, **kwargs):
        row = dict(stage=name, status='running')
        self.stages.append(row)
        self._write()
        print(f'Setup: {name}', flush=True)
        start = perf_counter()
        try:
            result = function(*args, **kwargs)
        except BaseException:
            row['status'] = 'failed'
            raise
        else:
            row['status'] = 'completed'
            return result
        finally:
            row['wall_seconds'] = perf_counter() - start
            self._write()
            print(f"Setup: {name} {row['status']} ({row['wall_seconds']:.1f}s)", flush=True)

    def _write(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.stages, indent=2))
        temporary.replace(self.path)
