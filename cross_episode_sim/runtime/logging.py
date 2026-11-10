"""Line timestamps for live terminal and stage logs (including fragmented writes)."""
from datetime import datetime
from zoneinfo import ZoneInfo
import threading


class TimestampedStream:
    def __init__(self, stream):
        self.stream = stream
        self.line_start = True
        self.lock = threading.RLock()

    def write(self, text):
        with self.lock:
            for part in text.splitlines(keepends=True):
                if self.line_start and part.strip():
                    now = datetime.now(ZoneInfo('America/New_York'))
                    self.stream.write(now.isoformat(timespec='seconds') + ' | ')
                self.stream.write(part)
                self.line_start = part.endswith(('\n', '\r'))
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)
