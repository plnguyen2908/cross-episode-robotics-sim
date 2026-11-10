import os
import signal
import subprocess
import sys
import time
from pathlib import Path

def test_cancel_pipeline_terminates_stage(tmp_path):
    pidfile=tmp_path/'pid'
    worker="import os,time,pathlib;pathlib.Path(%r).write_text(str(os.getpid()));time.sleep(60)" % str(pidfile)
    parent="from cross_episode_sim.evaluation.sensor_set_table import run_stage_command;import sys;run_stage_command([sys.executable,'-c',%r],sys.stdout)" % worker
    proc=subprocess.Popen([sys.executable,'-c',parent],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    child=None
    try:
        deadline=time.monotonic()+10
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert pidfile.exists()
        child=int(pidfile.read_text())
        proc.send_signal(signal.SIGTERM);proc.wait(timeout=12)
        assert not Path('/proc',str(child)).exists()
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        if child is not None:
            try:os.kill(child,signal.SIGKILL)
            except ProcessLookupError:pass
