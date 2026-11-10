"""Own worker process groups and restart failures without stopping siblings."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path


class ServiceSupervisor:
    def __init__(self, specs, output, restart_delay=2.):
        self.output=Path(output)
        self.workers={s['kind']:dict(spec=s,process=None,log=None,restarts=0,restart_at=0.) for s in specs}
        self.restart_delay=restart_delay

    def event(self, **value):
        print(json.dumps(dict(time=time.time(),**value)),flush=True)

    @staticmethod
    def terminate_group(process):
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        # A crashed parent can leave its model subprocess alive.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

    def tick(self):
        for kind,w in self.workers.items():
            p=w['process']
            if p is not None and p.poll() is not None:
                self.event(event='worker_exit',kind=kind,pid=p.pid,exit_code=p.returncode)
                self.terminate_group(p)
                w['log'].close();w['process']=None;w['log']=None
                w['restarts']+=1
                w['restart_at']=time.monotonic()+min(30.,self.restart_delay*2**min(w['restarts']-1,4))
            if w['process'] is None and time.monotonic()>=w['restart_at']:
                s=w['spec'];w['log']=(self.output/f'{kind}.log').open('a')
                w['process']=subprocess.Popen(s['command'],env=s['env'],stdout=w['log'],stderr=subprocess.STDOUT,start_new_session=True)
                self.event(event='worker_start',kind=kind,pid=w['process'].pid,restarts=w['restarts'])
        status={k:dict(pid=w['process'].pid if w['process'] else None,restarts=w['restarts']) for k,w in self.workers.items()}
        tmp=self.output/'workers.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(self.output/'workers.json')

    def close(self):
        for w in self.workers.values():
            if w['process'] is not None:self.terminate_group(w['process'])
            if w['log'] is not None:w['log'].close()
