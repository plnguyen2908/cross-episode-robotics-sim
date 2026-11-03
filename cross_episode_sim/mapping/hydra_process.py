"""Persistent native Hydra process for applications using another Python env."""
import json
import os
from pathlib import Path
import subprocess

from .hydra_mapping import HydraFrameWriter


class NativeHydraProcess:
    """Synchronous live sensor mapping; no simulator handle crosses this boundary.

    update(camera, tracking, detections, timestamp) returns native object nodes.
    close() saves Hydra's graph/mesh and terminates the worker. Use as a context.
    """
    def __init__(self,output,root=None):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.writer=HydraFrameWriter(self.output/'frames')
        root=Path(root or Path(__file__).resolve().parents[2]).resolve()
        env=os.environ.copy();prefix=root/'.venv-hydra'
        env['LD_LIBRARY_PATH']=f'{root}/.tools/hydra_ws/install/lib:{prefix}/lib'
        env['PYTHONPATH']=str(root)
        env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='4'
        self.log=open(self.output/'worker.log','w')
        self.process=subprocess.Popen([str(prefix/'bin/python'),'-u',str(root/'scripts/native_hydra_worker.py'),'--output',str(self.output/'map')],cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.closed=False

    def _request(self,message):
        if self.process.poll() is not None:raise RuntimeError(f'Hydra worker exited; see {self.output}/worker.log')
        self.process.stdin.write(json.dumps(message)+'\n');self.process.stdin.flush()
        for line in self.process.stdout:
            if not line.startswith('HYDRA_RESULT '):
                self.log.write(line);self.log.flush();continue
            result=json.loads(line[len('HYDRA_RESULT '):])
            if not result['ok']:raise RuntimeError(result['error'])
            return result
        raise RuntimeError(f'Hydra worker closed its output; see {self.output}/worker.log')

    def update(self,camera,tracking,detections,timestamp):
        if self.closed:raise RuntimeError('Mapper is closed')
        path=self.writer.update(camera,tracking,detections,timestamp)
        if path is None:return None
        return self._request(dict(frame=str(path)))

    def close(self):
        if self.closed:return
        try:
            if self.process.poll() is None:self._request(dict(close=True))
        finally:
            self.closed=True
            self.process.stdin.close()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
            self.process.stdout.close();self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
