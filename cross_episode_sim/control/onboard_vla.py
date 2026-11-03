"""Arm-only OpenPI process boundary. No world/map/target state enters inference."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def droid_observation(packet,gripper_closed_fraction):
    """DROID uses closure in [0,1], seven joint radians, two separate RGB images."""
    q=np.asarray(packet.arm.position,dtype=np.float32)
    if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Expected seven finite joint readings')
    g=float(gripper_closed_fraction)
    if not np.isfinite(g) or not 0<=g<=1:raise ValueError('Invalid measured closure')
    images={name:np.asarray(packet.cameras[name].rgb) for name in ('navigation','wrist')}
    for im in images.values():
        if im.ndim!=3 or im.shape[-1]!=3 or im.dtype!=np.uint8:raise ValueError('Expected uint8 RGB')
    return dict(**images,arm=q,gripper=np.array([g],dtype=np.float32))


class OpenPIArmPolicy:
    """Native DROID/PolaRiS joint-position checkpoint, eight outputs, no base head."""
    def __init__(self,checkpoint,output,gpu='0'):
        root=Path(__file__).resolve().parents[2];self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=False);self.count=0
        self.log=open(self.output/'worker.log','w')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_MEM_FRACTION='.80',
            OPENPI_DATA_HOME='/nobackup2/le/.cache/openpi',PYTHONPATH=str(root)+os.pathsep+str(root/'external/openpi/src'))
        self.process=subprocess.Popen([str(root/'.venv-openpi/bin/python'),'-u',str(root/'scripts/openpi_arm_worker.py'),
            '--checkpoint',str(Path(checkpoint).resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env)
        try:self.metadata=self._read()
        except Exception:self.close();raise
    def _read(self):
        for line in self.process.stdout:
            if line.startswith('VLA_RESULT '):
                result=json.loads(line[len('VLA_RESULT '):])
                if not result['ok']:raise RuntimeError(result['error'])
                return result
            self.log.write(line);self.log.flush()
        raise RuntimeError(f'OpenPI exited; see {self.output}/worker.log')
    def predict(self,packet,prompt,gripper_closed_fraction):
        frame=self.output/f'input_{self.count:04d}.npz'
        np.savez_compressed(frame,**droid_observation(packet,gripper_closed_fraction))
        self.process.stdin.write(json.dumps(dict(frame=str(frame),prompt=prompt))+'\n');self.process.stdin.flush()
        result=self._read();self.count+=1
        with open(self.output/'predictions.jsonl','a') as f:f.write(json.dumps(dict(timestamp=packet.timestamp,prompt=prompt,**result))+'\n')
        return np.asarray(result['actions']),result['inference_seconds']
    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write('{"close":true}\n');self.process.stdin.flush();self.process.wait(timeout=10)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.process.terminate();self.process.wait(timeout=10)
        self.log.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
