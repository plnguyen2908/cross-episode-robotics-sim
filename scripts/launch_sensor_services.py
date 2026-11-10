"""Launch persistent sensor-model services; GPU assignments are explicit."""
import os,subprocess,json,time,argparse,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];os.chdir(root)
p=argparse.ArgumentParser(description='Persistent local gRPC workers; keep this supervisor running')
p.add_argument('--gpus',default='0,1,2,3,4',help='GPU per qwen,molmo,fastsam,m2t2,onering worker')
p.add_argument('--output',type=Path);p.add_argument('--base-port',type=int,default=51100)
a=p.parse_args();gpus=a.gpus.split(',')
if len(gpus)!=5:p.error('Supply five GPU assignments, allowing repeats')
out=a.output.resolve() if a.output else Path(tempfile.mkdtemp(prefix='sensor_services_',dir=root/'runs'))
if a.output:out.mkdir(parents=True,exist_ok=False)
models=[('qwen',0,51100,None),('molmo',1,51101,None),('fastsam',2,51103,str(root/'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt')),('m2t2',4,51104,'/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth'),('onering',3,51105,'/nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt')]
models=[(kind,gpus[i],a.base_port+i,checkpoint) for i,(kind,_,_,checkpoint) in enumerate(models)]
from cross_episode_sim.runtime.service_supervisor import ServiceSupervisor
specs=[]
import signal
def stop(signum,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,stop)
for kind,gpu,port,checkpoint in models:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HOME='/nobackup2/le/.cache',PYTHONPATH=f'{root}/external/DAAAM/src:{root}:/nobackup/le/cross-episode-robotics-sim/external/molmospaces',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  exe='.venv-m2t2/bin/python' if kind=='m2t2' else '.venv-sift/bin/python'
  cmd=[exe,'scripts/serve_sensor_model.py','--kind',kind,'--port',str(port),'--output',str(out/kind)]
  if checkpoint:cmd+=['--checkpoint',checkpoint]
  specs.append(dict(kind=kind,command=cmd,env=env))
endpoints={kind:f'127.0.0.1:{port}' for kind,_,port,_ in models};(out/'endpoints.json').write_text(json.dumps(endpoints));print(json.dumps(endpoints),flush=True)
supervisor=ServiceSupervisor(specs,out)
try:
 while True:
  supervisor.tick();time.sleep(1)
except KeyboardInterrupt:pass
finally:
 supervisor.close()
