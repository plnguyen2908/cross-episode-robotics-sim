"""Run fresh sensor-only docking at four distances and two initial headings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tqdm import tqdm


def run_case(case, source, output, gpu_pair, checkpoint, seconds):
    destination=output/case['name']
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpu_pair),MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    root=Path.cwd()
    env['PYTHONPATH']=f'{root}/external/DAAAM/src:{root}:'+env.get('PYTHONPATH','')
    cmd=[sys.executable,'scripts/demo_semantic_navigation.py','--source',str(source),'--output',str(destination),
         '--navigation-mode','hybrid','--ring-checkpoint',str(checkpoint),'--ring-device','cuda:0','--vlm-device','cuda:1',
         '--goal-room','','--target','bowl','--explorer','frontier','--initial-backoff',str(case['backoff']),
         '--initial-yaw-offset',str(case['yaw']),'--seconds',str(seconds)]
    start=time.monotonic()
    with (output/(case['name']+'.log')).open('w') as log:
        code=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(**case,gpus=gpu_pair,exit_code=code,wall_seconds=time.monotonic()-start)
    report=destination/'report.json'
    if report.exists():
        r=json.loads(report.read_text())
        result.update(status=r['status'],docking_success=r.get('docking_success',False),
                      simulated_seconds=r['simulated_seconds'],contacts=len(r.get('external_contact_events',[])),
                      final_distance_m=r.get('readiness',{}).get('distance_m'),
                      selected_object=r.get('selected_object'),
                      events=r.get('docking_events',[]),video=str(destination/'navigation.mp4'))
    else:result.update(status='process_failed_without_report',docking_success=False)
    (output/(case['name']+'_result.json')).write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ring-checkpoint',type=Path,required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--seconds',type=float,default=120.)
    a=p.parse_args();gpus=a.gpus.split(',')
    if len(gpus)%2 or not gpus:p.error('Provide GPU pairs: one for One-Ring/detection and one for Qwen per worker')
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[dict(name=f'{level}_{heading}',level=level,backoff=backoff,yaw=yaw)
           for heading,yaw in [('forward',0.),('turned',.3)]
           for level,backoff in [('close',0.),('far',1.),('farther',2.5),('farthest',4.5)]]
    # Private reset validation only; never supplied to execution policies.
    import mujoco,numpy as np
    from scripts.demo_local_scan import setup,ContactAudit
    for case in cases:
        m,d,_=setup(a.source,case['backoff']);d.joint('robot_0/base_theta').qpos[0]+=case['yaw'];mujoco.mj_forward(m,d)
        audit=ContactAudit(m);audit.observe(d)
        body=m.body('robot_0/base').id;point=d.xpos[body].copy();point[2]=.04;gid=np.array([-1],dtype=np.int32)
        distance=mujoco.mj_ray(m,d,point,np.array([0.,0.,-1.]),None,1,body,gid)
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(gid[0])) if gid[0]>=0 else ''
        case['reset_validation']=dict(contacts=list(audit.events.values()),floor_distance=distance,floor_geom=name)
        if audit.events or not(name and 'floor' in name and 0<distance<.1):
            raise RuntimeError(f"Invalid reset: {case['name']}: {case['reset_validation']}")
    (a.output/'manifest.json').write_text(json.dumps(dict(source=str(a.source),cases=cases,seconds=a.seconds),indent=2))
    results=[]
    # Each worker owns a GPU pair and pulls the next case when free.
    import queue
    pending=queue.Queue()
    for case in cases:pending.put(case)
    def worker(pair):
        while True:
            try:case=pending.get_nowait()
            except queue.Empty:return
            yield run_case(case,a.source,a.output,pair,a.ring_checkpoint,a.seconds)
    # Consume each generator inside its worker, emitting completed results safely.
    completed=queue.Queue()
    def execute(pair):
        for result in worker(pair):completed.put(result)
    with ThreadPoolExecutor(max_workers=len(gpus)//2) as pool, tqdm(total=len(cases),desc='Distance suite') as bar:
        futures=[pool.submit(execute,gpus[i:i+2]) for i in range(0,len(gpus),2)]
        while len(results)<len(cases):
            try:r=completed.get(timeout=1.)
            except queue.Empty:
                for f in futures:
                    if f.done() and f.exception():raise f.exception()
                if all(f.done() for f in futures):break
                continue
            results.append(r);bar.update();tqdm.write(f"{r['name']}: {r['status']}")
            (a.output/'results.json').write_text(json.dumps(results,indent=2))
    if len(results)!=len(cases):raise RuntimeError('Suite did not complete every configured case')

if __name__=='__main__':main()
