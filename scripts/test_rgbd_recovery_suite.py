"""Run fixed sensor-only recovery cases; aggregate diagnostics after completion."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[4,5,6,7])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    streams={'two_room':Path('runs/slam_stream_masked'),'multi_room':Path('runs/slam_stream_multi_90')}
    cases=[]
    for name,stream in streams.items():
        for label,gap in [('control',None),('short_turn',(300,310)),('one_second',(100,120)),('two_seconds',(600,640))]:
            cases.append((name+'_'+label,stream,gap))
    def worker(slot):
        results=[]
        for name,stream,gap in cases[slot::len(a.gpus)]:
            out=a.output/name
            cmd=[sys.executable,'-u','scripts/run_rgbd_slam_stream.py','--stream',str(stream/'sensors'),'--diagnostics',str(stream/'diagnostics'),'--output',str(out),'--no-video','--device','CPU:0']
            if gap:cmd+=['--dropout',*map(str,gap),'--vggt-recovery']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpus[slot]),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.',MUJOCO_GL='egl',MPLCONFIGDIR=str(Path('.venv-vggt-cache/matplotlib').resolve()))
            tqdm.write(f'{name}: GPU {a.gpus[slot]} starting')
            with (a.output/f'{name}.log').open('w') as f:
                rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            record=dict(case=name,returncode=rc,dropout=gap)
            if rc==0:
                report=json.loads((out/'report.json').read_text())
                tracking=json.loads((out/'tracking.json').read_text())
                after=[r for r in tracking if gap and r['index']>=gap[1] and r['accepted']]
                report['first_accepted_after_dropout']=after[0]['index'] if after else None
                report['recovery_delay_seconds']=(after[0]['index']-gap[1])/20 if after else None
                record.update(report)
            results.append(record)
            tqdm.write(f'{name}: exit {rc}, final error {record.get("final_position_error_m")} m')
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,i) for i in range(len(a.gpus))]
        for future in tqdm(as_completed(futures),total=len(futures),desc='Recovery workers'):
            results.extend(future.result())
            (a.output/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
