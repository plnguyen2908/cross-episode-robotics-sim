"""Qualify right-door opening and drawer/cabinet/microwave open-close cycles.

Saved starts are test resets, not execution-policy inputs. Cabinet/microwave
start grasped and explicitly release/regrasp between opening and closing.
Videos are encoded only after physical execution. No automatic retries.
"""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from benchmark_sensor_door_wall import CASES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',default=['0'])
    p.add_argument('--cases',nargs='+',choices=['right','drawer','cabinet','microwave'],default=['right','drawer','cabinet','microwave'])
    p.add_argument('--skip-video-render',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def run(name,gpu):
        out=a.output/name
        env=dict(os.environ,MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        env['PYTHONPATH']='.:external/DAAAM/src:/nobackup/le/cross-episode-robotics-sim/external/molmospaces'
        if name=='drawer':
            command=[sys.executable,'scripts/test_sensor_drawer_cycle.py','--output',str(out)]
            recordings=[('opening/state_recording','open.mp4'),('cycle_recording','close.mp4')]
        else:
            source,mode=CASES[name]
            cfg=json.loads((Path('runs')/source/'config.json').read_text())
            cfg.update(output=str(out),contact_ik=mode,disable_local_contact_ik=False,
                       absolute_open_angle=90.,open_only=name=='right',release_regrasp=name!='right',shared_oracle_contact=name=='right')
            command=[sys.executable,'scripts/test_sensor_hinged_cycle.py']
            for k,v in cfg.items():
                if v is None or v is False:continue
                command.append('--'+k.replace('_','-'))
                if v is not True:command.append(str(v))
            recordings=[('state_recording','video.mp4')]
        print(f'{name}: GPU {gpu}, starting',flush=True);start=time.perf_counter()
        with (a.output/(name+'.log')).open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-start
        path=out/'report.json';report=json.loads(path.read_text()) if path.exists() else {}
        record=dict(case=name,gpu=gpu,execution_wall_seconds=elapsed,execution_returncode=result.returncode,
                    success=result.returncode==0 and report.get('success',False),error=report.get('error'),
                    simulated_seconds=report.get('simulated_seconds'),rendering_wall_seconds=0.,videos=[])
        if name=='drawer':
            opening_path=out/'opening/report.json'
            opening=json.loads(opening_path.read_text()) if opening_path.exists() else {}
            record['phase_simulated_seconds']={'opening':opening.get('simulated_seconds'),
                                              'release_close_release':report.get('simulated_seconds')}
            values=list(record['phase_simulated_seconds'].values())
            record['simulated_seconds']=sum(values) if all(v is not None for v in values) else None
        if not a.skip_video_render:
            render_start=time.perf_counter()
            for recording,video in recordings:
                if not (out/recording/'states.h5').exists():continue
                with (out/(video+'.log')).open('w') as log:
                    r=subprocess.run([sys.executable,'scripts/render_deferred_rollout.py','--recording',str(out/recording),'--output',str(out/video)],env=env,stdout=log,stderr=subprocess.STDOUT)
                record['videos'].append(dict(path=str(out/video),returncode=r.returncode))
            record['rendering_wall_seconds']=time.perf_counter()-render_start
        record['total_wall_seconds']=time.perf_counter()-start
        (a.output/(name+'_timing.json')).write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True);return record
    def worker(i):return [run(name,a.gpus[i]) for name in a.cases[i::len(a.gpus)]]
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        records=[item for group in pool.map(worker,range(len(a.gpus))) for item in group]
    (a.output/'summary.json').write_text(json.dumps(records,indent=2))
    return 0 if all(r['success'] and all(v['returncode']==0 for v in r['videos']) for r in records) else 1


if __name__=='__main__':raise SystemExit(main())
