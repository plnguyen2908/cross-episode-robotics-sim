"""Development episode first; evaluate frozen code only after full-task success."""
import argparse,hashlib,json
from pathlib import Path
from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline


def fingerprint():
    h=hashlib.sha256()
    for root in ('cross_episode_sim','scripts'):
        for p in sorted(Path(root).rglob('*.py')):
            h.update(str(p).encode());h.update(p.read_bytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cohort=json.loads(a.cohort.read_text());a.output.mkdir(parents=True,exist_ok=False)
    started_fingerprint=fingerprint()
    development=SensorSetTablePipeline(cohort['development_source'],a.output/'development').run()
    summary=dict(development=development,test_results=[],test_started=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    if not development['success']:
        print('Development failed. Test set remains untouched.',flush=True);return 1
    frozen=fingerprint();summary['development_start_sha256']=started_fingerprint
    if frozen!=started_fingerprint:
        summary['qualification_error']='Code changed during development; rerun the candidate unchanged before test evaluation'
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(summary['qualification_error'],flush=True);return 1
    summary['code_sha256']=frozen;summary['test_started']=True
    for index,episode in enumerate(cohort['episodes']):
        if fingerprint()!=frozen:raise RuntimeError('Code changed after qualification; refusing mixed-version evaluation')
        result=SensorSetTablePipeline(episode['source'],a.output/f'test_{index:02d}').run()
        if fingerprint()!=frozen:raise RuntimeError('Code changed during evaluation; result is not a frozen-version test')
        result['development_exposure']=episode.get('development_exposure',False)
        summary['test_results'].append(result)
        summary['successes']=sum(r['success'] for r in summary['test_results'])
        summary['evaluated']=len(summary['test_results'])
        summary['eligible']=len(cohort['episodes'])
        summary['success_rate']=summary['successes']/summary['evaluated']
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
