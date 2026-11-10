"""Run independent Qwen component cases and retain every failure and video."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);results=[]
    for case in json.loads(a.cases.read_text()):
        name=case['case'];out=a.output/name
        command=['bash','scripts/test_qwen_household.sh','--case',name,'--recording',case['recording'],
                 '--frame-index',str(case.get('frame_index',0)),'--output',str(out),'--render']
        if name=='pot_knob':command+=['--stove-convention']
        if case.get('floor_context'):command+=['--floor-context']
        if case.get('observed_aperture'):command+=['--observed-aperture',case['observed_aperture']]
        print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),'START',name,'LOG',out.with_suffix('.log'),flush=True)
        with out.with_suffix('.log').open('w',buffering=1) as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {'success':False,'error':'no report'}
        results.append(dict(case=name,returncode=run.returncode,report=report))
        (a.output/'suite_results.json').write_text(json.dumps(results,indent=2))
        print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),'END',name,report['success'],flush=True)


if __name__=='__main__':main()
