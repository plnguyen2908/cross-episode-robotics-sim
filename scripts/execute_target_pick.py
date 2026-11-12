"""Target-focused M2T2 pickup using the existing capture and motion executor."""
import argparse,sys,json
from pathlib import Path
from cross_episode_sim.runtime.episode import current_runtime
from cross_episode_sim.runtime import artifacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();runtime=current_runtime()
    if runtime is None:raise RuntimeError('Run through the persistent episode runtime')
    commands=[
        [sys.executable,'scripts/capture_m2t2_input.py','--source',str(a.source),'--output',str(a.output)],
        [sys.executable,'scripts/target_pick_proposals.py','--input',str(a.output/'input.npz'),'--output',str(a.output/'predictions.json')],
        [sys.executable,'scripts/test_m2t2_pick.py','--source',str(a.source),'--input',str(a.output/'input.npz'),'--predictions',str(a.output/'predictions.json'),'--output',str(a.output/'physics')]]
    for command in commands:
        result=runtime.run(command,sys.stdout)
        if result.returncode:return result.returncode
    report=json.loads(artifacts.read_text(a.output/'physics/report.json'))
    return 0 if report.get('execution_success') else 1

if __name__=='__main__':raise SystemExit(main())
