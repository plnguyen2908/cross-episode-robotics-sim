"""Generic-object pickup extension; cups/bowls retain the qualified backend."""
import argparse
import json
from cross_episode_sim.runtime import artifacts
import sys
from pathlib import Path
from cross_episode_sim.runtime.episode import current_runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runtime=current_runtime()
    if runtime is None:raise RuntimeError('Use the household pipeline or recorded-pick harness')
    commands=[
        [sys.executable,'scripts/capture_m2t2_input.py','--source',str(a.source),'--output',str(a.output)],
        [sys.executable,'scripts/household_grasp_proposals.py','--input',str(a.output/'input.npz'),
         '--output',str(a.output/'predictions.json')],
        [sys.executable,'scripts/test_household_contact_pick.py','--source',str(a.source),
         '--input',str(a.output/'input.npz'),'--predictions',str(a.output/'predictions.json'),
         '--output',str(a.output/'physics'),'--household-grasp']]
    label=json.loads(artifacts.read_text(a.source/'report.json'))['selected_object']['label']
    if 'cookware' in label or label.startswith('food container'):
        if 'cookware' in label:
            from cross_episode_sim.navigation.household_grasp_dock import refine_cookware_view
            refine_cookware_view(runtime,a.source,a.output/'fresh_dock')
        # Qualified rim pickup uses the full observed crop; generic bottle
        # proposal ranking is not appropriate for shallow cookware/container rims.
        commands[1][1]='scripts/cookware_grasp_proposals.py'
        commands[2].remove('--household-grasp')
    for command in commands:
        result=runtime.run(command,sys.stdout)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
