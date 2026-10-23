#!/usr/bin/env python3
"""Run every placement-qualified stovetop candidate through the two-burner task.

Reuse prior full stove reports; check geometry before spending GPU time. Each
new candidate uses its own config/output and the existing automatic GPU lease.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--watch', type=Path, action='append', default=[],
                        help='Already-running episode to monitor instead of duplicating')
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        parser.error('workers and timeout must be positive')
    from cross_episode_sim.paths import GRASP_SWEEP_DIR
    from cross_episode_sim.tasks.breakfast.scene import category
    from cross_episode_sim.generation.asset_suitability import top_down_geometry, two_pot_span_limit
    from cross_episode_sim.generation.composite_pilot import TASKS
    from tqdm import tqdm
    root = Path(tempfile.mkdtemp(prefix='stove_cookware_sweep_', dir=REPO/'runs'))
    print('OUTPUT='+str(root), flush=True)
    template = json.loads((REPO/'runs/composite_pilot_ibzcq_n4/warm_two_pots_06_object_instances/config.json').read_text())
    template['instruction'] = TASKS['warm_two_pots']['instruction']
    candidates = [r for r in json.loads((GRASP_SWEEP_DIR/'current_results.json').read_text())
                  if r.get('placement_passes',0)>0 and category(r) in {'pot','pan','saucepan','kettle'}
                  and 'lid' not in r['asset'].lower()]
    # Index existing evidence before creating any new candidate directories.
    prior = {}
    for manifest in (REPO/'runs').glob('composite_pilot_*/warm_two_pots_*/task_manifest.json'):
        try:
            data = json.loads(manifest.read_text())
            keys = {b['key'] for b in data['bindings'] if b['category']=='pot'}
            if len(keys)!=1: continue
            key = next(iter(keys)); report = manifest.parent/'report.json'
            if report.exists():
                r = json.loads(report.read_text())
                entry = dict(asset_key=key, status='passed' if r.get('success') is True else 'failed',
                             reused=True, output=str(manifest.parent), error=r.get('error'))
                score = (entry['status']=='passed', report.stat().st_mtime)
                if key not in prior or score > prior[key][0]: prior[key]=(score,entry)
        except (OSError,ValueError,KeyError): continue
    watching = {}
    for episode in args.watch:
        data=json.loads((episode/'task_manifest.json').read_text())
        for b in data['bindings']:
            if b['category']=='pot': watching[b['key']]=episode.resolve()
    results = {}; pending=[]
    for row in candidates:
        key=row['key']
        if key in watching and not (watching[key]/'report.json').exists():
            pending.append(row);continue
        if key in prior:
            results[key]=prior[key][1];continue
        if category(row)=='kettle' and row['source']!='robocasa':
            results[key]=dict(status='excluded',reason='stovetop/non-electric suitability unverified');continue
        try:
            count, size=top_down_geometry(row['source'],row['asset'],key)
            if count<=0 or max(size[:2])>two_pot_span_limit():
                results[key]=dict(status='excluded',reason='no top-down evidence or paired-burner footprint too large',size_m=size);continue
        except (OSError,ValueError) as exc:
            results[key]=dict(status='blocked',reason=str(exc));continue
        pending.append(row)
    def save():
        (root/'results.json').write_text(json.dumps(results,indent=2))
    save()
    def run(row):
        key=row['key']
        if key in watching:
            episode=watching[key]; deadline=time.monotonic()+args.timeout
            while not (episode/'report.json').exists() and time.monotonic()<deadline:
                time.sleep(5)
            if not (episode/'report.json').exists():
                return key,dict(status='blocked',reason='watched run timed out',output=str(episode))
            data=json.loads((episode/'report.json').read_text())
            return key,dict(status='passed' if data.get('success') is True else 'failed',
                            reused=True,output=str(episode),error=data.get('error'))
        folder=root/key;folder.mkdir()
        config=deepcopy(template)
        config['scene']['asset_keys']={f'pot_{i:02d}':key for i in (1,2)}
        config['scene_spec_sha256']=hashlib.sha256(json.dumps(config['scene'],sort_keys=True).encode()).hexdigest()
        cfg=folder/'config.json';cfg.write_text(json.dumps(config,indent=2))
        output=folder/'rollout'
        with (folder/'worker.log').open('w') as log:
            try:
                proc=subprocess.run([sys.executable,'-u','-m','cross_episode_sim.generation.composite_pilot',
                    '--config',str(cfg),'--output',str(output),'--timeout',str(args.timeout)],
                    cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=False)
            except OSError as exc:return key,dict(status='blocked',reason=str(exc))
        episode=output/config['episode_id'];report=episode/'report.json'
        data=json.loads(report.read_text()) if report.exists() else {}
        return key,dict(status='passed' if proc.returncode==0 and data.get('success') is True else 'failed',
                        output=str(episode),error=data.get('error'),log=str(folder/'worker.log'))
    print('Reused/excluded:',len(results),'New physical tests:',len(pending),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,row):row['key'] for row in pending}
        for future in tqdm(as_completed(futures),total=len(futures),desc='Stove candidates'):
            key,result=future.result();results[key]=result;save();tqdm.write(key+': '+result['status'])
    print('DONE '+str(root/'results.json'),flush=True)


if __name__=='__main__':main()
