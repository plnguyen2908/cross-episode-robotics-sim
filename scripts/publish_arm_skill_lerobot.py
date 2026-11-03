"""Publish a validated, additive arm-skill dataset version; never touch old exports."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--repo',default='plnguyen2908/cross-sim-data')
    p.add_argument('--prefix',default='lerobot_atomic/set_table_arm_v1');a=p.parse_args()
    sources=[]
    for split in ('fine_tuning','validation'):
        m=json.loads((a.source/split/'meta/arm_skills.json').read_text())
        assert m['status']=='complete' and m['all_segment_endpoints_decoded']
        assert m['split']==split and m['action_dim']==8 and m['fps']==15
        sources.append({x['source'] for x in m['episodes']})
    if sources[0]&sources[1]:raise ValueError('Source episode leakage')
    api=HfApi()
    try:exists=bool(list(api.list_repo_tree(a.repo,path_in_repo=a.prefix,repo_type='dataset',recursive=False)))
    except EntryNotFoundError:exists=False
    if exists:raise FileExistsError('Dataset prefix already exists; use a new version')
    result=api.upload_folder(repo_id=a.repo,repo_type='dataset',folder_path=a.source,path_in_repo=a.prefix,
        commit_message='Add stationary-arm set-table skill fine-tuning and validation (LeRobot v3)')
    print(result.commit_url)


if __name__=='__main__':main()
