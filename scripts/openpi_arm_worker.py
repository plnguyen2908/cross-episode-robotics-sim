"""Isolated upstream OpenPI inference; receives RGB, arm joints and gripper only."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    from openpi.training import config
    from openpi.policies import policy_config
    checkpoint=Path(a.checkpoint)
    if (checkpoint/'assets/set_table_arm/norm_stats.json').exists():
        from cross_episode_sim.finetuning.onboard_pi05 import make_config
        cfg=make_config(checkpoint,checkpoint/'assets',checkpoint)
    else:cfg=config.get_config('pi05_droid_jointpos_polaris')
    policy=policy_config.create_trained_policy(cfg,a.checkpoint)
    print('VLA_RESULT '+json.dumps(dict(ok=True,action_space='absolute_joint_position',action_dim=8)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('close'):break
        try:
            with np.load(request['frame']) as f:
                obs={'observation/exterior_image_1_left':f['navigation'],
                     'observation/wrist_image_left':f['wrist'],
                     'observation/joint_position':f['arm'],
                     'observation/gripper_position':f['gripper'],
                     'prompt':request['prompt']}
            t=time.monotonic();result=policy.infer(obs);actions=np.asarray(result['actions'])
            if actions.ndim!=2 or actions.shape[1]!=8 or not np.isfinite(actions).all():raise ValueError('Invalid native action chunk')
            print('VLA_RESULT '+json.dumps(dict(ok=True,actions=actions.tolist(),inference_seconds=time.monotonic()-t)),flush=True)
        except Exception:
            print('VLA_RESULT '+json.dumps(dict(ok=False,error=traceback.format_exc())),flush=True)


if __name__=='__main__':main()
