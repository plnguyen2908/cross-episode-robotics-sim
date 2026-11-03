"""Isolated upstream RING inference. Requests contain RGB, text and reset only."""
import argparse,json,sys,traceback,copy
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0);p.add_argument('--max-steps',type=int,default=600);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/OneRING'))
try:
    from architecture.models.spoc_models import REGISTERED_MODELS
    torch.manual_seed(a.seed)
    pkg=REGISTERED_MODELS['TuneSpocLlamaModelWTextGoal'];cfg=copy.deepcopy(pkg.config)
    cfg.batch_size=1;cfg.max_seq_len=a.max_steps
    # Identify the upstream checkpoint container, rather than silently loading
    # mismatched weights. The file is the explicitly downloaded upstream model.
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if 'model_state_dict' not in ckpt:
        raise ValueError('Expected the official AllenAct OneRING checkpoint; refusing permissive partial loading')
    del ckpt
    agent=pkg.model_cls.build_agent(cfg=cfg,ckpt_pth=a.checkpoint,device=a.device,load_from_pl=False)
    agent.eval();agent.reset();steps=0
    print('RING_RESULT '+json.dumps(dict(ok=True,ready=True,actions=agent.get_action_list())),flush=True)
    for line in sys.stdin:
        try:
            req=json.loads(line)
            if req.get('close'):break
            if req.get('reset'):
                agent.reset();steps=0;result=dict(ok=True,reset=True)
            else:
                if steps>=a.max_steps:raise RuntimeError('RING context budget exhausted; explicit episode reset required')
                with np.load(req['frame'],allow_pickle=False) as f:
                    obs={'raw_navigation_camera':f['navigation'],'raw_manipulation_camera':f['wrist'],'traj_index':np.array([0],np.int64)}
                with torch.inference_mode():action,probs=agent.get_action(obs,req['goal'])
                steps+=1
                result=dict(ok=True,action=action,probabilities=probs.detach().cpu().flatten().tolist(),step=steps)
            print('RING_RESULT '+json.dumps(result),flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr);print('RING_RESULT '+json.dumps(dict(ok=False,error=str(exc))),flush=True)
except Exception as exc:
    traceback.print_exc(file=sys.stderr);print('RING_RESULT '+json.dumps(dict(ok=False,error=str(exc))),flush=True)
