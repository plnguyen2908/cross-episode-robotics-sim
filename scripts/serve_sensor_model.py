"""One long-lived local gRPC model worker, loaded before accepting requests."""
import argparse,os
import numpy as np
from PIL import Image
from cross_episode_sim.perception.rpc import start_server


def build(kind,args):
    # Prevent a model in this worker from calling its own configured endpoint.
    os.environ.pop('SENSOR_MODEL_ENDPOINTS',None)
    if kind=='molmo':
        from cross_episode_sim.perception.molmo_pointing import MolmoPointer
        model=MolmoPointer(device=args.device)
        return lambda r:model.point(Image.fromarray(r['image']),r['target'])
    if kind=='qwen':
        from cross_episode_sim.perception.daaam_grounding import LocalVLM
        model=LocalVLM(device=args.device)
        return lambda r:model.label_with_prompt([Image.fromarray(x) for x in r['images']],r['prompt'],r.get('max_new_tokens',100))
    if kind=='fastsam':
        from daaam.config import SegmentationConfig
        from daaam.segmentation.services import SegmentationService
        model=SegmentationService(SegmentationConfig(model_name=os.path.abspath(args.checkpoint),model_config_path=None,device=args.device,min_mask_region_area=100))
        return lambda r:model.segment(r['image'])
    if kind=='m2t2':
        from cross_episode_sim.control.m2t2_policy import M2T2Policy
        model=M2T2Policy(args.checkpoint,'external/M2T2',device=args.device)
        # Standalone construction consumes RNG while initializing the network.
        # Preserve the post-load state for the first sample of each client.
        initial_cpu=model.torch.get_rng_state().clone()
        initial_cuda=model.torch.cuda.get_rng_state_all()
        def infer(r):
            if r.get('initial_sample',False):
                if r.get('seed',0)!=0:raise ValueError('Worker initial sampling seed is 0')
                model.torch.set_rng_state(initial_cpu)
                model.torch.cuda.set_rng_state_all(initial_cuda)
            else:model.torch.manual_seed(r.get('seed',0))
            return model.predict(r['points'],r['rgb'],held_points=r.get('held_points'),ee_pose=r.get('ee_pose'))
        return infer
    if kind=='onering':
        from cross_episode_sim.navigation.onering import OneRingProcess
        from types import SimpleNamespace as NS
        from pathlib import Path
        from uuid import uuid4
        # Each process owns a fresh trace directory; restarting a service must
        # not fail because its previous process left recordings behind.
        session=Path(args.output)/f'session_{uuid4().hex}'
        model=OneRingProcess(args.checkpoint,session,device=args.device)
        def infer(r):
            if r.get('operation')=='reset':return model.reset()
            packet=NS(timestamp=r['timestamp'],cameras={k:NS(rgb=r[k]) for k in ('navigation','wrist')})
            return model.predict(packet,r['goal'])
        return infer
    raise ValueError(kind)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=['molmo','qwen','fastsam','m2t2','onering'])
    p.add_argument('--port',required=True,type=int);p.add_argument('--device',default='cuda:0')
    p.add_argument('--checkpoint');p.add_argument('--output',default='runs/onering_service')
    a=p.parse_args();handler=build(a.kind,a)
    server,port=start_server(handler,f'127.0.0.1:{a.port}')
    print(f'READY {a.kind} 127.0.0.1:{port}',flush=True)
    try:server.wait_for_termination()
    except KeyboardInterrupt:server.stop(5).wait()
