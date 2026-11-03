"""Native OpenPI training with the official LeRobot v3 reader (no training by default)."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
import dataclasses
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/lerobot-v3'))
sys.path.insert(0,str(ROOT/'external/openpi/src'))
sys.path.insert(0,str(ROOT))
import lerobot
import lerobot.datasets.lerobot_dataset as v3
# Upstream OpenPI pins the old namespace. Alias only its dataset import to the
# official v3 package; do not reimplement video decoding or episode boundaries.
common=types.ModuleType('lerobot.common');datasets=types.ModuleType('lerobot.common.datasets')
common.datasets=datasets;datasets.lerobot_dataset=v3;lerobot.common=common
sys.modules.update({'lerobot.common':common,'lerobot.common.datasets':datasets,'lerobot.common.datasets.lerobot_dataset':v3})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/onboard_pi05_training'));p.add_argument('--name',default='v1')
    p.add_argument('--steps',type=int,default=3750);p.add_argument('--batch',type=int,default=8);p.add_argument('--fsdp-devices','--devices',dest='devices',type=int,default=1,help='GPUs per model shard group; 1 means a full replica on each visible GPU')
    p.add_argument('--resume',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--prepare-stats',action='store_true')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--benchmark-steps',type=int,default=0)
    p.add_argument('--benchmark-output',type=Path,default=Path('runs/pi05_training_benchmark.json'))
    a=p.parse_args()
    from cross_episode_sim.finetuning.onboard_pi05 import make_config
    from openpi.training import data_loader
    from openpi import transforms
    from openpi.shared import normalize
    cfg=make_config(a.checkpoint,a.assets,a.output,a.name,a.steps,a.batch,a.devices,a.resume)
    cfg=dataclasses.replace(cfg,num_workers=a.workers)
    print(json.dumps(dict(optimizer_updates=cfg.num_train_steps,global_batch=cfg.batch_size,
        sampled_training_windows=cfg.num_train_steps*cfg.batch_size,
        warmup_updates=cfg.lr_schedule.warmup_steps,decay_updates=cfg.lr_schedule.decay_steps)),flush=True)
    if a.train and a.benchmark_steps:p.error('Choose training or a standalone benchmark')
    audit=json.loads((a.dataset/'meta/arm_skills.json').read_text())
    if audit['status']!='complete' or audit['split']!='fine_tuning' or audit['fps']!=15:raise ValueError('Require complete 15Hz fine-tuning split')
    def dataset(data_config,horizon,model_config):
        return v3.LeRobotDataset(data_config.repo_id,root=a.dataset,video_backend='pyav',tolerance_s=1e-3,
            delta_timestamps={'action':[i/15 for i in range(horizon)]})
    data_loader.create_torch_dataset=dataset
    if a.prepare_stats:
        import pyarrow.parquet as pq
        from tqdm import tqdm
        table=pq.read_table(sorted((a.dataset/'data').rglob('*.parquet')))
        state=np.stack(table['observation.state'].to_pylist());action=np.stack(table['action'].to_pylist());ep=np.array(table['episode_index'])
        stats={k:normalize.RunningStats() for k in ('state','actions')}
        for e in tqdm(np.unique(ep),desc='Fine-tuning-only chunk statistics'):
            s=state[ep==e];x=action[ep==e];indices=np.minimum(np.arange(len(s))[:,None]+np.arange(15)[None,:],len(s)-1)
            chunks=x[indices].copy();chunks[:,:,:7]-=s[:,None,:7]
            stats['state'].update(s);stats['actions'].update(chunks)
        normalize.save(a.assets/'set_table_arm',{k:v.get_statistics() for k,v in stats.items()})
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    if dc.norm_stats is None:raise ValueError('Run --prepare-stats first')
    ds=dataset(dc,15,cfg.model)
    transformed=data_loader.TransformedDataset(ds,[*dc.data_transforms.inputs,
        transforms.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
    for i in (0,len(ds)//2,len(ds)-1):
        raw=ds[i];sample=transformed[i]
        assert sample['actions'].shape==(15,32) and np.isfinite(sample['actions']).all()
        assert set(sample['image'])=={'base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'}
        assert sample['image']['base_0_rgb'].shape==(224,224,3)
    print(json.dumps(dict(data_pipeline_validated=True,frames=len(ds),action_chunk=[15,32],active_actions=8,training_started=a.train)),flush=True)
    if a.train or a.benchmark_steps:
        spec=importlib.util.spec_from_file_location('native_openpi_train',ROOT/'external/openpi/scripts/train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if a.benchmark_steps:
            from scripts.benchmark_native_pi05 import benchmark
            try:
                benchmark(cfg,module,a.benchmark_steps,a.benchmark_output)
            except Exception as exc:
                report=json.loads(a.benchmark_output.read_text()) if a.benchmark_output.exists() else {}
                report.update(status="failed",failure=str(exc))
                a.benchmark_output.parent.mkdir(parents=True,exist_ok=True)
                a.benchmark_output.write_text(json.dumps(report,indent=2))
                raise
        else:module.main(cfg)


if __name__=='__main__':main()
