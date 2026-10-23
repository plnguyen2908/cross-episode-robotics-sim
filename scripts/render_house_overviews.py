"""Backfill missing 3D house images for successful episodes, without robot execution."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',type=Path,nargs='+',help='Episode or batch folders')
    p.add_argument('--gpu',default='auto')
    a=p.parse_args();episodes=set()
    for root in a.folders:
        reports=[root/'report.json'] if (root/'report.json').exists() else root.glob('*/report.json')
        for report in reports:
            if json.loads(report.read_text()).get('success') is True and not (report.parent/'house_overview.png').exists():
                episodes.add(report.parent.resolve())
    if not episodes:
        print('No successful episodes missing house_overview.png');return
    from cross_episode_sim.generation.gpu_pool import lease_gpu
    from cross_episode_sim.rendering_gpu import worker_environment,resolve_gpus
    errors=[]
    with lease_gpu(a.gpu) as gpu:
        os.environ.update(worker_environment(dict(os.environ,MUJOCO_GL='egl'),gpu,resolve_gpus([gpu])[gpu]))
        from tqdm import tqdm
        from cross_episode_sim.generation.house_overview import render_house_overview
        for folder in tqdm(sorted(episodes),desc='House overviews',unit='image'):
            try:render_house_overview(folder)
            except Exception as exc:
                errors.append((str(folder),str(exc)));tqdm.write(f'FAILED {folder.name}: {exc}')
    print(f'Rendered {len(episodes)-len(errors)}/{len(episodes)} house images')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
