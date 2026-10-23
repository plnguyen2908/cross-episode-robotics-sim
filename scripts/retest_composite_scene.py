"""Execute a saved composite configuration in its existing authored scene.

Physics restarts from setup. The current robot/controller code is used; source
recordings are preserved (navigation may cache maps beside the authored scene).
This isolates controller/mount regressions from scene
randomization. A fresh output directory is always created.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--gpu', default='auto')
    a = p.parse_args()
    source = a.source.resolve()
    config = json.loads((source / 'config.json').read_text())
    manifest = json.loads((source / 'task_manifest.json').read_text())
    os.environ.setdefault('ROBOCASA_DIR', str(ROOT / 'external/robocasa/robocasa'))
    from cross_episode_sim.generation.gpu_pool import lease_gpu
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    with lease_gpu(a.gpu) as gpu:
        os.environ.update(worker_environment(dict(os.environ, MUJOCO_GL='egl'),
            gpu, resolve_gpus([gpu])[gpu]))
        from cross_episode_sim.env import household
        from cross_episode_sim.generation.composite_pilot import execute
        out = Path(tempfile.mkdtemp(prefix='regression_'+config['task']+'_', dir=ROOT/'runs'))
        (out/'config.json').write_text(json.dumps(config, indent=2))
        (out/'retest_source.json').write_text(json.dumps(dict(source=str(source),
            mode='same authored scene; fresh physics; current robot and controller'), indent=2))
        print('OUTPUT='+str(out), flush=True)
        original = household.build_scene
        try:
            household.build_scene = lambda *_: manifest
            execute(config, out)
        finally:
            household.build_scene = original


if __name__ == '__main__':
    main()
