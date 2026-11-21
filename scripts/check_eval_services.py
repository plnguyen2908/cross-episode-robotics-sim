"""Start each gRPC model service from THIS checkout, send one request, stop it.

    source eval_env.sh
    .venv-sift/bin/python scripts/check_eval_services.py --base-port 52000 --gpu 0 \
        [--kinds fastsam m2t2 graspgenx qwen molmo] [--vlm-device cpu]

Uses the same command lines as the evaluation (scripts/serve_sensor_model.py with
.venv-sift, .venv-m2t2 and .venv-graspgenx) on ports base-port, base-port+1, ...;
never pass the ports a running evaluation uses (51500-51510, 51586).
--vlm-device cpu loads Qwen3-VL-8B / Molmo2-4B on the CPU (slow, ~1-3 min each, no GPU memory).
Exits non-zero if any service fails to start or answer.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def request(kind):
    rng = np.random.default_rng(0)
    if kind == 'fastsam':
        img = np.zeros((240, 320, 3), np.uint8); img[60:180, 100:220] = (200, 30, 30)
        return dict(image=img)
    if kind == 'm2t2':
        table = np.c_[rng.uniform(.3, .9, 6000), rng.uniform(-.3, .3, 6000), np.zeros(6000)]
        box = np.c_[rng.uniform(.55, .62, 3000), rng.uniform(-.03, .03, 3000), rng.uniform(0, .12, 3000)]
        pts = np.r_[table, box].astype(np.float32)
        return dict(points=pts, rgb=np.full_like(pts, .5), initial_sample=True, seed=0)
    if kind == 'graspgenx':
        a = rng.uniform(0, 2*np.pi, 3000); z = rng.uniform(0, .1, 3000)
        return dict(object_points=np.c_[.04*np.cos(a), .04*np.sin(a), z].astype(np.float32) + [.5, 0, .8], seed=0)
    img = np.full((224, 224, 3), 235, np.uint8); img[80:150, 70:150] = (30, 60, 200)
    if kind == 'qwen':
        return dict(images=[img], prompt='What color is the square? Answer with one word.', max_new_tokens=8)
    if kind == 'molmo':
        return dict(image=img, target='the blue square')
    raise ValueError(kind)


def summary(result):
    if isinstance(result, dict):
        return {k: (getattr(v, 'shape', None) or (v if not isinstance(v, (list, dict)) else f'{type(v).__name__}[{len(v)}]'))
                for k, v in list(result.items())[:6]}
    if isinstance(result, (list, tuple)):
        return f'{type(result).__name__}[{len(result)}]'
    return getattr(result, 'shape', None) or str(result)[:120]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kinds', nargs='+', default=['fastsam', 'm2t2', 'graspgenx', 'qwen', 'molmo'])
    p.add_argument('--base-port', type=int, default=52000)
    p.add_argument('--gpu', default='0')
    p.add_argument('--vlm-device', default='cuda:0', help='device for qwen/molmo (cuda:0 or cpu)')
    p.add_argument('--output', type=Path, default=ROOT/'runs/service_check')
    p.add_argument('--timeout', type=float, default=900)
    a = p.parse_args()
    reserved = set(range(51500, 51511)) | {51586}
    ports = {k: a.base_port + i for i, k in enumerate(a.kinds)}
    if reserved & set(ports.values()):
        sys.exit(f'refusing reserved ports {sorted(reserved & set(ports.values()))}')
    from cross_episode_sim.perception.rpc import ModelClient
    failures = 0
    for kind, port in ports.items():
        exe = {'m2t2': '.venv-m2t2/bin/python', 'graspgenx': '.venv-graspgenx/bin/python'}.get(kind, '.venv-sift/bin/python')
        cmd = [str(ROOT/exe), '-u', 'scripts/serve_sensor_model.py', '--kind', kind, '--port', str(port),
               '--output', str(a.output/kind)]
        if kind in ('qwen', 'molmo'):
            cmd += ['--device', a.vlm_device]
        if kind == 'fastsam':
            cmd += ['--checkpoint', os.environ.get('FASTSAM_CHECKPOINT', str(ROOT/'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt'))]
        if kind == 'm2t2':
            cmd += ['--checkpoint', os.environ['M2T2_CHECKPOINT']]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8')
        env.pop('SENSOR_MODEL_ENDPOINTS', None)
        (a.output/kind).mkdir(parents=True, exist_ok=True)
        logf = open(a.output/f'{kind}.log', 'w')
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        try:
            while True:
                text = (a.output/f'{kind}.log').read_text(errors='replace')
                if f'READY {kind}' in text:
                    break
                if proc.poll() is not None or time.time()-t0 > a.timeout:
                    raise RuntimeError(f'service did not start (exit {proc.poll()}); last log lines:\n' + '\n'.join(text.splitlines()[-15:]))
                time.sleep(2)
            load = time.time()-t0
            client = ModelClient(f'127.0.0.1:{port}', timeout=a.timeout, kind=kind)
            t1 = time.time(); result = client.call(**request(kind)); client.close()
            print(f'PASS {kind}: port {port}, loaded in {load:.0f}s, request {time.time()-t1:.1f}s -> {summary(result)}', flush=True)
        except Exception as exc:
            failures += 1
            print(f'FAIL {kind}: {type(exc).__name__}: {exc}', flush=True)
        finally:
            proc.terminate()
            try: proc.wait(30)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            logf.close()
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
