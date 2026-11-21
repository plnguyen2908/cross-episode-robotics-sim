"""Fetch a minimal CUDA compiler toolkit (nvcc + cudart headers/libs) without sudo.

Used by scripts/setup_eval_env.sh when no CUDA 12.x nvcc is available, to compile
torch CUDA extensions (cuRobo, M2T2 pointnet2_ops) against torch 2.7.1+cu126.
Packages come from the conda `nvidia` channel (same source as
scripts/setup_m2t2_cuda.py), unpacked into one prefix laid out like
/usr/local/cuda (bin/, include -> targets/x86_64-linux/include, lib64 -> .../lib).

    python scripts/fetch_cuda_toolkit.py <prefix> [--version 12.6]

Needs `zstandard` (in requirements-eval-sift.txt) for .conda archives.
"""
import argparse
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOTS = ['cuda-nvcc-tools', 'cuda-nvcc-dev_linux-64', 'cuda-nvvm-impl', 'cuda-nvvm-tools',
         'cuda-cudart-dev_linux-64', 'cuda-cccl_linux-64', 'cuda-crt-dev_linux-64',
         'cuda-crt-tools', 'cuda-profiler-api']


def get(url, timeout=300):
    req = urllib.request.Request(url, headers={'User-Agent': 'fetch-cuda-toolkit'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prefix', type=Path)
    p.add_argument('--version', default='12.6')
    a = p.parse_args()
    root = a.prefix.resolve()
    root.mkdir(parents=True, exist_ok=True)
    queue, done, got = list(ROOTS), set(), []
    while queue:
        name = queue.pop(0)
        if name in done or name == 'cuda-version':
            continue
        done.add(name)
        try:
            meta = json.loads(get('https://api.anaconda.org/package/nvidia/' + name, 60))
        except Exception as exc:  # optional split packages do not exist in every release
            print('SKIP', name, exc, flush=True)
            continue
        files = [x for x in meta.get('files', [])
                 if x['version'].startswith(a.version + '.') and x['basename'].startswith(('linux-64', 'noarch'))]
        if not files:
            print('SKIP', name, '(no', a.version, 'build)', flush=True)
            continue
        # Highest patch version, newest upload of it.
        x = sorted(files, key=lambda f: ([int(v) for v in f['version'].split('.') if v.isdigit()], f.get('upload_time', '')))[-1]
        print('Downloading', x['basename'], flush=True)
        got.append(x['basename'])
        queue += [s.split()[0] for s in x.get('attrs', {}).get('depends', []) if s.startswith('cuda-')]
        blob = get('https:' + x['download_url'])
        if x['basename'].endswith('.conda'):
            import zstandard
            z = zipfile.ZipFile(io.BytesIO(blob))
            member = next(n for n in z.namelist() if n.startswith('pkg-'))
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(member))) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as t:
                    t.extractall(root, filter='data')
        else:
            with tarfile.open(fileobj=io.BytesIO(blob), mode='r:bz2') as t:
                t.extractall(root, filter='data')
    target = root / 'targets/x86_64-linux'
    for link, dest in (('include', target / 'include'), ('lib64', target / 'lib')):
        path = root / link
        if dest.is_dir() and not path.exists():
            os.symlink(dest, path)
    (root / 'FETCHED_PACKAGES.txt').write_text('\n'.join(got) + '\n')
    nvcc = root / 'bin/nvcc'
    if not nvcc.is_file():
        raise SystemExit(f'nvcc missing after fetch: {nvcc}')
    print('CUDA toolkit ready:', root)


if __name__ == '__main__':
    main()
