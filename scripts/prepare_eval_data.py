#!/usr/bin/env python3
"""Download and prepare the v4 test episodes for the household evaluation.

    .venv-sift/bin/python scripts/prepare_eval_data.py --root /path/to/data/v4
    # then: scripts/eval_households.py --dataset /path/to/data/v4 --split test ...

From the Hugging Face dataset plnguyen2908/cross-episodic-sim it downloads only
what the evaluation needs: the test_same_map / test_new_map episode shards
(~110 GB compressed), their manifests, and eval/eval_scene_assets.tar.zst
(40 MB: the 2,426 mesh/texture files the 300 test scenes reference). Shards are
unpacked into --root so that <root>/<split>/<manifest folder> resolves; videos
and rejected-trial folders are skipped (~not used by the evaluation). The saved
scene.xml files refer to assets by absolute paths on the machine that generated
them; those paths are rewritten to <root>/assets. A sample of scenes is then
compiled in MuJoCo as a check. Re-running skips finished steps.

Disk: ~110 GB of downloads (deleted after unpacking with --delete-shards) plus
~120-150 GB unpacked. Needs `zstd` and `tar` on PATH.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = 'plnguyen2908/cross-episodic-sim'
SPLITS = ('test_same_map', 'test_new_map')
# Where the generating machine kept the assets -> folder inside the asset archive.
PATH_MAP = (('/home/leos/.cache/molmo-spaces-resources', 'assets/molmo-spaces-resources'),
            ('/home/leos/.cache/molmospaces/assets', 'assets/molmospaces-assets'),
            ('/nobackup/le/cross-episode-robotics-sim/external/robocasa', 'assets/robocasa'),
            ('/nobackup/le/cross-episode-robotics-sim/external/robosuite', 'assets/robosuite'),
            ('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4', '.'))
SKIP_IN_SHARDS = ('*.mp4', '*/rejected_trials/*', '*/rejected_trials')


def run(command):
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def download(root, splits):
    from huggingface_hub import snapshot_download
    patterns = ['eval/eval_scene_assets.tar.zst', 'episode_index.json', 'dataset_config.json', 'all_open_fixed_v4.yaml']
    for split in splits:
        patterns += [f'{split}/manifest.json', f'{split}/objects.json', f'{split}/batches/*/batch_configs.json',
                     f'episodes/{split}/*.tar.zst']
    snapshot_download(REPO, repo_type='dataset', local_dir=str(root), allow_patterns=patterns, max_workers=8)


def unpack(root, splits, delete_shards):
    for split in splits:
        for shard in sorted((root / 'episodes' / split).glob('*.tar.zst')):
            done = shard.with_suffix('.unpacked')
            if done.exists():continue
            excludes = ' '.join(f"--exclude='{p}'" for p in SKIP_IN_SHARDS)
            run(['bash', '-c', f"zstd -dcf '{shard}' | tar -xf - -C '{root}' {excludes}"])
            done.write_text('ok\n')
            if delete_shards:shard.unlink()
    assets = root / 'eval' / 'eval_scene_assets.tar.zst'
    if not (root / 'assets' / 'MANIFEST.tsv').exists():
        run(['bash', '-c', f"zstd -dcf '{assets}' | tar -xf - -C '{root}'"])


def episodes(root, splits):
    for split in splits:
        manifest = json.loads((root / split / 'manifest.json').read_text())
        for row in manifest['episodes'] if isinstance(manifest, dict) else manifest:
            yield root / split / row['folder']


def relocate(root, splits):
    """Rewrite the generating machine's absolute paths in scene.xml / config.json."""
    rewritten = missing = 0
    for folder in episodes(root, splits):
        if not (folder / 'scene.xml').exists():
            missing += 1;continue
        for name in ('scene.xml', 'config.json'):
            path = folder / name
            if not path.exists():continue
            text = path.read_text()
            new = text
            for old, sub in PATH_MAP:new = new.replace(old, str((root / sub).resolve()))
            if new != text:
                path.write_text(new);rewritten += 1
    print(f'rewrote {rewritten} files; {missing} manifest episodes missing scene.xml', flush=True)
    if missing:raise SystemExit('Some episodes were not unpacked; re-run the script')


def check(root, splits, sample):
    """Compile a random sample of scenes; every referenced file must exist under root."""
    import mujoco
    folders = list(episodes(root, splits));random.seed(0)
    bad = []
    for folder in random.sample(folders, min(sample, len(folders))):
        text = (folder / 'scene.xml').read_text()
        stray = [f for f in re.findall(r'file="(/[^"]+)"', text) if not os.path.exists(f)]
        try:mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
        except Exception as exc:bad.append((folder.name, str(exc)[:160]))
        if stray:bad.append((folder.name, f'{len(stray)} missing files, e.g. {stray[0]}'))
    print(f'compiled {min(sample, len(folders))} of {len(folders)} test scenes: {len(bad)} problems', flush=True)
    for row in bad[:10]:print('  ', *row)
    if bad:raise SystemExit('scene check failed')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True, help='data folder (pass it to eval_households.py --dataset)')
    p.add_argument('--splits', default=','.join(SPLITS))
    p.add_argument('--delete-shards', action='store_true', help='delete each .tar.zst after unpacking it')
    p.add_argument('--check-sample', type=int, default=20, help='scenes to compile as a check (0 = none)')
    p.add_argument('--skip-download', action='store_true')
    a = p.parse_args()
    if not shutil.which('zstd'):sys.exit('zstd is required (e.g. conda install zstd, or apt install zstd)')
    root = a.root.resolve();root.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    if not a.skip_download:download(root, splits)
    unpack(root, splits, a.delete_shards)
    relocate(root, splits)
    if a.check_sample:check(root, splits, a.check_sample)
    print(f'ready: --dataset {root}')


if __name__ == '__main__':
    main()
