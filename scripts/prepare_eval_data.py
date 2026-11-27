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

--memory also downloads the memory of the memory setting (~60 GB of archives) into
<root>/memory and <root>/fine_tuning (the history episodes' skill logs and, for the
70 tracked history episodes, their recorded scenes): memory/{episode_memory.json,
index.json, FROZEN.json, README.md, lists/} as files and memory/archives/*.tar.zst
(checked against memory/archives/MANIFEST.json, then unpacked into <root>). It
recreates the memory/visual_tour_root links and checks that every file the memory
lists name is there. It never downloads the fine-tuning episode shards.
--revision selects the dataset revision (default main) for everything.

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


def download(root, splits, revision='main'):
    from huggingface_hub import snapshot_download
    patterns = ['eval/eval_scene_assets.tar.zst', 'episode_index.json', 'dataset_config.json', 'all_open_fixed_v4.yaml']
    for split in splits:
        patterns += [f'{split}/manifest.json', f'{split}/objects.json', f'{split}/batches/*/batch_configs.json',
                     f'episodes/{split}/*.tar.zst']
    snapshot_download(REPO, repo_type='dataset', revision=revision, local_dir=str(root), allow_patterns=patterns, max_workers=8)


MEMORY_FILES = ['memory/episode_memory.json', 'memory/index.json', 'memory/FROZEN.json', 'memory/README.md',
                'memory/lists/*.json', 'memory/archives/*']
TOUR_LINKS = {'episodes': '../visual/episodes', 'merged': '../visual/merged_tour'}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):h.update(block)
    return h.hexdigest()


def download_memory(root, revision='main'):
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type='dataset', revision=revision, local_dir=str(root), allow_patterns=MEMORY_FILES, max_workers=8)


def unpack_memory(root, delete_archives):
    archives = root / 'memory' / 'archives'
    manifest = json.loads((archives / 'MANIFEST.json').read_text())
    for name, info in sorted(manifest.items()):
        archive = archives / name;done = archives / (name + '.unpacked')
        if done.exists():continue
        if not archive.exists():raise SystemExit(f'missing {archive}; re-run with --memory to download it')
        if archive.stat().st_size != info['bytes'] or sha256(archive) != info['sha256']:
            raise SystemExit(f'{archive}: size or sha256 differs from MANIFEST.json; delete it and re-run')
        run(['bash', '-c', f"zstd -dcf '{archive}' | tar -xf - -C '{root}'"])
        done.write_text(info['sha256'] + '\n')
        if delete_archives:archive.unlink()
    tour = root / 'memory' / 'visual_tour_root';tour.mkdir(exist_ok=True)
    for name, target in TOUR_LINKS.items():
        link = tour / name
        if link.is_symlink() and os.readlink(link) == target:continue
        if link.is_symlink() or link.exists():
            if not link.is_symlink():raise SystemExit(f'{link} exists and is not a link')
            link.unlink()
        link.symlink_to(target)


def check_memory(root):
    """Every file the memory metadata names is on disk (the history logs, maps, index, commands, streams, recordings)."""
    memory = root / 'memory';missing = []
    def need(path):
        if not Path(path).exists():missing.append(str(path))
    need(memory / 'text' / 'trees')
    ids = set()
    for path in sorted((memory / 'lists').glob('*.json')):
        for e in json.loads(path.read_text())['history_episodes']:
            ids.add(e['id']);need(root / e['skill_calls'])
            folder = root / (e.get('folder') or Path(e['skill_calls']).parent)
            need(folder / 'instruction.txt')
    for i in sorted(ids):need(memory / 'actions' / i / 'commands.npz')
    entries = json.loads((memory / 'episode_memory.json').read_text())['episodes'].values()
    houses = set()
    for e in entries:
        need(root / e['history']['list']);need(root / e['visual_memory']['map'] / 'graph.json')
        houses.add(e['house_id'][:12])
    for house in sorted(houses):need(memory / 'visual' / 'memory_index_dev2' / f'{house}.json')
    plan = memory / 'history_streams' / 'plan.json';need(plan)
    if plan.exists():
        for e in json.loads(plan.read_text())['episodes']:
            need(memory / 'history_streams' / 'streams' / e['id'] / 'sensors.h5')
            need(memory / 'history_streams' / 'episodes' / e['id'] / 'trajectory.npz')
            source = Path(e['source']);source = source if source.is_absolute() else root / source
            need(source / 'action_recording' / 'model.mjb');need(source / 'action_recording' / 'actions.npz')
    print(f'memory check: {len(ids)} history episodes, {len(houses)} houses, {len(missing)} missing files', flush=True)
    for row in missing[:10]:print('   missing', row)
    if missing:raise SystemExit('memory check failed; re-run with --memory')


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
    p.add_argument('--memory', action='store_true', help='also download and unpack the memory (memory/ and the history logs)')
    p.add_argument('--memory-only', action='store_true', help='only the memory steps (the test splits are already prepared)')
    p.add_argument('--revision', default='main', help='Hugging Face dataset revision (branch, tag or commit; default main)')
    a = p.parse_args()
    if not shutil.which('zstd'):sys.exit('zstd is required (e.g. conda install zstd, or apt install zstd)')
    root = a.root.resolve();root.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    if not a.memory_only:
        if not a.skip_download:download(root, splits, a.revision)
        unpack(root, splits, a.delete_shards)
        relocate(root, splits)
        if a.check_sample:check(root, splits, a.check_sample)
    if a.memory or a.memory_only:
        if not a.skip_download:download_memory(root, a.revision)
        unpack_memory(root, a.delete_shards)
        check_memory(root)
    print(f'ready: --dataset {root}' + (' (with memory)' if a.memory or a.memory_only else ''))


if __name__ == '__main__':
    main()
