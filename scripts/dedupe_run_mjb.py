#!/usr/bin/env python3
"""Hard-link byte-identical .mjb copies inside finished run folders.

Pipeline checkpoints save the episode's ~580 MB compiled model next to each
state; many are hard links already, but some are full copies. This replaces
every byte-identical copy (same size, same SHA-256, then a full compare) with
a hard link to one file. Every path and every file's contents stay the same;
only duplicate disk blocks are freed. Folders that a running process uses are
skipped. Usage: python3 scripts/dedupe_run_mjb.py [--dry-run] FOLDER...
"""
import argparse, filecmp, hashlib, os, subprocess, sys
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):h.update(block)
    return h.hexdigest()


def busy(folder):
    out = subprocess.run(['pgrep', '-af', str(folder)], capture_output=True, text=True).stdout
    return any(line and str(os.getpid()) != line.split()[0] and 'dedupe_run_mjb' not in line for line in out.splitlines())


def dedupe(folder, dry_run=False):
    by_size = defaultdict(list)
    for path in folder.rglob('*.mjb'):
        if path.is_file() and not path.is_symlink():by_size[(path.stat().st_dev, path.stat().st_size)].append(path)
    freed = linked = 0
    for (_, size), paths in by_size.items():
        inodes = {}
        for p in paths:inodes.setdefault(p.stat().st_ino, p)
        if len(inodes) < 2:continue
        groups = defaultdict(list)
        for p in inodes.values():groups[digest(p)].append(p)
        for same in groups.values():
            keep = same[0]
            for other in same[1:]:
                if not filecmp.cmp(keep, other, shallow=False):continue
                # every path that shares the duplicate's inode is relinked
                ino = other.stat().st_ino
                for p in paths:
                    if p.stat().st_ino != ino:continue
                    if not dry_run:
                        tmp = p.with_name('.' + p.name + '.dedupe_link')
                        if tmp.exists():tmp.unlink()
                        os.link(keep, tmp);os.replace(tmp, p)
                    linked += 1
                freed += size
    return linked, freed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', nargs='+', type=Path);p.add_argument('--dry-run', action='store_true')
    a = p.parse_args();total = 0
    for folder in a.folders:
        folder = folder.resolve()
        if busy(folder):print(f'skip {folder}: in use by a running process');continue
        linked, freed = dedupe(folder, a.dry_run);total += freed
        print(f'{folder.name}: {"would link" if a.dry_run else "linked"} {linked} files, {freed / 1e9:.1f} GB', flush=True)
    print(f'total {"would free" if a.dry_run else "freed"} {total / 1e9:.1f} GB')


if __name__ == '__main__':sys.exit(main())
