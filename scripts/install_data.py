"""Install the data bundle and resolve its path tokens for this machine.

    python scripts/install_data.py                      # download the release bundle
    python scripts/install_data.py --archive data.tar.gz # install a local archive
    python scripts/install_data.py --source dist/data    # install an unpacked bundle
    python scripts/install_data.py --relocate --path-map /old/cache=/new/cache

Text files in the bundle refer to RoboCasa, MolmoSpaces and this package with
${TOKEN}s. Installation copies the bundle to CES_DATA_DIR (default <repo>/data)
and replaces each token with the local directory; see cross_episode_sim.paths.
"""

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from cross_episode_sim.paths import DATA_DIR, localize, token_roots

RELEASE_URL = "https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/ces-data-v1.tar.gz"
TEXT_SUFFIXES = {".xml", ".json", ".txt", ".md", ".jsonl"}


def install(source, destination):
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"{destination} is not empty; remove it or set CES_DATA_DIR elsewhere")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    count = 0
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text()
            if "${" in text:
                path.write_text(localize(text))
                count += 1
    return count


def relocate(destination, mappings, backup):
    """Repair explicit absolute-root mappings, backing up each changed text file."""
    destination, backup = Path(destination), Path(backup)
    if not destination.is_dir():
        raise ValueError(f"Data directory does not exist: {destination}")
    roots = []
    for old, new in mappings:
        old, new = str(Path(old).expanduser()), str(Path(new).expanduser().resolve())
        if not Path(old).is_absolute() or old == "/":
            raise ValueError(f"Expected an absolute source directory: {old}")
        if not Path(new).is_dir():
            raise ValueError(f"Replacement directory does not exist: {new}")
        roots.append((old.rstrip("/") + "/", new.rstrip("/") + "/"))
    if not roots:
        raise ValueError("Relocation requires at least one --path-map OLD=NEW")
    if backup.exists():
        raise ValueError(f"Backup directory already exists: {backup}")
    if backup.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Backup directory must be outside the data directory")
    count = 0
    for path in destination.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        original = path.read_text()
        updated = original
        for old, new in sorted(roots, key=lambda pair: -len(pair[0])):
            updated = updated.replace(old, new)
        if updated == original:
            continue
        saved = backup / path.relative_to(destination)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        path.write_text(updated)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--archive", type=Path, help="Local bundle archive (.tar.gz)")
    group.add_argument("--source", type=Path, help="Unpacked bundle directory")
    group.add_argument("--relocate", action="store_true", help="Repair paths in the existing installed data")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--backup", type=Path, help="Relocation backup directory (must not exist)")
    parser.add_argument("--url", default=RELEASE_URL)
    args = parser.parse_args()
    if args.relocate:
        if not args.path_map or any("=" not in item for item in args.path_map):
            parser.error("--relocate requires --path-map OLD=NEW (repeat for multiple roots)")
        from datetime import datetime

        backup = args.backup or DATA_DIR.with_name(
            DATA_DIR.name + "_path_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        count = relocate(DATA_DIR, [item.split("=", 1) for item in args.path_map], backup)
        print(f"Relocated {count} files in {DATA_DIR}; originals saved in {backup}")
        return
    if args.path_map or args.backup:
        parser.error("--path-map and --backup require --relocate")
    roots = token_roots()  # fails early if RoboCasa or MolmoSpaces cannot be located
    for token, root in roots.items():
        print(f"{token:24s} -> {root}")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        source = args.source
        if source is None:
            archive = args.archive
            if archive is None:
                archive = scratch / "bundle.tar.gz"
                print(f"Downloading {args.url}")
                urllib.request.urlretrieve(args.url, archive)
            with tarfile.open(archive) as bundle:
                bundle.extractall(scratch / "unpacked", filter="data")
            source = scratch / "unpacked" / "data"
        count = install(source, DATA_DIR)
    print(f"Installed data to {DATA_DIR} ({count} files localized)")


if __name__ == "__main__":
    main()
