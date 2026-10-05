"""Install the data bundle and resolve its path tokens for this machine.

    python scripts/install_data.py                      # download the release bundle
    python scripts/install_data.py --archive data.tar.gz # install a local archive
    python scripts/install_data.py --source dist/data    # install an unpacked bundle

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

RELEASE_URL = (
    "https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/ces-data-v1.tar.gz"
)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--archive", type=Path, help="Local bundle archive (.tar.gz)")
    group.add_argument("--source", type=Path, help="Unpacked bundle directory")
    parser.add_argument("--url", default=RELEASE_URL)
    args = parser.parse_args()
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
