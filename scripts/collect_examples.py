"""Gather successful three-episode examples into one incrementally updated dataset."""
import argparse
import fcntl
import json
from pathlib import Path

from cross_episode_sim.evaluation.dataset import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', default=[Path('runs')])
    parser.add_argument('--output', type=Path, default=Path('datasets/successful_examples'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.collection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = collect(args.source, args.output)
    print(json.dumps(dict(dataset=str(args.output.resolve()), examples=index['count'],
                          unique_layouts=index['unique_layouts'])), flush=True)


if __name__ == '__main__':
    main()
