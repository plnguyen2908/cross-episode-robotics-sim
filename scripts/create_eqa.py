"""Add three history-grounded QA questions to each existing action-test example."""
import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from cross_episode_sim.evaluation.eqa import write_eqa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/test_set_v1'))
    args = parser.parse_args()
    root = args.dataset.resolve()
    entries = json.loads((root / 'dataset.json').read_text())['examples']
    total = 0
    for entry in tqdm(entries, desc='Create EQA', unit='example'):
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Example must be inside dataset')
        total += write_eqa(path)
    print(f'Created {total} questions for {len(entries)} examples in {root}')


if __name__ == '__main__':
    main()
