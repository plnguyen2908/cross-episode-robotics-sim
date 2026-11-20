"""Select official test episodes from split manifests, never directory scans."""
import json
from pathlib import Path


def test_sources(dataset, splits, task):
    sources = []
    for split in splits:
        root = (Path(dataset) / split).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        for row in manifest['episodes']:
            if task != 'all' and row['task'] != task:
                continue
            source = (root / row['folder']).resolve()
            if not source.is_relative_to(root):
                raise ValueError(f'Manifest path escapes split: {source}')
            if source in sources:
                raise ValueError(f'Duplicate test episode: {source}')
            sources.append(source)
    return sources


test_sources.__test__ = False
