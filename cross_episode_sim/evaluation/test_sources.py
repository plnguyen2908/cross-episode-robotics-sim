"""Select official test episodes from split manifests, never directory scans."""
import json
import os
from pathlib import Path


def test_sources(dataset, splits, task, resolve=True):
    """resolve=False keeps the listed path (a link stays a link): the memory index
    (memory/episode_memory.json) is found from the listed v4 path, not its v3 target."""
    sources = []
    for split in splits:
        root = (Path(dataset) / split).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        for row in manifest['episodes']:
            if task != 'all' and row['task'] != task:
                continue
            # The listed folder must lie inside the split as written. It may be
            # a link (v4 test_same_map links the v3 episodes it keeps), so the
            # check is on the normalised path, not the resolved target.
            listed = Path(os.path.normpath(root / row['folder']))
            if not listed.is_relative_to(root):
                raise ValueError(f'Manifest path escapes split: {listed}')
            source = listed.resolve() if resolve else listed
            if source in sources:
                raise ValueError(f'Duplicate test episode: {source}')
            sources.append(source)
    return sources


test_sources.__test__ = False
