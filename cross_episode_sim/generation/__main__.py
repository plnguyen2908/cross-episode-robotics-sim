"""Create independently reusable histories or a current example from JSON config."""
import argparse
import importlib
import json
from pathlib import Path

from tqdm import tqdm

from .library import CurrentConfig, HistoryConfig, create_current, create_history, history_config_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('history', 'current'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--library', type=Path, default=Path('runs/history_library'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--task-module', action='append', default=[],
                        help='Import a Python module registering current tasks (repeatable)')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and print it without simulation')
    args = parser.parse_args()
    for module in args.task_module:
        importlib.import_module(module)
    values = json.loads(args.config.read_text())
    if args.kind == 'current':
        values['histories'] = tuple(str((args.config.resolve().parent / p).resolve())
                                   for p in values['histories'])
    config = (HistoryConfig if args.kind == 'history' else CurrentConfig)(**values)
    if args.dry_run:
        from dataclasses import asdict
        print(json.dumps(history_config_dict(config) if args.kind == 'history' else asdict(config), indent=2))
        return
    if args.kind == 'history' and args.output:
        parser.error('History output is controlled by --library, not --output')
    with tqdm(total=1, desc=f'Validated {args.kind}', unit='task') as progress:
        result = (create_history(config, args.library, gpu=args.gpu) if args.kind == 'history'
                  else create_current(config, args.output, gpu=args.gpu))
        progress.update()
    print(result)


if __name__ == '__main__':
    main()
