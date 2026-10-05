"""Capture the code and settings used by an oracle validation run."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys


def capture_run(output, arguments):
    """Snapshot every loaded cross_episode_sim / molmo_spaces source file and the run settings."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__name__', '')
        path = getattr(module, '__file__', None)
        if not path or not path.endswith('.py') or name.split('.')[0] not in ('cross_episode_sim', 'molmo_spaces'):
            continue
        path = Path(path).resolve()
        relative = Path(*name.split('.')).with_suffix('.py')
        if path.name == '__init__.py':
            relative = relative.with_suffix('') / '__init__.py'
        content = path.read_bytes()
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(arguments).items()}
    packages = {}
    for package in ('mujoco', 'torch', 'nvidia-curobo', 'numpy', 'scipy'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    manifest = dict(source_sha256=hashes, arguments=settings, packages=packages,
                    python=sys.executable, environment={key: os.environ.get(key) for key in (
                        'CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'OMP_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    identity = dict(source_sha256={path: digest for path, digest in hashes.items()
                                   if Path(path).suffix in ('.py', '.sh')}, packages=packages,
                    arguments={key: value for key, value in settings.items() if key != 'output'})
    manifest['run_fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'run_fingerprint': manifest['run_fingerprint'],
                      'run_manifest': str(output / 'run_manifest.json')}), flush=True)
    return manifest['run_fingerprint']
