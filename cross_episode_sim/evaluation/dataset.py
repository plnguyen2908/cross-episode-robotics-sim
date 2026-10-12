"""Validate and collect complete cross-episode examples without moving source runs."""
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

EPISODES = ('history_1', 'history_2', 'current')


def scene_layout(manifest):
    """Recorded scene layout, excluding output paths and demonstrator bookkeeping."""
    binding_fields = ('body', 'key', 'role', 'source', 'destination', 'position', 'destination_position',
                      'initial_quaternion', 'initial_source', 'vessel_type', 'filling_position',
                      'serving_position', 'serving_room', 'pair')
    coffee_fields = ('machine_pose', 'machine_room', 'support_site', 'mug_target_pose', 'mug_home_xy',
                     'portafilter_seated_pose', 'box_parking_pose')
    return dict(bindings=[{k: item[k] for k in binding_fields if k in item} for item in manifest.get('bindings', [])],
                coffee={k: manifest['coffee'][k] for k in coffee_fields if k in manifest.get('coffee', {})},
                **{k: manifest[k] for k in ('robot_spawn', 'added_furniture', 'furniture_variation',
                                            'full_breakfast', 'serving_count') if k in manifest})


def copy_scene_layouts(source, destination):
    record = json.loads((source / 'episode.json').read_text())
    layouts = {}
    for episode in record['episodes']:
        path = (source / episode['run']).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError('Episode run must be inside source example')
        layouts[episode['name']] = scene_layout(json.loads((path / 'task_manifest.json').read_text()))
    write_json(destination / 'scene_layouts.json', layouts)


def copy_initial_scenes(source, destination):
    """Freeze authored initial XML and oracle grasp annotations, never rollout state."""
    record = json.loads((source / 'episode.json').read_text())
    for episode in record['episodes']:
        original = (source / episode['run']).resolve()
        if not original.is_relative_to(source.resolve()):
            raise ValueError('Episode run must be inside source example')
        target = destination / 'initial_scenes' / episode['name']
        if (target / 'snapshot.json').exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((original / 'task_manifest.json').read_text())
        adapter = json.loads((original / 'adapter.json').read_text())
        replacements = {manifest['scene_xml']: 'robocasa_scene.xml'}
        for binding in manifest['bindings']:
            path = binding.get('grasp_path')
            if path and path not in replacements:
                name = 'grasps/' + hashlib.sha256(path.encode()).hexdigest()[:12] + '.npz'
                replacements[path] = name
                (target / 'grasps').mkdir(exist_ok=True)
                shutil.copy2(path, target / name)
        xml = Path(manifest['scene_xml'])
        tree = ET.parse(xml)
        compiler = tree.getroot().find('compiler')
        for element in tree.getroot().iter():
            filename = element.get('file')
            if not filename:
                continue
            resource = Path(filename)
            if not resource.is_absolute():
                assetdir = compiler.get('meshdir' if element.tag == 'mesh' else 'texturedir', '') if compiler is not None else ''
                resource = (xml.parent / assetdir / resource).resolve()
            if resource.is_relative_to(source.resolve()):
                name = 'assets/' + hashlib.sha256(str(resource).encode()).hexdigest()[:16] + resource.suffix
                (target / 'assets').mkdir(exist_ok=True)
                if not (target / name).exists():
                    try:
                        os.link(resource, target / name)
                    except OSError:
                        shutil.copy2(resource, target / name)
                element.set('file', name)
            else:
                element.set('file', str(resource))
        if compiler is not None:
            compiler.set('meshdir', ''); compiler.set('texturedir', '')
        tree.write(target / 'robocasa_scene.xml')
        write_json(target / 'task_manifest.json', manifest)
        write_json(target / 'adapter.json', adapter)
        write_json(target / 'snapshot.json', dict(source=str(original), replacements=replacements,
                   shared_assets='External resource paths require the normal repository data installation'))


def load_initial_scene(snapshot, output):
    """Relocate the frozen scene to a fresh evaluator directory."""
    snapshot, output = Path(snapshot), Path(output)
    metadata = json.loads((snapshot / 'snapshot.json').read_text())
    shutil.copytree(snapshot, output, dirs_exist_ok=True)
    replacements = {key: str(output / relative) for key, relative in metadata['replacements'].items()}
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return replacements.get(value, value) if isinstance(value, str) else value
    manifest = relocate(json.loads((snapshot / 'task_manifest.json').read_text()))
    adapter = relocate(json.loads((snapshot / 'adapter.json').read_text()))
    write_json(output / 'task_manifest.json', manifest)
    write_json(output / 'adapter.json', adapter)
    return manifest, adapter


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inspect_example(root, *, check_video=True):
    """Require all three successes, measured goals, completed renders, and public files."""
    root = Path(root)
    record = json.loads((root / 'episode.json').read_text())
    manifest = json.loads((root / 'run_manifest.json').read_text())
    episodes = record.get('episodes', [])
    expected = {f'history_{i}' for i in range(1, len(episodes))} | {'current'}
    if len(episodes) < 2 or {e['name'] for e in episodes} != expected:
        raise ValueError('Expected numbered histories and exactly one current episode')
    if manifest.get('prepare_only') or not (root / 'prompt.txt').read_text().strip():
        raise ValueError('Preparation-only or missing instruction')
    evidence = {}
    for episode in episodes:
        name = episode['name']
        run = (root / episode['run']).resolve()
        if not run.is_relative_to(root.resolve()):
            raise ValueError('Episode run must be inside the source example')
        report = json.loads((run / 'report.json').read_text())
        goals = report.get('composite_goal_evidence', episode.get('goal'))
        if episode.get('success') is not True or report.get('success') is not True:
            raise ValueError(f'{name}: episode failed')
        if not isinstance(goals, dict) or not goals or any(v is not True for v in goals.values()):
            raise ValueError(f'{name}: missing or failed goal evidence')
        if report.get('video_status') != 'complete':
            raise ValueError(f'{name}: video not complete')
        video = root / f'{name}.mp4'
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f'{name}: missing video')
        frames = None
        if check_video:
            import cv2
            capture = cv2.VideoCapture(str(video))
            try:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                first = capture.read()[0]
                capture.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
                last = capture.read()[0]
                if frames < 1 or not first or not last:
                    raise ValueError(f'{name}: unreadable video endpoints')
            finally:
                capture.release()
        evidence[name] = dict(goals=goals, frames=frames, report=str(run / 'report.json'))
    layout = {k: v['settings'] for k, v in manifest['episodes'].items()}
    fingerprint = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return dict(seed=record['seed'], layout_fingerprint=fingerprint, evidence=evidence)


def collect(sources, output):
    """Incremental hardlink/copy collection. Each source run retains a distinct ID."""
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    index_path = output / 'dataset.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else dict(version=1, examples=[])
    by_source = {e['source']: e for e in index['examples']}
    candidates = set()
    for source in sources:
        source = Path(source).resolve()
        if (source / 'episode.json').is_file():
            candidates.add(source)
        else:
            candidates.update(p.parent for p in source.rglob('episode.json') if not p.is_relative_to(output))
    rejected = []
    for root in sorted(candidates):
        if str(root) in by_source:
            destination = output / by_source[str(root)]['path']
            if not (destination / 'scene_layouts.json').exists():
                copy_scene_layouts(root, destination)
            copy_initial_scenes(root, destination)
            continue
        try:
            info = inspect_example(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(source=str(root), reason=str(exc)))
            continue
        identifier = root.name + '_' + hashlib.sha256(str(root).encode()).hexdigest()[:8]
        parent = output / 'examples'; parent.mkdir(exist_ok=True)
        destination = parent / identifier
        temporary = Path(tempfile.mkdtemp(prefix='.collect_', dir=parent))
        try:
            for name in ['prompt.txt', 'episode.json', 'run_manifest.json', *[f"{e['name']}.mp4" for e in json.loads((root / 'episode.json').read_text())['episodes']]]:
                target = temporary / name
                try:
                    os.link(root / name, target)
                except OSError:
                    shutil.copy2(root / name, target)
            # Keep private evidence in a separate directory; model adapters read public inputs only.
            (temporary / 'oracle_reports').mkdir()
            for name, evidence in info['evidence'].items():
                shutil.copy2(evidence['report'], temporary / 'oracle_reports' / f'{name}.json')
            write_json(temporary / 'provenance.json', dict(source=str(root), **info))
            copy_scene_layouts(root, temporary)
            copy_initial_scenes(root, temporary)
            if destination.exists():
                # A prior collection may have stopped after publishing this example, before indexing it.
                provenance = json.loads((destination / 'provenance.json').read_text())
                if provenance['source'] != str(root):
                    raise ValueError(f'Collection ID collision: {identifier}')
            else:
                temporary.rename(destination)
            entry = dict(id=identifier, path=str(destination.relative_to(output)), source=str(root),
                         seed=info['seed'], layout_fingerprint=info['layout_fingerprint'])
            by_source[str(root)] = entry
            index['examples'] = list(by_source.values())
            write_json(index_path, index)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    index['examples'] = list(by_source.values())
    index['count'] = len(index['examples'])
    index['unique_layouts'] = len({e['layout_fingerprint'] for e in index['examples']})
    write_json(index_path, index)
    write_json(output / 'rejected.json', rejected)
    return index
