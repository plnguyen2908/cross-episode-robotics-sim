"""Discover qualified loose food, preserving tested size and provenance.

Read RoboCasa's category declarations without importing its rendering stack.
This is a semantic and geometry filter, not a new manipulation sweep.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cross_episode_sim.tasks.breakfast.scene import SWEEP, bounds, category, qualified_evidence

from cross_episode_sim.paths import robocasa_dir

FOOD_TYPES = {'vegetable', 'fruit', 'sweets', 'dairy', 'meat', 'bread_food',
              'pastry', 'cooked_food', 'iced_item'}
CUP_TYPES = {'fruit', 'sweets', 'dairy', 'iced_item'}


def food_types():
    """Discover all declared loose-food categories; packaged condiments stay out."""
    path = robocasa_dir() / 'models/objects/kitchen_objects.py'
    tree = ast.parse(path.read_text())
    declaration = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'OBJ_CATEGORIES' for t in n.targets))
    result = {}
    for entry in declaration.keywords:
        types = next(k.value for k in entry.value.keywords if k.arg == 'types')
        values = ast.literal_eval(types)
        values = {values} if isinstance(values, str) else set(values)
        if values & FOOD_TYPES and not values & {'packaged_food', 'condiment'}:
            result[entry.arg] = sorted(values & FOOD_TYPES)
    return result


def food_category(row, types):
    name = category(row)
    # Molmo's RoboTHOR identifiers carry a category followed by vendor/model IDs.
    if name.startswith('robothor_'):
        name = next((c for c in sorted(types, key=len, reverse=True)
                     if name.startswith('robothor_' + c + '_')), name)
    return name if name in types else None


def discover_food(rows):
    """Inspect every registry row, measuring only qualified edible candidates."""
    types = food_types()
    candidates, excluded = [], []
    cache_path = SWEEP / 'physical_food_geometry_cache.json'
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    for row in sorted(rows, key=lambda r: r['key']):
        cat = food_category(row, types)
        if cat is None:
            excluded.append(dict(key=row['key'], reason='not loose food or ice'))
            continue
        if row.get('placement_passes', 0) < 1 or row.get('qualified_grasps', 0) < 1:
            excluded.append(dict(key=row['key'], reason='no successful grasp and placement'))
            continue
        try:
            recording, setup, _ = qualified_evidence(row)
            source = recording / 'scene.xml'
            # Exported scene XML contains qualified scale and resolved collision
            # geometry. Track referenced asset stats as well as model/setup bytes.
            files = [source, recording / 'setup.json']
            files.append(Path(setup['model_xml']))
            for node in ET.parse(source).getroot().findall('./asset/*[@file]'):
                path = Path(node.get('file'))
                files.append(path if path.is_absolute() else source.parent / path)
            signature = hashlib.sha256(source.read_bytes() + json.dumps([
                (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files
            ]).encode()).hexdigest()
            saved = cache.get(row['key'], {})
            if saved.get('signature') == signature:
                geometry = saved['geometry']
            else:
                model = mujoco.MjModel.from_xml_path(str(source))
                data = mujoco.MjData(model); mujoco.mj_forward(model, data)
                low, high = bounds(model, data, 'test_object_main')
                geometry = dict(half_extents=((high-low)/2).tolist(),
                    mass_kg=float(model.body_subtreemass[model.body('test_object_main').id]))
                cache[row['key']] = dict(signature=signature, geometry=geometry)
            candidates.append(dict(key=row['key'], category=cat, types=types[cat],
                                   source=row['source'], **geometry))
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            excluded.append(dict(key=row['key'], reason='unavailable or stale evidence', detail=str(exc)))
    # Atomic replacement: other scene-preparation processes never read half a cache.
    temporary = cache_path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(cache, indent=2)); temporary.replace(cache_path)
    return dict(candidates=candidates, excluded=excluded, registry_rows=len(rows))


def fit_reason(candidate, cavity, vessel_type):
    if vessel_type == 'cup' and (not set(candidate['types']) & CUP_TYPES
                                 or candidate.get('category') == 'egg'):
        return 'food category is unsuitable for a cup'
    half = np.asarray(candidate['half_extents'])
    if not np.all(np.isfinite(half)) or np.any(half <= 0):
        return 'invalid food dimensions'
    if np.linalg.norm(half[:2]) + .002 > cavity['radius']:
        return 'too wide for measured basin at qualified scale'
    if 2*half[2] > cavity['rim_z'] - cavity['bottom_z'] - .002:
        return 'too tall to stay below measured rim at qualified scale'
    return None


def choose_food(candidates, seed, used):
    """Category-balanced, seeded variety; avoid repeats when alternatives exist."""
    rng = random.Random(seed)
    fresh = [c for c in candidates if c['key'] not in used]
    pool = fresh or candidates
    if not pool:
        raise ValueError('No qualified physical food fits this vessel')
    used_categories = {c['category'] for c in candidates if c['key'] in used}
    categories = sorted({c['category'] for c in pool})
    new_categories = [c for c in categories if c not in used_categories]
    cat = rng.choice(new_categories or categories)
    return rng.choice(sorted((c for c in pool if c['category'] == cat), key=lambda c: c['key']))
