"""Surplus interchangeable vessels and count-based breakfast scoring."""
import copy
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cross_episode_sim.tasks.full_breakfast.scene import place_on_counter
from cross_episode_sim.tasks.servings import clone_vessel


def add_vessel_pool(manifest, selection, seed, extra):
    if not extra:
        return
    if type(extra) is not int or not 0 <= extra <= 6:
        raise ValueError('extra vessels per type must be in [0, 6]')
    from cross_episode_sim.tasks.coffee.brew import fill_site
    from cross_episode_sim.tasks.scene_variation import add_furniture
    scene = Path(manifest['scene_xml'])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    breakfast = manifest['full_breakfast']
    # Fixed pool size independent of the hidden serving count. Otherwise a
    # constant surplus would let the planner subtract it from catalog size.
    pool_size = 3 + extra
    extra = pool_size - breakfast['people']
    by_role = {i['role']: i for i in manifest['bindings']}
    originals = {kind: [by_role[s[kind]] for s in breakfast['settings']] for kind in ('cup', 'bowl')}
    keep = [i['position'][:2] for i in manifest['bindings'] if i.get('position')]
    def counter_sides():
        # Both long edges of a free-standing kitchen table are usable. Wall
        # counters retain their one outward edge.
        for key in list(manifest['supports']):
            if key.startswith('kitchen_work_table') and not key.endswith('_back'):
                manifest['supports'][key+'_back'] = manifest['supports'][key]
                manifest['outward'][key+'_back'] = (-np.asarray(manifest['outward'][key])).tolist()
        return [k for k in manifest['supports'] if k == 'kitchen' or k.startswith('kitchen_work_table')]
    counters = counter_sides()
    pool = {kind: list(values) for kind, values in originals.items()}
    for kind in ('cup', 'bowl'):
        for index in range(extra):
            tree = ET.parse(scene); root = tree.getroot()
            source = originals[kind][0]
            role = f'{kind}_pool_{index}'
            info = clone_vessel(root, source, role)
            body = root.find(f".//body[@name='{info['body']}']")
            body.set('pos', '0 0 5')
            if kind == 'cup':
                original_site = root.find(f".//body[@name='{source['body']}']/site[@name='{fill_site(source['body'])}']")
                if original_site is None:
                    raise ValueError('Coffee mug template has no fill indicator')
                for site in list(body.findall('site')):
                    if 'coffee_fill' in site.get('name', ''):
                        body.remove(site)
                site = copy.deepcopy(original_site); site.set('name', fill_site(info['body'])); body.append(site)
            else:
                info['content_geoms'] = [g.get('name') for g in body.findall('geom') if 'breakfast_contents' in g.get('name', '')]
            tree.write(scene)
            for attempt in range(3):
                try:
                    support, position = place_on_counter(scene, info['body'], counters, manifest['supports'],
                        manifest['outward'], rng, keep, clearance=.22,
                        coffee_workspace=manifest['coffee'], face_outward=kind == 'cup')
                    break
                except ValueError:
                    if attempt == 2:
                        raise
                    add_furniture(scene, manifest, seed+2000+len(pool['cup'])+len(pool['bowl'])+attempt,
                                  1, work_table=True)
                    counters = counter_sides()
            info.update(source=support, position=position, vessel_type=kind, contents='empty')
            info.pop('pair', None)
            keep.append(position[:2]); pool[kind].append(info); manifest['bindings'].append(info)
    # Random IDs prevent the catalog from revealing which objects the witness uses.
    rename = {}
    for kind, infos in pool.items():
        for info, number in zip(infos, rng.permutation(len(infos))):
            rename[info['role']] = f'{kind}_{int(number)+1:02d}'
            info['vessel_type'] = kind
    manifest['vessel_role_aliases'] = rename
    for info in manifest['bindings']:
        info['role'] = rename.get(info['role'], info['role'])
        if info.get('pair') in rename:
            info['pair'] = rename[info['pair']]
    for setting in breakfast['settings']:
        for kind in ('cup', 'bowl'):
            setting[kind] = rename[setting[kind]]
    breakfast['cups'] = [i['role'] for i in pool['cup']]
    breakfast['bowls'] = [i['role'] for i in pool['bowl']]
    manifest['coffee']['coffee_mugs'] = [i['body'] for i in pool['cup']]
    # Keep the oracle demonstration's requested subset separate from the pool.
    breakfast['interchangeable_vessels'] = True
    breakfast['extra_vessels_per_type'] = extra
    selection['selected_objects'] = manifest['bindings']


def count_serving_goal(c):
    """Check exact counts, contents, support and a distinct cup/bowl matching."""
    config = c.manifest['full_breakfast']
    count = config['people']
    dining = c.task_supports['dining']
    served = {'cup': [], 'bowl': []}
    for kind, roles in (('cup', config['cups']), ('bowl', config['bowls'])):
        for role in roles:
            info = c.info(role)
            try:
                on_table = c.assignment(objects=[info['body']])[info['body']] == dining
            except RuntimeError:
                on_table = False
            if on_table:
                served[kind].append(info)
    cups, bowls = served['cup'], served['bowl']
    correct_count = len(cups) == len(bowls) == count
    paired = correct_count and any(all(c.beside(cup['role'], bowl['role']) for cup, bowl in zip(cups, order))
                                  for order in itertools.permutations(bowls))
    return dict(served_exact_count=correct_count, served_pairs=bool(paired),
                served_vessels_placed=correct_count and all(c.served(i['role']) for i in cups+bowls),
                served_cups_filled=correct_count and all(i['body'] in c.coffee_filled for i in cups),
                served_bowls_filled=correct_count and all(i.get('contents') == 'filled' for i in bowls))
