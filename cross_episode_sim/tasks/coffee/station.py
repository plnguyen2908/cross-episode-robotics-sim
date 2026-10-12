"""Author an independent coffee support without relocating kitchen food prep."""
import json
import shutil
from pathlib import Path

from cross_episode_sim.tasks.coffee.placement import surface_placement
from cross_episode_sim.tasks.scene_variation import add_furniture, sync_selection


def prepare_station(run, output, room, seed):
    """Create a roomy coffee table; preserve the dining table for place settings.

    The native living side table is too small for the machine and portafilter
    loading area together. This station is authored before physics, with a clear
    robot workspace and the same surface height as the validated counter.
    """
    if room not in ('living', 'dining'):
        raise ValueError('Coffee station room must be living or dining')
    base = Path(output) / 'station_base'
    base.mkdir()
    manifest = json.loads((run / 'task_manifest.json').read_text())
    selection = json.loads((run / 'adapter.json').read_text())
    scene = base / 'robocasa_scene.xml'
    shutil.copyfile(run / 'robocasa_scene.xml', scene)
    add_furniture(scene, manifest, seed, 1, coffee_room=room)
    record = manifest['added_furniture'][-1]
    x, y, _ = record['position']
    pose = surface_placement([x, y + record['size'][1] / 2], [0., 1.], record['size'][2])
    manifest['scene_xml'] = str(scene.resolve())
    manifest['coffee']['machine_room'] = room
    manifest['food_preparation_room'] = 'kitchen'
    sync_selection(manifest, selection)
    (base / 'task_manifest.json').write_text(json.dumps(manifest, indent=2))
    (base / 'adapter.json').write_text(json.dumps(selection, indent=2))
    return base, pose, record['body']
