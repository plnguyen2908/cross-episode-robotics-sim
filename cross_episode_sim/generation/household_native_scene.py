"""Native RoboCasa scene authoring, runnable in the RoboCasa environment."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def native_house(output, params):
    """Create the requested native layout; no dependency on an earlier rollout."""
    import robosuite
    from robocasa.models.scenes import dining_room, scene_registry
    from robocasa.scripts.render_layout_gallery import NativeKitchenGallery  # noqa: F401

    from cross_episode_sim.tasks.living_furniture import furnish_living_room
    from cross_episode_sim.tasks.three_room_layout import connect_living_room

    get_path = scene_registry.get_layout_path
    original_builder = dining_room.build_three_room_layout
    from cross_episode_sim.generation.native_room_layout import build_three_room_layout
    layout_metadata = {}
    def configured_layout(rng, base_layout=1):
        layout = build_three_room_layout(rng, params['kitchen_layout'])
        layout_metadata.update(layout[1])
        asset = params.get('fridge_asset')
        if asset:
            from cross_episode_sim.paths import robocasa_dir
            registry = robocasa_dir() / 'models/assets/fixtures/fixture_registry/fridge_side_by_side.yaml'
            import yaml
            if asset not in yaml.safe_load(registry.read_text()) or asset == 'default':
                raise ValueError('Unknown side-by-side fridge asset: ' + str(asset))
            def replace(node):
                if isinstance(node, dict):
                    if node.get('name') == 'fridge':
                        node.update(type='fridge_side_by_side', config_name=asset)
                        node.pop('default_config_name', None)
                    for value in node.values(): replace(value)
                elif isinstance(node, (list, tuple)):
                    for value in node: replace(value)
            replace(layout)
        return layout
    dining_room.build_three_room_layout = configured_layout
    scene_registry.get_layout_path = lambda i: get_path(i-1100 if i in (1101, 1102, 1104, 1109) else i)
    env = None
    try:
        env = robosuite.make('NativeKitchenGallery', robots='PandaOmron',
            layout_ids=[1101], style_ids=[params['kitchen_style']],
            seed=params['seed'], use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        env.reset()
        root = ET.fromstring(env.model.get_xml())
    finally:
        if env is not None:
            env.close()
        scene_registry.get_layout_path = get_path
        dining_room.build_three_room_layout = original_builder
    (output/'native_layout_metadata.json').write_text(json.dumps(layout_metadata, indent=2))
    world = root.find('worldbody')
    # Native layouts put appliances in differently named wall groups. Expose
    # stable scene bindings while preserving their physical transforms/assets.
    for appliance in ('stove','microwave','fridge'):
        roots=[b for b in world.findall('body') if b.get('name','').startswith(appliance+'_')
               and not b.get('name','').startswith(appliance+'_housing_')
               and b.get('name','').endswith('_main')]
        if len(roots)!=1:continue
        old=roots[0].get('name').removesuffix('main')
        new=appliance+'_main_group_'
        if old==new:continue
        for node in root.iter():
            for key,value in list(node.attrib.items()):
                if value.startswith(old):node.set(key,new+value[len(old):])
    robot_prefixes = ('robot0_', 'mobilebase0_', 'left_eef_target', 'right_eef_target')
    for body in list(world.findall('body')):
        if body.get('name', '').startswith(robot_prefixes):
            world.remove(body)
    for tag in ('actuator', 'sensor', 'tendon', 'equality', 'contact', 'keyframe'):
        section = root.find(tag)
        if section is not None:
            for entry in list(section):
                if tag == 'keyframe' or any(v.startswith((*robot_prefixes, 'gripper0_')) for v in entry.attrib.values()):
                    section.remove(entry)
    root.findall('option')[-1].set('integrator', 'implicitfast')
    scene = output / 'scene.xml'
    ET.ElementTree(root).write(scene)
    if not connect_living_room(scene):
        raise ValueError('Requested native house lacks the three-room loop')
    transforms = dict(params.get('furniture', {}))
    dining = transforms.pop('dining_set', {})
    furnish_living_room(scene, transforms, params.get("furniture_set", "baseline"))
    # Transform the dining table, all its legs and all chairs about the table center.
    tree = ET.parse(scene)
    bodies = tree.getroot().find('worldbody').findall('body')
    anchor = next(b for b in bodies if b.get('name') == 'dining_table_dining_room_main')
    center = np.fromstring(anchor.get('pos'), sep=' ')
    rot = Rotation.from_euler('z', dining.get('yaw_deg', 0), degrees=True)
    shift = np.array([dining.get('dx', 0), dining.get('dy', 0), 0])
    records = []
    for b in bodies:
        name = b.get('name', '')
        if '_dining_room_' not in name or not name.startswith(('dining_', 'chair_')):
            continue
        pos = np.fromstring(b.get('pos', '0 0 0'), sep=' ')
        quat = np.fromstring(b.get('quat', '1 0 0 0'), sep=' ')
        b.set('pos', ' '.join(map(str, center + shift + rot.apply(pos-center))))
        b.set('quat', ' '.join(map(str, (rot*Rotation.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True))))
        records.append(dict(body=name, before=pos.tolist(), after=np.fromstring(b.get('pos'), sep=' ').tolist()))
    tree.write(scene)
    from cross_episode_sim.generation.room_graph import extend_room_graph
    extend_room_graph(scene, params)
    from cross_episode_sim.generation.room_routes import room_door_states
    (output / 'room_doors.json').write_text(json.dumps(room_door_states(params)))
    mechanism=params.get('room_door_mechanism','pull')
    if mechanism not in ('pull','latched'):
        raise ValueError('room_door_mechanism must be pull or latched')
    (output/'room_door_options.json').write_text(json.dumps(dict(mechanism=mechanism)))
    return scene, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene, transforms = native_house(args.output, json.loads(args.params.read_text()))
    (args.output/'dining_transforms.json').write_text(json.dumps(transforms, indent=2))
