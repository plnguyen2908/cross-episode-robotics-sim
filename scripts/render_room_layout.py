"""Render one native RoboCasa kitchen in the enlarged three-room layout."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import robosuite
from PIL import Image
from robocasa.scripts.render_layout_gallery import NativeKitchenGallery  # noqa: F401

from cross_episode_sim.tasks.living_furniture import furnish_living_room
from cross_episode_sim.tasks.room_doors import install_room_doors
from cross_episode_sim.tasks.three_room_layout import connect_living_room


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', type=int, choices=(1, 2, 4, 9), required=True)
    p.add_argument('--style', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--furniture-config', type=Path, help='JSON map of furniture names to dx/dy/yaw_deg')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    print(f'Layout {args.layout}: building native kitchen', flush=True)
    # RoboCasa's island-placement helper asks the registry for the native YAML;
    # our generated 110x layouts retain exactly that native kitchen section.
    from robocasa.models.scenes import scene_registry
    native_get_layout_path = scene_registry.get_layout_path
    def gallery_layout_path(layout_id):
        if layout_id in (1101, 1102, 1104, 1109):
            layout_id = int(layout_id) - 1100
        return native_get_layout_path(layout_id)
    scene_registry.get_layout_path = gallery_layout_path
    env = robosuite.make('NativeKitchenGallery', robots='PandaOmron',
        layout_ids=[1100+args.layout], style_ids=[args.style], seed=7,
        use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    try:
        env.reset()
        scene = args.output/'scene.xml'
        scene.write_text(env.model.get_xml())
    finally:
        env.close()
        scene_registry.get_layout_path = native_get_layout_path
    assert connect_living_room(scene), 'Expected supported three-room architecture'
    furnish_living_room(scene, json.loads(args.furniture_config.read_text()) if args.furniture_config else None)
    print(f'Layout {args.layout}: installing doors and rendering', flush=True)
    spec = mujoco.MjSpec.from_file(str(scene))
    doors = install_room_doors(spec)
    spec.visual.global_.offwidth = 1400
    spec.visual.global_.offheight = 1200
    model = spec.compile(); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = []
    for name in ('floor_room_main', 'floor_dining_room_main', 'floor_living_room_main'):
        bid = model.body(name).id
        for gid in np.flatnonzero(model.geom_bodyid == bid):
            half = np.abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
            bounds.extend([data.geom_xpos[gid]-half, data.geom_xpos[gid]+half])
    bounds = np.asarray(bounds); lo, hi = bounds.min(0), bounds.max(0)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [*(lo[:2]+hi[:2])/2, 0]
    cam.azimuth = 90; cam.elevation = -90
    cam.distance = max((hi[0]-lo[0])/1.1667, hi[1]-lo[1]) / (2*np.tan(np.radians(22.5))) * 1.12 + 4.
    opt = mujoco.MjvOption(); opt.geomgroup[0] = 0; opt.sitegroup[:] = 0
    with mujoco.Renderer(model, height=1200, width=1400) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        Image.fromarray(renderer.render().copy()).save(args.output/'top_view.png')
    contacts=[]
    for contact in data.contact:
        names=[model.geom(int(g)).name or '' for g in contact.geom]
        if any(n.startswith('room_door_') for n in names) and contact.dist < -1e-5:
            contacts.append(dict(geoms=names, penetration_m=-float(contact.dist)))
    spec.to_file(str(args.output/'scene_with_doors.xml'))
    (args.output/'manifest.json').write_text(json.dumps(dict(layout=args.layout,
        style=args.style, seed=7, doors=doors, initial_door_contacts=contacts,
        robot_navigation_validated=False), indent=2))
    print(f'Layout {args.layout}: rendered; door contacts={len(contacts)}', flush=True)


if __name__ == '__main__':
    main()
