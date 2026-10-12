"""Fixed whole-room views plus the live robot camera for cross-episode videos."""
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from cross_episode_sim.tasks.scene_variation import floor_area

ROOMS = ('kitchen', 'dining', 'living')


def install_room_cameras(spec, model, data):
    """Fixed wall-corner cameras, 2.6 m above the room floor, facing inward.

    These are world-attached perspective cameras with ordinary wall occlusion.
    Their placement depends only on architecture, never task objects or robots.
    """
    from scipy.spatial.transform import Rotation

    from cross_episode_sim.tasks.floor_objects import FLOORS
    # Viewed in house coordinates: choose a corner clear of the kitchen's wall
    # cabinets and opposite the main doorway in the other two rooms.
    corners = {'kitchen': (0, 0), 'dining': (0, 0), 'living': (1, 0)}
    existing = {camera.name for camera in spec.cameras}
    for room in ROOMS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FLOORS[room]) < 0:
            continue
        name = 'history_' + room + '_wall_camera'
        if name in existing:
            continue
        low, high, top, _ = floor_area(model, data, room)
        side = np.asarray(corners[room])
        # The kitchen doorway is in the adjacent wall. Move the mount along
        # the side wall so its protruding jamb does not hide the doorway.
        inset = np.array([.20, .90 if room == 'kitchen' else .20])
        position = np.r_[np.where(side, high-inset, low+inset), top+2.6]
        # A moderate lens and shallower pitch limit stretched near-wall edges.
        target = np.r_[(low+high)/2, top+1.1]
        forward = target-position; forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        quat = Rotation.from_matrix(np.column_stack((right, up, -forward))).as_quat(scalar_first=True)
        spec.worldbody.add_camera(name=name, pos=position, quat=quat, fovy=75)


def room_cameras(model, data):
    """Return the named, physically fixed cameras installed with the house."""
    names = ['history_' + room + '_wall_camera' for room in ROOMS]
    for name in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            raise ValueError(f'Missing wall camera {name}; rebuild the house model')
    return names


def render_history_views(controller, *, record_raw=False):
    """2x2 synchronized panels, with no privileged stage/goal overlays."""
    if not hasattr(controller, '_whole_room_cameras'):
        controller._whole_room_cameras = room_cameras(controller.model, controller.data)
    frames, raw = [], []
    cameras = [*controller._whole_room_cameras, controller.profile.namespace + controller.profile.head_camera]
    for name, camera in zip((*ROOMS, 'robot wrist camera' if controller.profile.head_camera == 'wrist_cam' else 'robot head camera'), cameras):
        controller.renderer.update_scene(controller.data, camera=camera,
                                        scene_option=getattr(controller, 'render_scene_option', None))
        raw.append(controller.renderer.render().copy())
        frame = Image.fromarray(raw[-1])
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 22), fill='black')
        draw.text((8, 5), name, fill='white')
        frames.append(np.asarray(frame))
    if record_raw:
        # Existing recorder hooks expect the first-person camera at index 2.
        controller.record_camera_frames([raw[0], raw[1], raw[3]])
    return np.concatenate((np.concatenate(frames[:2], axis=1),
                           np.concatenate(frames[2:], axis=1)), axis=0)
