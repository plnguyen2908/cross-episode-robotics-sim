"""Synchronized room panels preserve the separate raw first-person recorder."""
from types import SimpleNamespace

import numpy as np

from cross_episode_sim.tasks.history_views import render_history_views


def test_four_views_same_state_and_raw_camera_hook():
    calls, recorded = [], []
    class Renderer:
        def update_scene(self, data, camera, scene_option):
            calls.append((data.time, camera))
        def render(self):
            return np.full((64, 80, 3), len(calls), np.uint8)
    c = SimpleNamespace(_whole_room_cameras=['kitchen', 'dining', 'living'],
                        data=SimpleNamespace(time=3), renderer=Renderer(),
                        profile=SimpleNamespace(namespace='robot_0/', head_camera='wrist_cam'),
                        record_camera_frames=lambda frames: recorded.append(frames))
    frame = render_history_views(c, record_raw=True)
    assert frame.shape == (128,160,3)
    assert calls == [(3,'kitchen'), (3,'dining'), (3,'living'), (3,'robot_0/wrist_cam')]
    assert np.all(recorded[0][2] == 4)  # raw first-person, without panel labels
    assert [int(frame[y,x,0]) for y,x in [(40,40),(40,120),(104,40),(104,120)]] == [1,2,3,4]


def test_live_observation_refreshes_visuals_and_hides_collision_proxies(monkeypatch):
    from cross_episode_sim.evaluation import sessions
    refreshed, rendered = [], []
    option = object()
    c = SimpleNamespace(data=SimpleNamespace(time=4), model=object(), render_scene_option=option,
                        update_machine_visuals=refreshed.append)
    session = sessions.ControllerSession.__new__(sessions.ControllerSession)
    session._c = c; session._cameras = ['wrist']; session._image_size = (2,2)
    session.robot_state = lambda: np.zeros(11)
    session.sim_time = lambda: 4
    session._renderer = SimpleNamespace(update_scene=lambda *args, **kwargs: rendered.append(kwargs),
                                        render=lambda: np.zeros((2,2,3),np.uint8))
    monkeypatch.setattr(sessions, 'camera_calibration', lambda *args: {})
    observation = session.observe()
    assert refreshed == [4]
    assert rendered == [dict(camera='wrist', scene_option=option)]
    assert observation['images']['wrist'].shape == (2,2,3)


def test_room_cameras_are_fixed_inside_high_wall_corners():
    import mujoco

    from cross_episode_sim.tasks.floor_objects import FLOORS
    from cross_episode_sim.tasks.history_views import install_room_cameras, room_cameras
    from cross_episode_sim.tasks.scene_variation import floor_area
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody>' + ''.join(
        f'<body name="{body}" pos="{index*8} 0 0"><geom type="box" size="3 2 .05"/></body>'
        for index, body in enumerate(FLOORS.values())) + '</worldbody></mujoco>')
    original = spec.compile(); data = mujoco.MjData(original); mujoco.mj_forward(original, data)
    install_room_cameras(spec, original, data)
    model = spec.compile(); current = mujoco.MjData(model); mujoco.mj_forward(model, current)
    names = room_cameras(model, current)
    for room, name in zip(FLOORS, names):
        cid = model.camera(name).id
        lo, hi, top, _ = floor_area(model, current, room)
        position = current.cam_xpos[cid]
        assert np.all(position[:2] > lo) and np.all(position[:2] < hi)
        assert np.allclose(np.minimum(position[:2]-lo, hi-position[:2]),
                           [.20, .90 if room == 'kitchen' else .20])
        assert np.isclose(position[2]-top, 2.6)
        assert model.cam_bodyid[cid] == 0
        assert model.cam_mode[cid] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        assert model.cam_fovy[cid] == 75
        forward = -current.cam_xmat[cid].reshape(3,3)[:,2]
        target = np.r_[(lo+hi)/2, top+1.1]-position
        assert np.allclose(forward, target/np.linalg.norm(target))
    install_room_cameras(spec, original, data)
    assert len(list(spec.cameras)) == 3  # scene reloads never duplicate cameras
