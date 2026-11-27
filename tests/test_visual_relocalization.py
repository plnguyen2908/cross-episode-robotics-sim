import json, math
from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.navigation import visual_relocalization as vr
from cross_episode_sim.sensors.onboard import RGBDCamera

MEMORY_K = np.array([[293.2257088892713, 0, 299.5], [0, 293.2257088892713, 224.5], [0, 0, 1]])


def camera(width=640, height=480, fovy=75.):
    f = height/2/math.tan(math.radians(fovy)/2)
    K = np.array([[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1]])
    return RGBDCamera('navigation', np.zeros((height, width, 3), np.uint8), np.ones((height, width), np.float32),
                      np.ones((height, width), bool), K, np.zeros(5), np.eye(4))


def memory(tmp_path):
    merged = tmp_path/'merged'/'abcdefabcdef_set_table'; merged.mkdir(parents=True)
    (merged/'graph.json').write_text(json.dumps(dict(episodes={'ep0': {}}, components={'ep0': 0}, nodes=[])))
    (tmp_path/'episodes'/'ep0').mkdir(parents=True)
    (tmp_path/'episodes'/'ep0'/'calibration.json').write_text(json.dumps(dict(K=MEMORY_K.tolist(), mount=np.eye(4).tolist())))
    return merged


def test_resize_640_to_memory_intrinsics():
    out = vr.resize_camera(camera(), MEMORY_K)
    assert out.rgb.shape == (450, 600, 3) and out.depth_m.shape == (450, 600) and out.valid_depth.dtype == bool
    assert np.allclose(out.K, MEMORY_K)


def test_resize_refuses_other_field_of_view():
    with pytest.raises(ValueError):
        vr.resize_camera(camera(fovy=60.), MEMORY_K)


def test_memory_directory_from_root_house_task(tmp_path):
    merged = memory(tmp_path)
    assert vr.memory_directory(tmp_path, 'abcdefabcdef' + '0'*52, 'set_table') == merged
    assert vr.memory_directory(merged) == merged
    with pytest.raises(FileNotFoundError):
        vr.memory_directory(tmp_path, 'ffffff' + '0'*58, 'set_table')


def pose(yaw):
    T = np.eye(4); T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]; T[:2, 3] = [1., 2.]
    return T


def test_keeps_a_view_every_10_degrees_and_finishes_after_full_turn(tmp_path, monkeypatch):
    reloc = vr.ScanRelocalization(memory(tmp_path))
    ran = []
    monkeypatch.setattr(vr.ScanRelocalization, 'finish', lambda self, reason='full_turn': ran.append(reason) or setattr(self, 'done', True))
    cam = camera(); finished = False
    for step in range(0, 400):  # 1 degree per frame
        lost = SimpleNamespace(accepted=False, T_map_base=np.eye(4))
        assert not reloc.observe(cam, lost)  # lost tracking adds nothing
        finished = reloc.observe(cam, SimpleNamespace(accepted=True, T_map_base=pose(math.radians(step))))
        if finished: break
    assert finished and ran == ['full_turn'] and step in (350, 351)
    assert len(reloc.cameras) == 36
    # Poses are relative to the scan start, so the first is the identity.
    assert np.allclose(reloc.poses[0], np.eye(4))
    assert math.isclose(math.degrees(math.atan2(reloc.poses[1][1, 0], reloc.poses[1][0, 0])), 10, abs_tol=1e-6)


def test_short_scan_reports_too_few_views(tmp_path):
    reloc = vr.ScanRelocalization(memory(tmp_path))
    reloc.observe(camera(), SimpleNamespace(accepted=True, T_map_base=pose(0.)))
    assert reloc.finish('exploration_ended')['status'] == 'too_few_scan_views'


def test_memory_gate_keeps_memory_only_when_confident():
    from cross_episode_sim.navigation.visual_relocalization import decide, MEMORY_FIT_GATE
    assert decide(dict(accepted=True, fine_fit_mean_capped_distance_m=MEMORY_FIT_GATE-.001))['use_memory']
    assert decide(dict(accepted=True, fine_fit_mean_capped_distance_m=MEMORY_FIT_GATE))['decision'] == 'use memory'
    assert not decide(dict(accepted=True, fine_fit_mean_capped_distance_m=MEMORY_FIT_GATE+.001))['use_memory']
    assert not decide(dict(accepted=False, fine_fit_mean_capped_distance_m=.01))['use_memory']
    assert decide(dict(accepted=True))['decision'] == 'explore (memory not confident)'


def test_explore_handover_only_after_confident_relocalization(tmp_path):
    import json
    from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline as SensorSetTable
    (tmp_path/'report.json').write_text(json.dumps(dict(memory_localized=True, memory_handover=dict(fit_m=.02))))
    assert SensorSetTable.memory_handover(tmp_path) == dict(fit_m=.02)
    (tmp_path/'report.json').write_text(json.dumps(dict(memory_localized=False, memory_handover=None)))
    assert SensorSetTable.memory_handover(tmp_path) is None
