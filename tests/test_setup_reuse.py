"""Setup reuses immutable asset work while retaining physical placement checks."""
import xml.etree.ElementTree as ET
from pathlib import Path

from cross_episode_sim.tasks import floor_objects


def test_scatter_exports_repeated_rejected_asset_only_once(tmp_path, monkeypatch):
    scene = tmp_path / 'scene.xml'
    scene.write_text('<mujoco><option timestep=".002"/><asset/><worldbody>' + ''.join(
        f'<body name="{name}" pos="{index*10} 0 -.05"><geom type="box" size="3 3 .05"/></body>'
        for index, name in enumerate(floor_objects.FLOORS.values())) + '</worldbody></mujoco>')
    exported_keys = []
    monkeypatch.setattr(floor_objects, 'candidates', lambda seed: [dict(key=k) for k in ('large', 'small1', 'small2')])
    monkeypatch.setattr(floor_objects, 'qualified_evidence', lambda row: (Path(row['key']), None, None))

    def export(recording, output):
        exported_keys.append(recording.name)
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'robocasa_scene.xml'
        radius = .5 if recording.name == 'large' else .02
        path.write_text(f'<mujoco><asset/><worldbody><body name="test_object_main">'
                        f'<freejoint name="test_object_joint"/><geom name="test_object_geom" '
                        f'type="sphere" size="{radius}" mass=".1"/></body></worldbody></mujoco>')
        return path, 'test_object_main', None, None

    monkeypatch.setattr(floor_objects, 'export_initial_scene', export)
    placed = floor_objects.scatter_objects(scene, [dict(room='kitchen')]*2, 12)
    assert [p['key'] for p in placed] == ['small1', 'small2']
    assert exported_keys == ['large', 'small1', 'small2']
    assert all(abs(p['position'][2]-.02) < .002 for p in placed)  # genuinely settled
    root = ET.parse(scene)
    assert len(root.findall('.//freejoint')) == 2  # rejected geometry was removed


def test_settle_reused_geometry_matches_fresh_compile(tmp_path):
    import mujoco
    import numpy as np

    scene = tmp_path / 'scene.xml'
    scene.write_text('<mujoco><worldbody><body name="table"><geom type="box" '
                     'size="2 2 .05" pos="0 0 -.05"/></body>'
                     '<body name="vessel" pos="0 0 1"><freejoint/>'
                     '<geom type="box" size=".02 .04 .03" mass=".1"/>'
                     '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_path(str(scene))
    initial = model.qpos0.copy()
    tree = ET.parse(scene)
    node = tree.find("./worldbody/body[@name='vessel']")
    node.set('pos', '.3 .2 .034')
    node.set('quat', '.9238795325 0 0 .3826834324')
    tree.write(scene)
    fresh = tmp_path / 'fresh.xml'
    fresh.write_bytes(scene.read_bytes())
    reused_record, fresh_record = dict(body='vessel'), dict(body='vessel')
    floor_objects.settle(scene, [reused_record], {model.body('table').id}, model=model)
    floor_objects.settle(fresh, [fresh_record], {model.body('table').id})
    np.testing.assert_allclose(reused_record['position'], fresh_record['position'], atol=1e-10)
    for attr in ('pos', 'quat'):
        a = ET.parse(scene).find("./worldbody/body[@name='vessel']").get(attr)
        b = ET.parse(fresh).find("./worldbody/body[@name='vessel']").get(attr)
        np.testing.assert_allclose(np.fromstring(a, sep=' '), np.fromstring(b, sep=' '), atol=1e-10)
    np.testing.assert_array_equal(model.qpos0, initial)
