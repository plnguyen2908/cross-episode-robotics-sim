import xml.etree.ElementTree as ET

import pytest

from cross_episode_sim.fixtures import stove_knob


@pytest.mark.parametrize('field,value', [('euler', '0 0 -1.5708'), ('quat', '.7071 0 0 .7071')])
def test_replacement_preserves_gallery_stove_orientation(tmp_path, monkeypatch, field, value):
    scene = tmp_path/'scene.xml'
    scene.write_text(f'<mujoco><asset/><worldbody><body name="stove_main_group_main" pos="4 2 .5" {field}="{value}"/></worldbody></mujoco>')
    native = '<mujoco><asset/><worldbody><body name="stove_main_group_main" quat="1 0 0 0"/></worldbody></mujoco>'
    monkeypatch.setattr(stove_knob, 'read_localized', lambda p: native)
    stove_knob.install_front_control_stove(scene)
    body = ET.parse(scene).find('./worldbody/body')
    assert body.get(field) == value
    assert body.get('pos') == '4 2 .5'
    assert sum(k in body.attrib for k in ('quat','euler','axisangle','xyaxes','zaxis')) == 1
