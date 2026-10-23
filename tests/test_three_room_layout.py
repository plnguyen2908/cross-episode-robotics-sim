import xml.etree.ElementTree as ET

import pytest

from cross_episode_sim.tasks.three_room_layout import connect_living_room


def test_shift_furniture_and_cut_both_wall_layers(tmp_path):
    root = ET.Element('mujoco'); world = ET.SubElement(root, 'worldbody')
    for name, pos, size in [
        ('north_living_room_main', '8 -2.98 1.5', '2.4 1.5 .02'),
        ('south_living_room_main', '8 -8.2 1.5', '2.4 1.5 .02'),
        ('east_living_room_main', '10.4 -5.6 1.5', '2.6 1.5 .02'),
        ('floor_living_room_main', '8 -5.6 -.02', '2.6 2.4 .02'),
        ('sofa_living_room_main', '8 -7.5 .3', '1 .4 .3'),
        ('wall_right_room_main', '5.52 -1.5 1.5', '1.54 1.5 .02'),
        ('wall_right_backing_room_main', '5.64 -1.5 1.38', '1.54 1.62 .1'),
    ]:
        body = ET.SubElement(world, 'body', name=name, pos=pos)
        ET.SubElement(body, 'geom', type='box', size=size)
    for part in ('left', 'right', 'header', 'infill'):
        b = ET.SubElement(world, 'body', name=f'doorframe_1_{part}_dining_room_main', pos='5.52 -4 1.175')
        ET.SubElement(b, 'geom', type='box', size='.09 .05 1.175')
    p = tmp_path/'scene.xml'; ET.ElementTree(root).write(p)
    assert connect_living_room(p)
    r = ET.parse(p)
    bodies = {b.get('name'): b for b in r.findall('./worldbody/body')}
    assert bodies['sofa_living_room_main'].get('pos') == '9.0 -6.5 0.3'
    assert bodies['floor_living_room_main'].find('geom').get('size') == '3.6 3.4 0.02'
    assert bodies['sofa_living_room_main'].find('geom').get('size') == '1 .4 .3'
    assert list(map(float, bodies['east_living_room_main'].get('pos').split())) == pytest.approx([12.4, -3.6, 1.5])
    assert 'wall_right_room_main' not in bodies
    assert 'wall_right_backing_room_main' not in bodies
    assert 'wall_right_backing_room_main_loop_header' in bodies
    assert 'doorframe_2_left_living_room_main' in bodies
    assert bodies['doorframe_1_left_dining_room_main'].get('pos') == '5.52 -4 1.175'
    before = p.read_bytes()
    assert not connect_living_room(p)
    assert before == p.read_bytes()


def test_unrelated_scene_unchanged(tmp_path):
    p = tmp_path/'scene.xml'; p.write_text('<mujoco><worldbody/></mujoco>')
    assert not connect_living_room(p)
    assert p.read_text() == '<mujoco><worldbody/></mujoco>'
