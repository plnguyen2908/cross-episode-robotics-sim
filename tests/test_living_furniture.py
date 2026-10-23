import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cross_episode_sim.tasks.living_furniture import furnish_living_room


def scene(tmp_path):
    root=ET.Element('mujoco');world=ET.SubElement(root,'worldbody')
    for name, pos in {
        'doorframe_2_left_living_room_main':'5.52 -1.15 1.175',
        'east_living_room_main':'12.32 -3.6 1.5',
        'north_living_room_main':'8.9 .02 1.5',
        'south_living_room_main':'8.9 -7.26 1.5',
        'sofa_seat_living_room_main':'8.9 -6.58 .3',
        'sofa_back_living_room_main':'8.9 -6.93 .75',
        'coffee_table_living_room_main':'8.9 -5.33 .23',
        'coffee_leg_0_living_room_main':'8.4 -5.6 .21',
        'side_table_living_room_main':'11.75 -1.1 .39',
    }.items():
        ET.SubElement(world,'body',name=name,pos=pos)
    p=tmp_path/'scene.xml';ET.ElementTree(root).write(p);return p


def test_fixed_training_layout_and_rigid_eval_transform(tmp_path):
    p=scene(tmp_path)
    records=furnish_living_room(p,{'coffee_table':{'dx':.1,'dy':.1,'yaw_deg':90}})
    assert len(records)==8
    bodies={b.get('name'):b for b in ET.parse(p).findall('./worldbody/body')}
    center=np.fromstring(bodies['coffee_table_living_room_main'].get('pos'),sep=' ')
    leg=np.fromstring(bodies['coffee_leg_0_living_room_main'].get('pos'),sep=' ')
    assert leg-center == pytest.approx([.27,-.5,-.02])
    assert center == pytest.approx([9.,-5.23,.23])
    assert (tmp_path/'living_furniture.json').exists()


@pytest.mark.parametrize('config',[
    {'armchair_left':{'dx':-2}},
    {'armchair_left':{'dx':2.4}},
    {'sofa':{'dy':-3}},
    {'unknown':{}},
    {'console':{'yaw_deg':float('nan')}},
])
def test_invalid_eval_transform_does_not_modify_scene(tmp_path,config):
    p=scene(tmp_path);before=p.read_bytes()
    with pytest.raises(ValueError):
        furnish_living_room(p,config)
    assert p.read_bytes()==before


def test_rotation_only_pilot_preserves_wall_clearance(tmp_path):
    from cross_episode_sim.generation.composite_pilot import configurations
    config = next(c for c in configurations()
                  if c['task'] == 'warm_two_pots' and c['variant'] == 'furniture_rotation')
    records = furnish_living_room(scene(tmp_path), config['scene']['furniture'])
    chairs = [r for r in records if r['name'].startswith('armchair')]
    assert sorted(r['yaw_deg'] for r in chairs) == [-23., 23.]


def test_compact_chair_variant_changes_geometry(tmp_path):
    p=scene(tmp_path)
    furnish_living_room(p, furniture_set='compact_chairs')
    root=ET.parse(p).getroot()
    seat=root.find(".//geom[@name='furn_armchair_left_0']")
    assert list(map(float,seat.get('size').split())) == [.32,.32,.30]
