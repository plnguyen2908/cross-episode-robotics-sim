"""Serving contracts, reproducible sampling, and furniture access checks."""
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from cross_episode_sim.skills.breakfast_interface import current_skill_plan
from cross_episode_sim.tasks.full_breakfast.__main__ import command, sample
from cross_episode_sim.tasks.scene_variation import add_furniture, floor_area, preserves_connections, sync_selection
from cross_episode_sim.tasks.servings import clone_vessel, coffee_roles, vessel_roles


def test_randomized_vessels_stay_out_of_portafilter_arm_workspace():
    from cross_episode_sim.tasks.coffee.placement import clear_manipulation_workspace, to_world
    spec = dict(machine_offset=[.04205918872815523, -.008887507095925444])
    # Real seed-19 mug: clear of the parked basket, but struck by link 5
    # during the detach skill's lowering motion.
    assert not clear_manipulation_workspace(spec, [1.909606923107665, -.5122738640819334])
    for pose in [spec, dict(machine_pose=dict(dx=.3, dy=.2, dz=0, yaw_deg=90))]:
        assert not clear_manipulation_workspace(pose, to_world(pose, [1.9, -.5]))
        assert not clear_manipulation_workspace(pose, to_world(pose, [1.60, -.5]))
        assert clear_manipulation_workspace(pose, to_world(pose, [.8, -.5]))


@pytest.mark.parametrize('people', [1, 2, 3, 4])
def test_history_current_counts_and_public_plan(people):
    order, episodes = sample(19, 'varied', people=people, extra_furniture=2)
    assert len(order) == 2
    for task, settings in episodes.items():
        assert settings['people'] == people
        settings['prepare_only'] = True
        args = command(task, settings, '/tmp/unused', 'test')
        assert args[args.index('--people' if task == 'breakfast' else '--cups')+1] == str(people)
        assert '--prepare-only' in args
        assert args[args.index('--extra-furniture')+1] == '2'
    cups = coffee_roles(people)
    bowls = [r for r in vessel_roles(people) if r.startswith('bowl')]
    manifest = dict(coffee=dict(coffee_mugs=[r+'_body' for r in cups]),
                    bindings=[dict(role=r, body=r+'_body') for r in cups+bowls],
                    full_breakfast=dict(settings=[dict(cup=c, bowl=b) for c, b in zip(cups, bowls)]))
    plan = current_skill_plan(manifest)
    assert len(plan) == 8+13*people
    served = [p['arguments']['role'] for p in plan if p['operation'] == 'place' and p['arguments']['destination'] == 'serving']
    assert set(served) == set(cups+bowls) and len(served) == 2*people
    assert sum(p['operation'] == 'press_brew' for p in plan) == people


def test_seeded_variation_and_count_override_independence():
    assert sample(19, 'varied') == sample(19, 'varied')
    assert sample(19, 'varied') != sample(20, 'varied')
    assert {sample(s, 'varied')[1]['breakfast']['people'] for s in range(20)} == {1, 2, 3}
    assert sample(7, 'baseline')[1]['coffee']['placement_seed'] == 7756
    original = sample(19, 'varied')[1]
    changed = sample(19, 'varied', people=1)[1]
    for task, settings in original.items():
        assert {k: v for k, v in settings.items() if k != 'people'} == {
            k: v for k, v in changed[task].items() if k != 'people'}
        if task != 'breakfast':
            dx, dy = settings['machine_offset']
            assert -.25 <= dx <= .22 and -.015 <= dy <= .015
            args = command(task, settings, '/tmp/unused', 'test')
            assert args[args.index('--machine-room')+1] == settings['machine_room']
            if settings['machine_room'] == 'kitchen':
                start = args.index('--machine-offset')+1
                assert list(map(float, args[start:start+2])) == [dx, dy]
            else:
                assert '--machine-offset' not in args
    for invalid in [0, 5, True, 1.5]:
        with pytest.raises(ValueError):
            sample(7, 'varied', people=invalid)


def test_machine_room_sampling_includes_other_rooms_and_preserves_baseline():
    rooms = {sample(seed, 'varied')[1][task]['machine_room']
             for seed in range(20) for task in ('coffee', 'full_breakfast')}
    assert rooms == {'kitchen', 'living', 'dining'}
    assert 'machine_room' not in sample(19, 'baseline')[1]['full_breakfast']


def test_bowl_filling_rejects_coffee_support_outside_kitchen():
    from types import SimpleNamespace

    from cross_episode_sim.tasks.full_breakfast.episode import FullBreakfastWorkflow
    support = ['kitchen_counter']
    c = SimpleNamespace(task_supports={'kitchen': 'kitchen_counter', 'coffee_station': 'living_table'},
                        manifest={'full_breakfast': {'bowls': ['bowl']}},
                        info=lambda role: {'body': 'bowl_body'},
                        assignment=lambda objects: {'bowl_body': support[0]})
    assert FullBreakfastWorkflow.bowls_in_kitchen(c)
    support[0] = 'living_table'
    assert not FullBreakfastWorkflow.bowls_in_kitchen(c)
    c.bowls_in_kitchen = lambda: FullBreakfastWorkflow.bowls_in_kitchen(c)
    with pytest.raises(RuntimeError, match='kitchen support'):
        FullBreakfastWorkflow.fill_bowls(c)


def test_clone_preserves_assets_but_owns_body_and_joint_names():
    root = ET.fromstring('<mujoco><worldbody><body name="bowl_two_test_object_main"><freejoint name="bowl_two_joint"/><geom name="bowl_two_geom" mesh="bowl_two_mesh" material="ceramic"/></body></worldbody></mujoco>')
    info = dict(role='bowl_two', body='bowl_two_test_object_main', position=[0, 0, 1])
    before = copy.deepcopy(info)
    new = clone_vessel(root, info, 'bowl_three')
    node = root.find(".//body[@name='bowl_three_test_object_main']")
    assert node.find('freejoint').get('name') == 'bowl_three_joint'
    assert node.find('geom').get('mesh') == 'bowl_two_mesh'
    assert new['body'] != info['body'] and info == before
    assert len(root.findall('.//body')) == 2


def test_furniture_cannot_cut_a_corridor():
    free = np.ones((9, 9), bool)
    harmless = free.copy(); harmless[0, 0] = False
    blocked = free.copy(); blocked[4, :] = False
    assert preserves_connections(free, harmless)
    assert not preserves_connections(free, blocked)


def test_new_support_is_registered_for_contact_assignment_and_navigation():
    manifest = dict(scene_xml='scene.xml', supports=dict(kitchen='counter', kitchen_work_table='work_table'),
                    outward=dict(kitchen=[0, -1], kitchen_work_table=[0, 1]),
                    bindings=[dict(body='cup', source='kitchen_work_table', position=[2., -2., .97])])
    selection = dict(tables=[dict(body='counter', objects=[])], empty_robot_stances={'counter': []})
    sync_selection(manifest, selection)
    sync_selection(manifest, selection)
    assert [t['body'] for t in selection['tables']] == ['counter', 'work_table']
    assert selection['selected_objects'] == manifest['bindings']
    np.testing.assert_allclose(selection['empty_robot_stances']['work_table'][0], [2, -1.55, -np.pi/2])


def test_rotated_floor_bounds_and_real_furniture(tmp_path):
    scene = tmp_path/'scene.xml'
    scene.write_text('''<mujoco><worldbody>
      <body name="floor_room_main" pos="0 0 -.02" quat=".70710678 0 0 .70710678"><geom type="box" size="2 4 .02"/></body>
      <body name="floor_dining_room_main" pos="0 -6 -.02"><geom type="box" size="4 4 .02"/></body>
      <body name="floor_living_room_main" pos="8 -6 -.02"><geom type="box" size="4 4 .02"/></body>
    </worldbody></mujoco>''')
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    lo, hi, _, _ = floor_area(model, data, 'kitchen')
    np.testing.assert_allclose(hi-lo, [8, 4], atol=1e-6)
    manifest = dict(bindings=[], robot_spawn=[0, 0, 0])
    add_furniture(scene, manifest, 19, 3)
    assert len(manifest['added_furniture']) == 3
    model = mujoco.MjModel.from_xml_path(str(scene))
    for record in manifest['added_furniture']:
        assert model.body(record['body']).id > 0
        assert record['kind'] in ('stool', 'side_table', 'low_cabinet')


@pytest.mark.parametrize('room', ['living', 'dining'])
def test_coffee_station_preserves_kitchen_support_and_has_correct_room(tmp_path, room):
    import json

    from cross_episode_sim.tasks.coffee.placement import VALIDATED_MOUNT, place_spec, to_world
    from cross_episode_sim.tasks.coffee.station import prepare_station
    run = tmp_path / 'base'; run.mkdir()
    out = tmp_path / 'out'; out.mkdir()
    (run / 'robocasa_scene.xml').write_text('''<mujoco><worldbody>
      <body name="floor_room_main" pos="0 0 -.02"><geom type="box" size="4 4 .02"/></body>
      <body name="floor_dining_room_main" pos="0 -9 -.02"><geom type="box" size="4 4 .02"/></body>
      <body name="floor_living_room_main" pos="9 -9 -.02"><geom type="box" size="4 4 .02"/></body>
    </worldbody></mujoco>''')
    manifest = dict(bindings=[], coffee={}, supports={'kitchen': 'counter'}, outward={'kitchen': [0, -1]})
    (run / 'task_manifest.json').write_text(json.dumps(manifest))
    (run / 'adapter.json').write_text(json.dumps(dict(tables=[dict(body='counter')], empty_robot_stances={})))
    base, pose, support = prepare_station(run, out, room, 19)
    authored = json.loads((base / 'task_manifest.json').read_text())
    assert authored['supports']['kitchen'] == 'counter'
    assert authored['supports']['coffee_station'] == support != 'counter'
    assert authored['food_preparation_room'] == 'kitchen'
    assert authored['coffee']['machine_room'] == room
    spec = {}; place_spec(spec, pose)
    model = mujoco.MjModel.from_xml_path(str(base / 'robocasa_scene.xml'))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    lo, hi, _, _ = floor_area(model, data, room)
    mount = to_world(spec, VALIDATED_MOUNT)
    assert np.all(mount[:2] >= lo) and np.all(mount[:2] <= hi)
    assert mount[2] == pytest.approx(.922)


@pytest.mark.parametrize('task', ['breakfast', 'coffee', 'full_breakfast'])
def test_navigation_speed_settings_reach_every_episode(task):
    _, episodes = sample(20, 'varied')
    settings = episodes[task]
    settings.update(nav_speed=.24, turn_speed=.6)
    args = command(task, settings, '/tmp/unused', 'test')
    assert args[args.index('--nav-speed')+1] == '0.24'
    assert args[args.index('--turn-speed')+1] == '0.6'


@pytest.mark.parametrize('task', ['breakfast', 'coffee', 'full_breakfast'])
def test_batch_without_explicit_speeds_uses_validated_faster_defaults(task):
    _, episodes = sample(20, 'varied')
    args = command(task, episodes[task], '/tmp/unused', 'test')
    assert args[args.index('--nav-speed')+1] == '0.24'
    assert args[args.index('--turn-speed')+1] == '0.6'
