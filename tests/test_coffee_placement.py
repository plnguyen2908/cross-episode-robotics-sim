import xml.etree.ElementTree as ET

import numpy as np

from cross_episode_sim.tasks.coffee.placement import (
    SLOTS, blocking_fixtures, counter_for, offset, sample_offset, shift, shift_spec)


def scene_with_fixtures():
    root = ET.fromstring('<mujoco><worldbody>'
                         '<body name="paper_towel_main_group_main" pos="1.917 -0.119 1.088"/>'
                         '<body name="toaster_main_group_main" pos="3.698 -0.194 1.018"/>'
                         '<body name="knife_block_main_group_main" pos="4.072 -0.167 1.086"/>'
                         '</worldbody></mujoco>')
    return root


def test_sampled_offsets_stay_inside_their_slot():
    rng = np.random.default_rng(3)
    for _ in range(50):
        name, off = sample_offset(rng)
        low, high = SLOTS[name]
        assert low <= off[0] <= high and off[1] == 0.


def test_shift_spec_moves_world_points_and_records_offset():
    spec = dict(lid_anchor=[2.13, -.53, 1.03], machine_x=2.48, dosing_parking_xy=[3.84, -.52])
    shift_spec(spec, np.array([1.55, 0., 0.]))
    np.testing.assert_allclose(spec['lid_anchor'], [3.68, -.53, 1.03])
    np.testing.assert_allclose(spec['dosing_parking_xy'], [5.39, -.52])
    assert np.isclose(spec['machine_x'], 4.03)
    np.testing.assert_allclose(offset(spec), [1.55, 0., 0.])
    np.testing.assert_allclose(shift([2.12, -.94], offset(spec)), [3.67, -.94])


def test_only_fixtures_under_the_apparatus_are_removed():
    root = scene_with_fixtures()
    assert blocking_fixtures(root, np.array([.22, 0., 0.])) == []
    assert blocking_fixtures(root, np.array([-.25, 0., 0.])) == ['paper_towel_main_group_main']
    assert blocking_fixtures(root, np.array([1.55, 0., 0.])) == [
        'toaster_main_group_main', 'knife_block_main_group_main']


def test_support_follows_the_counter_under_the_machine():
    assert counter_for(np.array([0., 0., 0.])) == 'counter_main_main_group_main'
    assert counter_for(np.array([1.55, 0., 0.])) == 'counter_right_main_group_main'


def test_transform_reduces_to_the_counter_shift():
    from cross_episode_sim.tasks.coffee.placement import to_world
    spec = dict(machine_offset=[1.55, 0.])
    np.testing.assert_allclose(to_world(spec, [2.12, -.94]), shift([2.12, -.94], offset(spec)))
    np.testing.assert_allclose(to_world(spec, [2.12, -.2643, 1.04]), [3.67, -.2643, 1.04])


def test_yawed_placement_keeps_the_layout_rigid():
    from cross_episode_sim.tasks.coffee.placement import (
        VALIDATED_DOCK, VALIDATED_MOUNT, to_validated, to_world)
    spec = dict(machine_pose=dict(dx=.5, dy=-6., dz=-.09, yaw_deg=90.))
    mount = to_world(spec, VALIDATED_MOUNT)
    dock = to_world(spec, list(VALIDATED_DOCK[:2]) + [VALIDATED_MOUNT[2]])
    np.testing.assert_allclose(mount, VALIDATED_MOUNT + [.5, -6., -.09])
    # Distances are preserved and the dock rotates with the machine.
    assert np.isclose(np.linalg.norm(dock[:2] - mount[:2]),
                      np.linalg.norm(np.array(VALIDATED_DOCK[:2]) - VALIDATED_MOUNT[:2]))
    np.testing.assert_allclose(dock[:2] - mount[:2], [.695, -.0173], atol=1e-3)
    pose = np.eye(4); pose[:3, 3] = [2.2, -.46, 1.]
    np.testing.assert_allclose(to_validated(spec, to_world(spec, pose)), pose, atol=1e-12)


def test_surface_placement_faces_out_over_the_chosen_edge():
    from cross_episode_sim.tasks.coffee.placement import (
        VALIDATED_FRONT_EDGE_Y, VALIDATED_MOUNT, surface_placement, to_world)
    pose = surface_placement([2.4, -5.71], [0., 1.], .83)
    spec = dict(machine_pose=dict(dx=pose[0], dy=pose[1], dz=pose[2], yaw_deg=np.degrees(pose[3])))
    edge = to_world(spec, [VALIDATED_MOUNT[0], VALIDATED_FRONT_EDGE_Y])
    np.testing.assert_allclose(edge, [2.4, -5.71], atol=1e-9)
    assert np.isclose(abs(np.degrees(pose[3])), 180.) and np.isclose(pose[2], -.09)


def test_walls_in_the_robot_workspace_reject_a_placement():
    import mujoco
    from cross_episode_sim.tasks.coffee.placement import workspace_wall_conflicts
    xml = ('<mujoco><worldbody>'
           '<body name="east_living_room_main" pos="10.32 -5.6 1.5"><geom type="box" size=".02 2.66 1.5"/></body>'
           '<body name="side_table_living_room_main" pos="9.75 -4.1 .4"><geom type="box" size=".32 .32 .38"/></body>'
           '</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    near = dict(machine_pose=dict(dx=7.73, dy=-3.93, dz=-.14, yaw_deg=180.))   # north edge, wall 0.45 m aside
    far = dict(machine_pose=dict(dx=7.70, dy=-3.74, dz=-.14, yaw_deg=270.))    # west edge, wall behind
    assert workspace_wall_conflicts(model, data, near) == ['east_living_room_main']
    assert workspace_wall_conflicts(model, data, far) == []
