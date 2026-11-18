from pathlib import Path
import mujoco
import pytest
from cross_episode_sim.evaluation import fixture_articulation as guard

XML = '''<mujoco><worldbody>
<body name="robot_0/base"><joint name="robot_0/base_x" type="slide"/><geom size=".1"/></body>
<body name="cab"><joint name="cab_doorhinge" type="hinge"/><geom size=".1"/>
<body name="drawer"><joint name="cab_slidejoint" type="slide"/><geom size=".1"/></body></body>
<body name="room_door_0"><joint name="room_door_0_hinge_left"/><geom size=".1"/></body>
<body name="obj"><freejoint name="obj_free"/><geom size=".1"/></body>
</worldbody></mujoco>'''


def episode(tmp_path, keep):
    (tmp_path/'action_recording').mkdir()
    (tmp_path/'scene.xml').write_text(XML)
    spec = mujoco.MjSpec.from_string(XML)
    for body in spec.bodies:
        for joint in list(body.joints):
            if joint.name not in keep and joint.type != mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(joint)
    mujoco.mj_saveModel(spec.compile(), str(tmp_path/'action_recording/model.mjb'), None)
    return tmp_path


def test_authored_fixture_joints_exclude_robot_room_doors_and_free_joints(tmp_path):
    assert guard.authored_fixture_joints(episode(tmp_path, set())/'scene.xml') == {'cab_doorhinge', 'cab_slidejoint'}


def test_frozen_fixture_is_refused(tmp_path):
    source = episode(tmp_path, {'cab_slidejoint'})
    assert guard.missing_fixture_joints(source) == ['cab_doorhinge']
    with pytest.raises(RuntimeError, match='rearticulate_episode'):
        guard.require_articulated(source)


def test_fully_articulated_scene_passes(tmp_path):
    source = episode(tmp_path, {'cab_doorhinge', 'cab_slidejoint', 'room_door_0_hinge_left', 'robot_0/base_x'})
    assert guard.require_articulated(source) == dict(fixture_joints_checked=True)
