from types import SimpleNamespace

import numpy as np

from cross_episode_sim.control.grasp_pad_enclosure import MAX_FACE_SKEW_DEG, PadEnclosure, support_prism


class Gripper:
    """Open opposing pads 86.6 mm apart along tool y, as on the calibrated robot."""
    tcp_body = 'tool'
    tcp_local = np.eye(4)
    joint_names = ()

    def fk(self, joints):
        return {'tool': np.eye(4)}

    def boxes(self, joints):
        pad = lambda name, y: SimpleNamespace(name=name, center=np.array([0, y, -.015]),
                                              axes=np.eye(3), half=np.array([.011, .002, .019]))
        palm = SimpleNamespace(name='robot_0/gripper/base', center=np.array([0, 0, -.11]),
                               axes=np.eye(3), half=np.array([.03, .1, .045]))
        return [pad('robot_0/gripper/left_pad', .0453), pad('robot_0/gripper/right_pad', -.0453), palm]


def visible_box_faces(half=(.027, .0378, .0432), step=.003):
    """Front (-x) and top faces only: what one camera sees of an upright box."""
    x, y, z = half
    ys, zs, xs = (np.arange(-v, v+step/2, step) for v in (y, z, x))
    front = np.array([(-x, b, c+z) for b in ys for c in zs])
    top = np.array([(a, b, 2*z) for a in xs for b in ys])
    return np.r_[front, top]


def top_down(yaw, height):
    c, s = np.cos(yaw), np.sin(yaw)
    tcp = np.eye(4)
    # Tool z points down; tool y is the closing direction.
    tcp[:3, :3] = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]])
    tcp[2, 3] = height
    return tcp


def test_support_prism_fills_hidden_volume_below_observed_top():
    prism = support_prism(visible_box_faces(), 0.)
    hidden = np.abs(prism[:, :2]-np.array([.02, .03])).max(axis=1) < .004
    assert hidden.any() and prism[hidden, 2].min() < .005 and prism[hidden, 2].max() > .08


def test_face_aligned_grasp_is_clear_and_corner_grasp_is_rejected():
    surface = visible_box_faces()
    enclosure = PadEnclosure(Gripper(), (), surface, 0.)
    # Closing across the 54 mm width: both pads stand outside the faces.
    assert not enclosure.blocked(top_down(np.pi/2, .0764))
    # Closing along the 75.6 mm depth still fits in the open gripper.
    assert not enclosure.blocked(top_down(0., .0764))
    # Closing 30 degrees off the long axis puts a pad inside the hidden corner.
    assert enclosure.blocked(top_down(np.deg2rad(30), .0764))


def test_corner_clip_is_skewed_while_face_aligned_grasp_is_not():
    enclosure = PadEnclosure(Gripper(), (), visible_box_faces(), 0.)
    assert enclosure.face_skew(top_down(np.pi/2, .0764)) < 10
    # Pads clear of the box on both sides of its +x,+y corner, closing diagonally.
    corner = top_down(np.deg2rad(45), .0764)
    corner[:2, 3] = .027, .0378
    assert not enclosure.blocked(corner)
    assert enclosure.face_skew(corner) > MAX_FACE_SKEW_DEG
    # Nothing between the pads: no estimate rather than a false angle.
    away = top_down(0., .0764)
    away[0, 3] = .5
    assert enclosure.face_skew(away) is None


def test_pad_free_gripper_is_rejected_instead_of_passing_vacuously():
    gripper = Gripper()
    gripper.boxes = lambda joints: []
    try:
        PadEnclosure(gripper, (), visible_box_faces(), 0.)
    except ValueError:
        return
    raise AssertionError('missing pads must not pass')


def test_off_middle_endpoint_is_moved_onto_the_target_middle():
    # Closing along world y across the 54 mm side of the box seen from its 76 mm side.
    surface = visible_box_faces(half=(.0378, .027, .0432))
    enclosure = PadEnclosure(Gripper(), (), surface, 0.)
    tcp = top_down(0., .08)
    tcp[1, 3] = .010
    moved = enclosure.centred(tcp)
    assert moved is not None and abs(moved[1, 3]) < .003 and np.allclose(moved[[0, 2], 3], tcp[[0, 2], 3])
    assert np.allclose(moved[:3, :3], tcp[:3, :3])
    rows = list(enclosure.centred_variants([dict(tcp=tcp.tolist(), contact=[0, .010, .08], confidence=.8)]))
    assert len(rows) == 2 and 'closing_centre_shift_m' in rows[0] and rows[1]['tcp'] == tcp.tolist()
    assert abs(rows[0]['contact'][1]) < .003


def test_middle_endpoint_and_wall_grasp_are_left_alone():
    surface = visible_box_faces(half=(.0378, .027, .0432))
    enclosure = PadEnclosure(Gripper(), (), surface, 0.)
    assert enclosure.centred(top_down(0., .08)) is None
    # Closing across the 200 mm side: more target lies beyond both pads.
    wide = PadEnclosure(Gripper(), (), visible_box_faces(half=(.0378, .10, .0432)), 0.)
    tcp = top_down(0., .08)
    tcp[1, 3] = .010
    assert wide.centred(tcp) is None
