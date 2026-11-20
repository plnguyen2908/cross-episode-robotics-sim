import numpy as np
import pytest
from cross_episode_sim.perception.stove_control_convention import (
    burner_ordinal, select_observed_row, DocumentedStovePointer, front_burner_from_row, front_burner_side,
)


def test_two_knob_manual_maps_left_and_right_front_burners():
    assert burner_ordinal('front left stove burner knob') == 0
    assert burner_ordinal('front-right stove burner knob') == 1
    assert burner_ordinal('back left stove burner knob') is None
    assert burner_ordinal('microwave start button') is None


def test_selection_uses_observed_order_not_proposal_order():
    selected, row = select_observed_row([[400, 120], [150, 118]], 0, (640, 480))
    assert selected == [150, 118] and row == [[150, 118], [400, 120]]


@pytest.mark.parametrize('points', [
    [[150, 100]],                       # one knob only
    [[150, 100], [152, 101]],           # duplicate proposal
    [[100, 100], [250, 100], [400, 100]],  # a third control is not this stove
])
def test_does_not_guess_from_partial_or_inconsistent_row(points):
    with pytest.raises(ValueError):
        select_observed_row(points, 0, (640, 480))


def test_unrelated_controls_keep_original_pointing():
    class Pointer:
        def point(self, image, target):
            return dict(points=[[12, 30]], raw=target)
    assert DocumentedStovePointer(Pointer()).point(None, 'microwave start button') == {
        'points': [[12, 30]], 'raw': 'microwave start button',
    }


def test_front_burners_from_measured_knobs_match_authored_stove():
    # Stove002 as built (fix168 world frame): knob fronts and front burner sites.
    knobs = np.array([[3.027, -.682, .823], [3.465, -.682, .823]])
    burners = {'left': (2.953, -.475), 'right': (3.31, -.474)}
    yaw = np.radians(80.); robot = np.array([3.1, -1.4])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    base = np.c_[(knobs[:, :2] - robot) @ R, knobs[:, 2]]
    for side, true in burners.items():
        p = front_burner_from_row(base, side)
        assert np.linalg.norm(R @ p[:2] + robot - true) < .01 and abs(p[2] - .91) < .01
    with pytest.raises(ValueError):
        front_burner_from_row(base * [1, 3, 1], 'left')   # spacing not this stove
    assert [front_burner_side(t) for t in ('front left stove burner', 'front right stove burner',
                                           'front left stove burner knob', 'rear left stove burner')] == ['left', 'right', None, None]
