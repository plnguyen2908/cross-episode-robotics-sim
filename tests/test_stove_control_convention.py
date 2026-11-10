import numpy as np
import pytest
from cross_episode_sim.perception.stove_control_convention import (
    burner_ordinal, select_observed_row, DocumentedStovePointer,
)


def test_manual_distinguishes_oven_controls_from_burners():
    assert burner_ordinal('front left stove burner knob') == 2
    assert burner_ordinal('front-right stove burner knob') == 6
    assert burner_ordinal('back left stove burner knob') == 3
    assert burner_ordinal('microwave start button') is None


def test_selection_uses_observed_order_not_proposal_order():
    row = np.c_[np.arange(7) * 40 + 50, np.arange(7) * 2 + 100]
    selected, _ = select_observed_row(row[[3, 6, 1, 0, 4, 2, 5]], 2, (640, 480))
    assert selected == row[2].tolist()


@pytest.mark.parametrize('points', [
    [[40*i+50, 100] for i in range(6)],
    [[50, 100]] * 7,
    [[40*i+50, 100 if i != 3 else 200] for i in range(7)],
])
def test_does_not_guess_from_partial_or_inconsistent_row(points):
    with pytest.raises(ValueError):
        select_observed_row(points, 2, (640, 480))


def test_unrelated_controls_keep_original_pointing():
    class Pointer:
        def point(self, image, target):
            return dict(points=[[12, 30]], raw=target)
    assert DocumentedStovePointer(Pointer()).point(None, 'microwave start button') == {
        'points': [[12, 30]], 'raw': 'microwave start button',
    }
