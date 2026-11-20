"""Explicit appliance-manual knowledge, applied to observed image proposals.

This module has no simulator access. Select the convention explicitly for a
compatible appliance; it is not inferred from scene object/joint identities.

Two-knob stove (user decision, November 19, 2026): evaluation scenes keep only
the front-left and front-right knobs of the front-control stove
(evaluation/stove_knobs.py; every saved pot episode turns exactly those two).
Looking at the stove front, the left knob controls the front-left burner and
the right knob the front-right burner. The earlier seven-knob counting rule
(front_control_seven_v1) applied to the full panel and is retired with it.
"""
import re
import numpy as np


FRONT_CONTROL_TWO = ('front left', 'front right')
ROW_PROMPT = ('both round control knobs on the front of the stove; one point at the center of each knob')


def burner_ordinal(target):
    """0 (left knob) / 1 (right knob) for a named front burner knob, else None."""
    text = re.sub(r'[-_]+', ' ', target.lower())
    if 'stove' not in text or 'knob' not in text:
        return None
    for index, label in enumerate(FRONT_CONTROL_TWO):
        if re.search(r'\b' + label + r'\b', text):
            return index
    return None


def select_observed_row(points, ordinal, image_size, count=len(FRONT_CONTROL_TWO)):
    """Require ``count`` distinct image proposals; return (chosen, row ordered left to right)."""
    points = np.asarray(points, float)
    if points.shape != (count, 2) or not np.isfinite(points).all():
        raise ValueError(f'Both stove knobs must be observed (got {len(points)} points)')
    width, height = image_size
    if np.any(points < 0) or np.any(points >= [width, height]):
        raise ValueError('Control point outside the camera image')
    row = points[np.argsort(points[:, 0])]
    if np.min(np.linalg.norm(np.diff(row, axis=0), axis=1)) < 8:
        raise ValueError('Duplicate or unresolved control proposals')
    if not 0 <= ordinal < len(row):
        raise ValueError('Control ordinal outside the documented row')
    return row[ordinal].tolist(), row.tolist()


class DocumentedStovePointer:
    """Delegate normal pointing, but resolve burner-knob names with the two-knob manual."""
    convention = 'front_control_two_v1'

    def __init__(self, pointer):
        self.pointer = pointer
        self.events = []

    def point(self, image, target):
        ordinal = burner_ordinal(target)
        if ordinal is None:
            return self.pointer.point(image, target)
        answer = self.pointer.point(image, ROW_PROMPT)
        event = dict(convention=self.convention, target=target,
                     one_based_ordinal=ordinal + 1, proposals=answer.get('points', []))
        try:
            selected, row = select_observed_row(answer.get('points', []), ordinal, image.size)
            event.update(accepted=True, ordered_row=row, selected=selected)
        except ValueError as exc:
            event.update(accepted=False, reason=str(exc))
            self.events.append(event)
            return dict(points=[], raw=event)
        self.events.append(event)
        return dict(points=[selected], raw=event)


# Burner layout of the same stove (user decision, November 19, 2026: "add
# burner layout to the stove manual"). Measured on the authored Stove002 in
# metres, relative to each knob's observed front face: the burner sits this far
# behind the knob, and this far along the knob row (left to right is
# positive). The pointing model's burner spot was 2, 12, 30 and 15 cm off in
# fix167-fix170 (checker tolerance 9 cm).
FRONT_BURNER_BEHIND_KNOB = .207
FRONT_BURNER_ALONG_ROW = {'left': -.074, 'right': -.155}
FRONT_BURNER_ABOVE_KNOBS = .087
KNOB_SPACING_RANGE = (.30, .60)   # authored centre-to-centre distance 0.438 m


def front_burner_side(target):
    """'left' / 'right' for a named front burner, else None (rear/centre zones are not in this layout)."""
    text = re.sub(r'[-_]+', ' ', str(target).lower())
    if 'burner' not in text or 'stove' not in text or 'knob' in text or 'rear' in text or 'back' in text:
        return None
    if 'front' not in text:
        return None
    return 'right' if re.search(r'\bright\b', text) else 'left' if re.search(r'\bleft\b', text) else None


def front_burner_from_row(knobs_base, side):
    """Burner centre (robot base frame) from the two knob points measured in depth, ordered left to right.

    Left/right are as seen facing the stove, which is the image order. The
    outward direction is horizontal, perpendicular to the knob row and toward
    the robot (base origin).
    """
    knobs = np.asarray(knobs_base, float)
    if knobs.shape != (2, 3) or not np.isfinite(knobs).all():
        raise ValueError('Two measured knob points are required')
    along = knobs[1, :2] - knobs[0, :2]
    spacing = float(np.linalg.norm(along))
    if not KNOB_SPACING_RANGE[0] <= spacing <= KNOB_SPACING_RANGE[1]:
        raise ValueError(f'Measured knobs are {spacing:.2f} m apart, not this stove')
    if abs(knobs[1, 2] - knobs[0, 2]) > .03:
        raise ValueError('Measured knobs are not at one height')
    along /= spacing
    outward = np.array([-along[1], along[0]])
    if outward @ (-knobs.mean(axis=0)[:2]) < 0:
        outward = -outward
    knob = knobs[0 if side == 'left' else 1, :2]
    centre = knob + FRONT_BURNER_ALONG_ROW[side] * along - FRONT_BURNER_BEHIND_KNOB * outward
    return np.r_[centre, float(knobs[:, 2].mean()) + FRONT_BURNER_ABOVE_KNOBS]


def measured_knob_row(pointer, camera, to_point):
    """Ask the pointer for both knobs in one camera; return 2 base-frame points, left to right, or raise."""
    from PIL import Image
    answer = pointer.point(Image.fromarray(camera.rgb), ROW_PROMPT)
    _, row = select_observed_row(answer.get('points', []), 0, (camera.rgb.shape[1], camera.rgb.shape[0]))
    points = [to_point(p) for p in row]
    if any(p is None for p in points):
        raise ValueError('A stove knob has no valid depth')
    return np.asarray(points, float), row
