"""Explicit appliance-manual knowledge, applied to observed image proposals.

This module has no simulator access. Select the convention explicitly for a
compatible appliance; it is not inferred from scene object/joint identities.
"""
import re
import numpy as np


FRONT_CONTROL_SEVEN = (
    'oven temperature', 'oven time', 'front left', 'rear left',
    'center', 'rear right', 'front right',
)


def burner_ordinal(target):
    text = re.sub(r'[-_]+', ' ', target.lower())
    if 'stove' not in text or 'knob' not in text:
        return None
    text = text.replace('back ', 'rear ')
    for index, label in enumerate(FRONT_CONTROL_SEVEN[2:], 2):
        if re.search(r'\b' + label + r'\b', text):
            return index
    return None


def select_observed_row(points, ordinal, image_size):
    """Require seven distinct, approximately collinear image proposals."""
    points = np.asarray(points, float)
    if points.shape != (7, 2) or not np.isfinite(points).all():
        raise ValueError('The complete seven-knob row must be observed')
    width, height = image_size
    if np.any(points < 0) or np.any(points >= [width, height]):
        raise ValueError('Control point outside the camera image')
    row = points[np.argsort(points[:, 0])]
    if np.min(np.linalg.norm(np.diff(row, axis=0), axis=1)) < 8:
        raise ValueError('Duplicate or unresolved control proposals')
    centered = row - row.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    if np.max(abs(centered @ axes[1])) > max(5., .04 * np.ptp(row[:, 0])):
        raise ValueError('Proposals do not form a single control-panel row')
    if not 0 <= ordinal < len(row):
        raise ValueError('Control ordinal outside the documented row')
    return row[ordinal].tolist(), row.tolist()


class DocumentedStovePointer:
    """Delegate normal pointing, but resolve burner names using the manual."""
    convention = 'front_control_seven_v1'

    def __init__(self, pointer):
        self.pointer = pointer
        self.events = []

    def point(self, image, target):
        ordinal = burner_ordinal(target)
        if ordinal is None:
            return self.pointer.point(image, target)
        answer = self.pointer.point(
            image, 'all seven round knobs in the single horizontal row on the '
            'front of the stove, including the two oven controls at the left; '
            'one point at the center of each knob')
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
