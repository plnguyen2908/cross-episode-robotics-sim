"""Cheap proposal ranking using calibrated gripper links and observed points.

This is not a motion validator. It only ranks likely blocked endpoints later;
the existing IK, full swept collision and physical grasp checks remain required.
"""
import numpy as np


class GraspLinkageRanker:
    def __init__(self, geometry, joints, target_surface, margin=.003):
        self.points = np.asarray(target_surface)
        self.margin = margin
        tool = geometry.fk(joints)[geometry.tcp_body] @ geometry.tcp_local
        self.links = []
        for box in geometry.boxes(joints):
            if '/gripper/' not in box.name or '/left_pad' in box.name or '/right_pad' in box.name:
                continue
            self.links.append(((box.center-tool[:3, 3]) @ tool[:3, :3],
                               tool[:3, :3].T @ box.axes, box.half.copy()))

    def blocked(self, tcp):
        tcp = np.asarray(tcp)
        # Evaluate in tool coordinates once; finger configuration is unchanged.
        points = (self.points-tcp[:3, 3]) @ tcp[:3, :3]
        return any(np.any(np.all(np.abs((points-center) @ axes) <= half+self.margin, axis=1))
                   for center, axes, half in self.links)

    def rank(self, rows):
        """Stable partition: retain every proposal and within-group ordering."""
        flags = [self.blocked(row['tcp']) for row in rows]
        return [row for blocked in (False, True) for row, flag in zip(rows, flags) if flag == blocked]

    def iter_ranked(self, rows, batch_size=256):
        """Rank bounded confidence windows lazily; don't inspect 100k poses first."""
        from itertools import islice
        source = iter(rows)
        while batch := list(islice(source, batch_size)):
            yield from self.rank(batch)
