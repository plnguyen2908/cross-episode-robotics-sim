"""Open-pad endpoint check against the target completed down to its support.

One depth view shows only the near faces of a resting object. The linkage
checks exclude the pads so they may touch the target, and observed points
alone cannot show a pad placed inside the hidden half. Open pads can then
straddle a corner: closing wedges the object out and the gripper closes empty.

Filling every observed target column down to the measured support is a
sensor-only estimate of that hidden volume. This is not a force-closure test;
the existing IK, swept collision and measured grip checks remain required.

Pads clear of that volume can still clip a corner: the faces they would close
on then run oblique to the pads and the object slides out. face_skew measures
that angle on the same completed volume.

A proposal can also sit off the target's middle along the closing line. One pad
then starts almost on the target and its finger linkage sweeps through the
target on the way down. centred moves such an endpoint onto the middle of the
completed volume that lies between the open pads.
"""
import itertools

import numpy as np

# Recorded closed-empty corner grasps measured 45-65 degrees and held grasps
# at most 36. A stricter 30 degree limit left a live snack pick searching for
# over ten minutes, so the limit sits between the two recorded groups.
MAX_FACE_SKEW_DEG = 40.
# A recorded snack-box pick had every top-down proposal 5-11 mm off the middle
# of a 54 mm wide box and all of them rejected by the swept linkage check.
MIN_CENTRING_SHIFT = .003
MAX_CENTRING_SHIFT = .02


def support_prism(target_surface, support_z, cell=.004):
    """Columns from the support up to the highest observed point per cell."""
    points = np.asarray(target_surface, float)
    keys, inverse = np.unique(np.floor(points[:, :2]/cell).astype(int), axis=0, return_inverse=True)
    top = np.full(len(keys), -np.inf)
    np.maximum.at(top, inverse.ravel(), points[:, 2])
    levels = np.arange(support_z, top.max()+cell, cell)
    heights = np.tile(levels, len(keys))
    keep = heights <= np.repeat(top, len(levels))
    return np.c_[np.repeat((keys+.5)*cell, len(levels), axis=0), heights][keep]


class PadEnclosure:
    def __init__(self, geometry, joints, target_surface, support_z, margin=.003):
        self.points = support_prism(target_surface, support_z)
        self.fine = support_prism(target_surface, support_z, cell=.002)
        self.margin = margin
        tool = geometry.fk(joints)[geometry.tcp_body] @ geometry.tcp_local
        self.pads = []
        for box in geometry.boxes(joints):
            if '/left_pad' not in box.name and '/right_pad' not in box.name:
                continue
            self.pads.append(((box.center-tool[:3, 3]) @ tool[:3, :3],
                              tool[:3, :3].T @ box.axes, box.half.copy()))
        if not self.pads:
            raise ValueError('Missing calibrated opposing finger pads')
        corners = np.concatenate([center+(np.array(list(itertools.product((-1, 1), repeat=3)))*half) @ axes.T
                                  for center, axes, half in self.pads])
        # Volume the open pads sweep while closing along tool y.
        self.width = corners[:, 0].min(), corners[:, 0].max()
        self.depth = corners[:, 2].min(), corners[:, 2].max()
        self.opening = np.abs(corners[:, 1]).min()

    def blocked(self, tcp):
        tcp = np.asarray(tcp)
        # Evaluate in tool coordinates once; finger configuration is unchanged.
        points = (self.points-tcp[:3, 3]) @ tcp[:3, :3]
        return any(np.any(np.all(np.abs((points-center) @ axes) <= half+self.margin, axis=1))
                   for center, axes, half in self.pads)

    def centred(self, tcp):
        """The endpoint moved along its closing line onto the target's middle, or None.

        Only when the whole completed target at the pads fits between them: a
        rim or wall grasp, with more target beyond a pad, is left where it is.
        """
        tcp = np.asarray(tcp, float)
        points = (self.fine-tcp[:3, 3]) @ tcp[:3, :3]
        slab = points[(points[:, 0] >= self.width[0]) & (points[:, 0] <= self.width[1])
                      & (points[:, 2] >= self.depth[0]) & (points[:, 2] <= self.depth[1])]
        if len(slab) < 20:
            return None
        low, high = slab[:, 1].min(), slab[:, 1].max()
        shift = (low+high)/2
        if high-low >= 2*(self.opening-self.margin) or not MIN_CENTRING_SHIFT <= abs(shift) <= MAX_CENTRING_SHIFT:
            return None
        moved = tcp.copy()
        moved[:3, 3] += shift*tcp[:3, 1]
        return moved

    def centred_variants(self, rows):
        """Each proposal preceded by its centred endpoint, when it has one."""
        for row in rows:
            moved = self.centred(row['tcp'])
            if moved is not None:
                shift = moved[:3, 3]-np.asarray(row['tcp'])[:3, 3]
                yield dict(row, tcp=moved.tolist(), contact=(np.asarray(row['contact'])+shift).tolist(),
                           closing_centre_shift_m=float(np.linalg.norm(shift)))
            yield row

    def face_skew(self, tcp, bins=8):
        """Largest angle in degrees between a facing target side and the pads.

        Returns None when too little completed target lies between the pads to
        estimate either side.
        """
        tcp = np.asarray(tcp)
        points = (self.fine-tcp[:3, 3]) @ tcp[:3, :3]
        between = points[(points[:, 0] >= self.width[0]) & (points[:, 0] <= self.width[1])
                         & (points[:, 2] >= self.depth[0]) & (points[:, 2] <= self.depth[1])
                         & (np.abs(points[:, 1]) < self.opening)]
        step = (self.width[1]-self.width[0])/bins
        column = np.clip(((between[:, 0]-self.width[0])/step).astype(int), 0, bins-1)
        used = [index for index in range(bins) if np.count_nonzero(column == index) >= 3]
        if len(used) < 3:
            return None
        across = self.width[0]+(np.array(used)+.5)*step
        sides = (np.array([between[column == index, 1].max() for index in used]),
                 np.array([between[column == index, 1].min() for index in used]))
        pairs = list(itertools.combinations(range(len(used)), 2))
        # Median pairwise slope: one partly observed column must not set the angle.
        slopes = [np.median([(side[j]-side[i])/(across[j]-across[i]) for i, j in pairs]) for side in sides]
        return float(np.degrees(np.arctan(np.abs(slopes).max())))
