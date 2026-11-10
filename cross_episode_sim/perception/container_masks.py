"""Select an enclosing container instance when segmentation splits off its wall."""
import numpy as np


def enclosing_container_mask(part, masks, max_area_ratio=4., min_overlap=.95):
    """Use a bounded parent mask, never fill unobserved geometry or the image.

    FastSAM often returns both a basket and a smaller mask of its back wall.
    Choosing the smallest mask at the semantic point then excludes its floor.
    This operation is only for container perception; cup/bowl masks stay intact.
    Subsequent RGB confirmation and measured support extraction are required.
    """
    part = np.asarray(part, bool)
    area = int(part.sum())
    if not area:
        return part.copy()
    parents = [np.asarray(mask, bool) for mask in masks
               if np.shape(mask) == part.shape
               and area <= np.count_nonzero(mask) <= max_area_ratio * area
               and np.count_nonzero(np.asarray(mask, bool) & part) >= min_overlap * area]
    return max(parents, key=np.count_nonzero).copy() if parents else part.copy()
