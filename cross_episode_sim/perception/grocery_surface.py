"""Separate a portable grocery surface from support using onboard metric depth."""
import numpy as np
from scipy.ndimage import binary_dilation, label


def above_support_mask(camera, mask):
    """Reject tabletop-only segments; retain top faces of boxes and bottle sides.

    Estimate support from pixels surrounding the proposed instance, never a
    stored table height. Keep only the connected foreground above that support.
    """
    mask = np.asarray(mask, bool)
    valid = camera.valid_depth & np.isfinite(camera.depth_m) & (camera.depth_m > 0)
    yy, xx = np.indices(mask.shape)
    z = camera.depth_m
    optical = np.stack(((xx-camera.K[0, 2])*z/camera.K[0, 0],
                        (yy-camera.K[1, 2])*z/camera.K[1, 1], z), axis=-1)
    height = optical @ camera.T_base_optical[2, :3] + camera.T_base_optical[2, 3]
    ring = binary_dilation(mask, iterations=12) & ~mask & valid
    if ring.sum() < 30:
        return None
    levels, counts = np.unique(np.round(height[ring]/.005), return_counts=True)
    support = float(levels[counts.argmax()]*.005)
    foreground = mask & valid & (height > support+.012)
    components, count = label(foreground)
    if not count:
        return None
    sizes = np.bincount(components.ravel());sizes[0] = 0
    if sizes.max() < 30:
        return None
    return components == sizes.argmax()
