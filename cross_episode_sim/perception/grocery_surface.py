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
    # A merged bottle/table mask can put the surrounding ring on the FLOOR.
    # Include the mask itself when finding broad horizontal support planes;
    # otherwise the entire tabletop is incorrectly retained as foreground.
    xyz=optical@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
    region=(ring|mask)&valid
    levels,counts=np.unique(np.round(height[region]/.005),return_counts=True)
    planes=[]
    for level,count in zip(levels,counts):
        if count<30:continue
        band=region&(abs(height-level*.005)<.006)
        xy=xyz[band,:2]
        if len(xy)<30:continue
        _,_,axes=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
        extent=np.ptp(xy@axes.T,axis=0)
        if extent.max()>.25 and extent.min()>.10:
            planes.append(float(level*.005))
    if not planes:return None
    support=max(planes)
    foreground = mask & valid & (height > support+.012)
    components, count = label(foreground)
    if not count:
        return None
    sizes = np.bincount(components.ravel());sizes[0] = 0
    if sizes.max() < 30:
        return None
    return components == sizes.argmax()
