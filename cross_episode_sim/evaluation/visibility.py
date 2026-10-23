"""Private current-view visibility for oracle pickup; no labels or poses exposed."""
import numpy as np


def visible_bodies(model, renderer, data, cameras, bodies, scene_option=None):
    import mujoco
    if renderer is None:
        return set()
    roots = {model.body(name).id: name for name in bodies}
    geom_names = {}
    for gid, bid in enumerate(model.geom_bodyid):
        bid = int(bid)
        while bid and bid not in roots:
            bid = int(model.body_parentid[bid])
        if bid in roots:
            geom_names[gid] = roots[bid]
    visible = set()
    try:
        renderer.enable_segmentation_rendering()
        for camera in cameras:
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            # Some MuJoCo versions retain sparse segids after geom-group
            # filtering, while Renderer.render allocates a dense ngeom LUT.
            # Renumber render IDs only; physical objid/objtype stay unchanged.
            scene = getattr(renderer, 'scene', None)
            if scene is not None and isinstance(scene.ngeom, (int, np.integer)):
                for index, geom in enumerate(scene.geoms[:scene.ngeom]):
                    if geom.segid >= 0:
                        geom.segid = index
            frame = renderer.render()
            ids = frame[:, :, 0][frame[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
            counts = {}
            for gid, count in zip(*np.unique(ids, return_counts=True)):
                name = geom_names.get(int(gid))
                if name is not None:
                    counts[name] = counts.get(name, 0) + int(count)
            visible.update(name for name, count in counts.items() if count >= 4)
    finally:
        renderer.disable_segmentation_rendering()
    return visible
