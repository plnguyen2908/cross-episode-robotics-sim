"""Check measured grasp orientation and object size before scene allocation."""
from functools import lru_cache

import mujoco
import numpy as np


@lru_cache(maxsize=256)
def top_down_geometry(source, asset, key):
    from cross_episode_sim.tasks.breakfast.scene import bounds, qualified_evidence
    recording, _, grasps = qualified_evidence(dict(source=source, asset=asset, key=key))
    try:
        model = mujoco.MjModel.from_xml_path(str(recording / 'scene.xml'))
    except ValueError as exc:
        if 'Error opening file' not in str(exc):raise
        # Shared asset storage can transiently refuse a read. Retry once;
        # persistent missing resources are audited by the selection caller.
        model = mujoco.MjModel.from_xml_path(str(recording / 'scene.xml'))
    data = mujoco.MjData(model)
    with np.load(recording / 'states.npz') as archive:
        mujoco.mj_setState(model, data, archive['states'][0], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    bid = next(i for i in range(model.nbody) if model.body(i).name.startswith('test_object_')
               and model.body_jntnum[i]
               and model.jnt_type[model.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE)
    rotation = data.xmat[bid].reshape(3, 3)
    poses = [np.asarray(g[field]) for g in grasps
             for field in ('planned_object_T_tcp', 'object_T_tcp') if field in g]
    count = sum((rotation @ pose[:3, :3])[2, 2] <= -np.cos(np.pi / 6) for pose in poses)
    low, high = bounds(model, data, model.body(bid).name)
    return int(count), tuple(high-low)


def two_pot_span_limit():
    """Conservative upright footprint that keeps adjacent front pots separate."""
    import xml.etree.ElementTree as ET
    from cross_episode_sim.paths import FIXTURE_ASSETS_DIR
    root = ET.parse(FIXTURE_ASSETS_DIR / 'front_control_stove.xml')
    points = [np.fromstring(root.find(
        f".//site[@name='stove_main_group_burner_{side}_place_site']").get('pos'), sep=' ')
        for side in ('front_left', 'front_right')]
    return float(np.linalg.norm((points[1]-points[0])[:2]) - .02)
