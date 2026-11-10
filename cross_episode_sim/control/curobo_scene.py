"""Bulk-load observed boxes using cuRobo's own preallocated scene storage."""
from curobo._src.geom.types import SceneCfg


def load_observed_cuboids(planner, boxes):
    """Equivalent to update_world(SceneCfg(cuboid=boxes)), without per-box sync.

    Keep the same obstacles, poses, sizes, collision cache and graph reset.
    Native load_batch writes into the existing buffers used by CUDA graphs.
    """
    names=[box.name for box in boxes]
    if len(set(names))!=len(names):raise ValueError('Duplicate observed obstacle names')
    checker=planner.scene_collision_checker
    data=checker.data
    if data.cuboids is None or len(boxes)>data.cuboids.max_n:
        raise ValueError('Observed cuboids exceed allocated collision cache')
    scene=SceneCfg(cuboid=boxes)
    data.clear(0)
    data.cuboids.load_batch(boxes,0)
    data.scene_model=scene
    checker.scene_model=scene
    if planner.graph_planner is not None:planner.graph_planner.reset_buffer()
