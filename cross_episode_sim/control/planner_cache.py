"""Reuse sensor-planning solvers only for identical calibrated robot geometry."""
import hashlib,json
from pathlib import Path
from collections import OrderedDict


def sensor_arm_planner(kinematics,collision_cache,constructor=None,activation_distance=None):
    if constructor is None:
        from cross_episode_sim.robot.arm_planner import RobotArmPlanner
        constructor=RobotArmPlanner
    from cross_episode_sim.runtime.episode import current_runtime
    runtime=current_runtime()
    options={} if activation_distance is None else dict(activation_distance=activation_distance)
    if runtime is None or runtime.artifact_root is None:
        return constructor(None,{},kinematics=kinematics,collision_cache=collision_cache,**options)
    # File locations do not change robot geometry; actual URDF content does.
    config={k:v for k,v in kinematics.items() if k not in ('urdf_path','asset_root_path')}
    config['urdf']=Path(kinematics['urdf_path']).read_text()
    if activation_distance is not None:config['planner_activation_distance']=activation_distance
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=getattr(runtime,'arm_planners',None)
    if cache is None:runtime.arm_planners=cache=OrderedDict()
    previous=cache.get(key)
    if previous is not None and all(previous[0].get(k,0)>=v for k,v in collision_cache.items()):
        cache.move_to_end(key)
        arm=previous[1];arm.planner.clear_scene_cache()
        runtime.metrics['planner_cache_hits']=runtime.metrics.get('planner_cache_hits',0)+1
        return arm
    if key in cache:del cache[key]
    while len(cache)>=2:cache.popitem(last=False)
    arm=constructor(None,{},kinematics=kinematics,collision_cache=collision_cache,**options)
    cache[key]=(dict(collision_cache),arm)
    runtime.metrics['planner_initializations']=runtime.metrics.get('planner_initializations',0)+1
    return arm
