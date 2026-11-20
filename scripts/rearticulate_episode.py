"""Test-time copy of a saved episode whose kitchen fixtures all keep their joints.

Datagen compiles each scene with only the task-bound cabinet/drawer articulated
(controller/house_scene.py froze the rest). A sensor-only robot cannot tell a
frozen cabinet from the working one, so for evaluation the scene is rebuilt from
its own config with the current compiler, which keeps every fixture hinge and
slide (user decision, November 16, 2026). The saved initial state is carried
over joint by joint; the added fixture joints start at their authored (closed)
positions. Every body of the saved scene must start where it was saved.

The source folder is not modified: the output links to its files and only
replaces action_recording/model.mjb and initial_state.npz.
"""
import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np


def carry_state(old, old_data, new, new_data):
    """Copy time, joints (by name), actuator controls (by name) and mocap from old to new."""
    new_data.time = old_data.time
    sizes = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3),
             mujoco.mjtJoint.mjJNT_SLIDE: (1, 1), mujoco.mjtJoint.mjJNT_HINGE: (1, 1)}
    names = {new.joint(j).name: j for j in range(new.njnt)}
    missing = []
    for j in range(old.njnt):
        name = old.joint(j).name
        if name not in names:
            missing.append(name)
            continue
        k = names[name]
        if old.jnt_type[j] != new.jnt_type[k]:
            raise RuntimeError(f'Joint type changed: {name}')
        nq, nv = sizes[int(old.jnt_type[j])]
        new_data.qpos[new.jnt_qposadr[k]:new.jnt_qposadr[k]+nq] = old_data.qpos[old.jnt_qposadr[j]:old.jnt_qposadr[j]+nq]
        new_data.qvel[new.jnt_dofadr[k]:new.jnt_dofadr[k]+nv] = old_data.qvel[old.jnt_dofadr[j]:old.jnt_dofadr[j]+nv]
    if missing:
        raise RuntimeError('Rebuilt scene lacks saved joints: '+', '.join(missing[:10]))
    actuators = {new.actuator(a).name: a for a in range(new.nu)}
    for a in range(old.nu):
        k = actuators.get(old.actuator(a).name)
        if k is None:
            raise RuntimeError('Rebuilt scene lacks actuator '+old.actuator(a).name)
        new_data.ctrl[k] = old_data.ctrl[a]
        if old.actuator_actnum[a] and new.actuator_actnum[k]:
            new_data.act[new.actuator_actadr[k]:new.actuator_actadr[k]+new.actuator_actnum[k]] = \
                old_data.act[old.actuator_actadr[a]:old.actuator_actadr[a]+old.actuator_actnum[a]]
    if old.nmocap != new.nmocap:
        raise RuntimeError('Mocap bodies differ between saved and rebuilt scene')
    new_data.mocap_pos[:] = old_data.mocap_pos
    new_data.mocap_quat[:] = old_data.mocap_quat
    if old.neq == new.neq:
        new_data.eq_active[:] = old_data.eq_active
    added = sorted(set(names) - {old.joint(j).name for j in range(old.njnt)})
    return added


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=1e-4)
    p.add_argument('--compiler', type=Path, default=Path(__file__).resolve().parents[1]/'cross_episode_sim/controller/house_scene.py',
                   help='house_scene.py whose _make_house keeps fixture joints (used when building with another checkout)')
    a = p.parse_args()
    source = a.source.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    for entry in source.iterdir():
        if entry.name != 'action_recording':
            os.symlink(entry, a.output/entry.name)
    recording = a.output/'action_recording'
    recording.mkdir()
    for entry in (source/'action_recording').iterdir():
        if entry.name not in ('model.mjb', 'initial_state.npz'):
            os.symlink(entry, recording/entry.name)

    old = mujoco.MjModel.from_binary_path(str(source/'action_recording/model.mjb'))
    old_data = mujoco.MjData(old)
    with np.load(source/'action_recording/initial_state.npz') as saved:
        kind = int(saved['state_type'])
        mujoco.mj_setState(old, old_data, saved['state'], kind)
    mujoco.mj_forward(old, old_data)

    import importlib.util
    from cross_episode_sim.controller import house_scene
    spec = importlib.util.spec_from_file_location('articulated_house_scene', a.compiler)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    if 'articulated_fixture_joints' not in a.compiler.read_text():
        raise RuntimeError('Compiler does not keep fixture joints: '+str(a.compiler))
    house_scene._make_house = patched._make_house
    try:
        from cross_episode_sim.generation.training_dataset import configure_runtime
        configure_runtime()
    except ImportError:
        pass
    from cross_episode_sim.generation.composite_scenes import scene_specification
    from cross_episode_sim.generation.environment_factory import create_environment
    config = json.loads((source/'config.json').read_text())
    build = a.output/'.rebuild'
    env = create_environment(config, build)
    try:
        env.configure(deepcopy(config['scene']), deepcopy(scene_specification(config)))
        new = env.controller.model
        new_data = mujoco.MjData(new)
        added = carry_state(old, old_data, new, new_data)
        mujoco.mj_forward(new, new_data)
        moved = []
        for b in range(1, old.nbody):
            name = old.body(b).name
            gap = float(np.linalg.norm(new_data.body(name).xpos - old_data.body(name).xpos))
            if gap > a.tolerance:
                moved.append((name, gap))
        if moved:
            raise RuntimeError('Rebuilt scene moves saved bodies: '+json.dumps(sorted(moved, key=lambda x: -x[1])[:10]))
        mujoco.mj_saveModel(new, str(recording/'model.mjb'), None)
        state = np.empty(mujoco.mj_stateSize(new, kind))
        mujoco.mj_getState(new, new_data, state, kind)
        np.savez_compressed(recording/'initial_state.npz', state=state, state_type=kind)
    finally:
        env.close()
        shutil.rmtree(build, ignore_errors=True)
    report = dict(source=str(source), saved_joints=old.njnt, rebuilt_joints=new.njnt, added_joints=added,
                  bodies_checked=old.nbody-1, tolerance_m=a.tolerance,
                  note='Test-time rebuild: all fixture articulations kept; saved initial state carried over.')
    (a.output/'rearticulation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(report, added_joints=len(added))), flush=True)


if __name__ == '__main__':
    main()
