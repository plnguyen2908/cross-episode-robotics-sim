"""Load a saved episode for evaluation with every kitchen fixture articulated.

Older datagen compiled each scene with only the task-bound cabinet/drawer
articulated (the other fixture joints were deleted from action_recording/model.mjb).
A policy may choose any cabinet, drawer or appliance, and a robot cannot tell a
frozen door from a working one, so evaluation must use the full kitchen
(user decision, November 18, 2026).

The episode's scene.xml still authors every joint. When the saved model lacks
some of them, that scene.xml is compiled again with the current house compiler
(which keeps all fixture hinges and slides), the saved initial
state is carried over joint by joint, and every saved body must start within
``tolerance`` metres of where it was saved. The added fixture joints start at
their authored (closed) positions. Nothing is written into the source folder,
and no rebuilt model is kept on disk.

Two-knob stove (user decision, November 19, 2026): the rebuild also deletes
the five stove knobs no pot task uses (evaluation/stove_knobs.py), so every
evaluated method sees the same stove. Only those bodies and their geoms may be
absent from the rebuilt model; every other saved element must match.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

_JOINT_SIZES = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3),
                mujoco.mjtJoint.mjJNT_SLIDE: (1, 1), mujoco.mjtJoint.mjJNT_HINGE: (1, 1)}


def _removed_bodies(old, new):
    """Saved bodies the rebuild deliberately deleted (the unused stove knobs) and absent from ``new``."""
    from cross_episode_sim.evaluation.stove_knobs import removed_knob_bodies
    present = {new.body(b).name for b in range(new.nbody)}
    return {name for name in removed_knob_bodies()
            if mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_BODY, name) >= 0 and name not in present}


def carry_state(old, old_data, new, new_data):
    """Copy time, joints and actuators (by name), activations and mocap from old to new.

    Returns the names of joints that exist only in ``new``.
    """
    new_data.time = old_data.time
    names = {new.joint(j).name: j for j in range(new.njnt)}
    removed = _removed_bodies(old, new)
    missing = []
    for j in range(old.njnt):
        name = old.joint(j).name
        if name not in names:
            if old.body(int(old.jnt_bodyid[j])).name not in removed:
                missing.append(name)
            continue
        k = names[name]
        if old.jnt_type[j] != new.jnt_type[k]:
            raise RuntimeError(f'Joint type changed: {name}')
        nq, nv = _JOINT_SIZES[int(old.jnt_type[j])]
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
    return sorted(set(names) - {old.joint(j).name for j in range(old.njnt)})


_PREFIX = {'body': 'body_', 'geom': 'geom_', 'jnt': 'jnt_', 'dof': 'dof_', 'actuator': 'actuator_',
           'site': 'site_', 'tendon': 'tendon_', 'eq': 'eq_', 'cam': 'cam_', 'light': 'light_'}
# Integer fields that are physical settings rather than array addresses or ids.
_INT_SETTINGS = {'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_priority', 'jnt_limited',
                 'actuator_ctrllimited', 'actuator_forcelimited', 'actuator_actlimited', 'actuator_dyntype',
                 'actuator_gaintype', 'actuator_biastype', 'tendon_limited', 'body_gravcomp'}


def _element_maps(old, new):
    """Old index -> new index for every shared element, by name (geoms by body + ordinal)."""
    def by_name(count, getter):
        return {getter(i).name: i for i in range(count) if getter(i).name}
    maps = {}
    for kind, count_old, count_new, getter_old, getter_new in (
            ('body', old.nbody, new.nbody, old.body, new.body),
            ('jnt', old.njnt, new.njnt, old.joint, new.joint),
            ('actuator', old.nu, new.nu, old.actuator, new.actuator),
            ('site', old.nsite, new.nsite, old.site, new.site),
            ('tendon', old.ntendon, new.ntendon, old.tendon, new.tendon),
            ('eq', old.neq, new.neq, old.eq, new.eq),
            ('cam', old.ncam, new.ncam, old.camera, new.camera),
            ('light', old.nlight, new.nlight, old.light, new.light)):
        target = by_name(count_new, getter_new)
        maps[kind] = {i: target[getter_old(i).name] for i in range(count_old)
                      if getter_old(i).name in target}
    def geom_keys(model):
        seen, keys = {}, {}
        for g in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[g])).name
            k = seen.get(body, 0); seen[body] = k+1
            keys[(body, k)] = g
        return keys
    new_geoms = geom_keys(new)
    maps['geom'] = {g: new_geoms[k] for k, g in geom_keys(old).items() if k in new_geoms}
    dof = {}
    for j, k in maps['jnt'].items():
        n = int(old.jnt_dofadr[j+1] if j+1 < old.njnt else old.nv) - int(old.jnt_dofadr[j])
        for d in range(n):
            dof[int(old.jnt_dofadr[j])+d] = int(new.jnt_dofadr[k])+d
    maps['dof'] = dof
    return maps


def _physical_fields(model, kind):
    count = {'body': model.nbody, 'geom': model.ngeom, 'jnt': model.njnt, 'dof': model.nv,
             'actuator': model.nu, 'site': model.nsite, 'tendon': model.ntendon, 'eq': model.neq,
             'cam': model.ncam, 'light': model.nlight}[kind]
    for name in dir(model):
        if not name.startswith(_PREFIX[kind]):
            continue
        value = getattr(model, name)
        if not isinstance(value, np.ndarray) or not value.shape or value.shape[0] != count:
            continue
        if np.issubdtype(value.dtype, np.floating) or name in _INT_SETTINGS:
            yield name


_VISUAL_GROUPS = ('global_', 'quality', 'headlight', 'map', 'scale', 'rgba')


def copy_saved_visual(old, new):
    """Copy the saved model's global visual settings (model.vis) into the rebuild.

    Datagen's controller sets them at run time (e.g. headlight ambient 0.5),
    and a fresh compile of scene.xml starts from MuJoCo defaults (ambient 0.1):
    rebuilt scenes rendered darker than the saved episodes and their history
    (found 2026-11-19 comparing onboard camera images). Materials, textures,
    lights and geom colours already match; only these globals differed.
    """
    import numpy as np
    for group in _VISUAL_GROUPS:
        src, dst = getattr(old.vis, group), getattr(new.vis, group)
        for field in dir(src):
            if field.startswith('_'):
                continue
            value = getattr(src, field)
            if callable(value):
                continue
            current = getattr(dst, field)
            if isinstance(current, np.ndarray):
                current[...] = value
            else:
                try:
                    setattr(dst, field, value)
                except (AttributeError, TypeError):
                    pass


def copy_saved_physics(old, new):
    """Give every shared element of ``new`` the exact physical parameters it had in ``old``.

    Controllers retune the robot at runtime (base gains, gripper force range) and
    the saved model holds those values; a fresh compile starts from defaults.
    Copying by element name keeps the evaluated robot and objects identical to
    the episode, so the only difference is the added fixture joints.
    """
    maps = _element_maps(old, new)
    for kind, mapping in maps.items():
        if not mapping:
            continue
        src = np.fromiter(mapping.keys(), int); dst = np.fromiter(mapping.values(), int)
        for name in _physical_fields(old, kind):
            getattr(new, name)[dst] = getattr(old, name)[src]
    for field in ('timestep', 'gravity', 'wind', 'magnetic', 'density', 'viscosity', 'impratio', 'tolerance',
                  'iterations', 'ls_iterations', 'integrator', 'cone', 'solver', 'noslip_iterations'):
        setattr(new.opt, field, getattr(old.opt, field))
    copy_saved_visual(old, new)
    mismatch = []
    for kind, mapping in maps.items():
        if not mapping:
            continue
        src = np.fromiter(mapping.keys(), int); dst = np.fromiter(mapping.values(), int)
        for name in _physical_fields(old, kind):
            if not np.array_equal(getattr(new, name)[dst], getattr(old, name)[src], equal_nan=True):
                mismatch.append(name)
    removed = _removed_bodies(old, new)
    allowed = dict(body=len(removed), actuator=0,
                   geom=sum(1 for g in range(old.ngeom) if old.body(int(old.geom_bodyid[g])).name in removed),
                   jnt=sum(1 for j in range(old.njnt) if old.body(int(old.jnt_bodyid[j])).name in removed))
    unmapped = {k: getattr(old, n) - len(maps[k]) - allowed[k] for k, n in (('body', 'nbody'), ('geom', 'ngeom'), ('jnt', 'njnt'),
                                                             ('actuator', 'nu'))}
    if mismatch or any(unmapped.values()):
        raise RuntimeError(f'Rebuilt scene differs from the saved one: fields {mismatch[:6]}, unmatched {unmapped}')
    return maps


def _saved(source):
    model = mujoco.MjModel.from_binary_path(str(source/'action_recording/model.mjb'))
    data = mujoco.MjData(model)
    with np.load(source/'action_recording/initial_state.npz') as saved:
        mujoco.mj_setState(model, data, saved['state'], int(saved['state_type']))
    mujoco.mj_forward(model, data)
    return model, data


def _rebuild(source, old):
    """Compile the episode's own saved scene.xml with the current compiler.

    Uses only files saved in the episode folder (no generator code, no house
    sampling): the same house compiler datagen used, which now keeps every
    fixture hinge/slide. Loose objects that were free in the saved model stay
    free; everything else not articulated stays fixed as authored.
    """
    from cross_episode_sim.controller.house_scene import make_house
    manifest = json.loads((source/'task_manifest.json').read_text())
    free = [old.body(int(old.jnt_bodyid[j])).name for j in range(old.njnt)
            if old.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and not old.joint(j).name.startswith('robot')]
    dynamic = list(dict.fromkeys(free + list(manifest.get('dynamic_fixtures', []))))
    spawn = manifest['robot_spawn']
    from cross_episode_sim.evaluation.stove_knobs import remove_extra_stove_knobs
    model, _, _ = make_house(source/'scene.xml', dynamic, spawn[:2], spawn[2] if len(spawn) > 2 else 0.,
                             manifest.get('robot', 'franka_tidybot'), spec_edit=remove_extra_stove_knobs)
    return model


def load_saved_scene(source, tolerance=1e-4):
    """Return (model, data, report) for a saved episode as every evaluated method sees it."""
    from cross_episode_sim.evaluation.stove_knobs import paint_kept_knobs, PAINTED_TASKS
    source = Path(source)
    model, data, report = _load_saved_scene(source, tolerance)
    task = json.loads((source/'config.json').read_text()).get('task') if (source/'config.json').exists() else None
    report['painted_knobs'] = paint_kept_knobs(model) if task in PAINTED_TASKS else []
    return model, data, report


# The saved mobile base is a 50 kN/m position servo with no force limit on a 19 kg base:
# commanded 10 cm into a counter it pushes ~5 kN. Frozen fixtures absorbed that; articulated
# cabinet doors get crushed into their cabinets and the simulation diverges. Normal driving
# (0.5 m/s, 0.8 rad/s limits) needs tens of newtons; these limits are ~10x that.
BASE_FORCE_LIMIT_N = 200.
BASE_TORQUE_LIMIT_NM = 50.


def limit_base_force(model):
    """Cap the planar base actuators' force/torque; returns {actuator: limit} applied.

    Call AFTER the robot controller is constructed: MolmoSpacesController rewrites actuator
    gains and force limits at start-up. Every evaluated method must apply the same cap.
    """
    applied = {}
    for name, limit in (('robot_0/base_x_act', BASE_FORCE_LIMIT_N), ('robot_0/base_y_act', BASE_FORCE_LIMIT_N),
                        ('robot_0/base_theta_act', BASE_TORQUE_LIMIT_NM)):
        try: a = model.actuator(name).id
        except KeyError: continue
        model.actuator_forcelimited[a] = 1; model.actuator_forcerange[a] = [-limit, limit]; applied[name] = limit
    return applied


def _load_saved_scene(source, tolerance=1e-4):
    """Return (model, data, report) for a saved episode with all fixture joints present."""
    from cross_episode_sim.evaluation.fixture_articulation import authored_fixture_joints
    source = Path(source)
    old, old_data = _saved(source)
    present = {old.joint(j).name for j in range(old.njnt)}
    from cross_episode_sim.evaluation.stove_knobs import has_extra_stove_knobs, removed_knob_bodies
    removed_names = set(removed_knob_bodies())
    frozen = sorted(name for name in authored_fixture_joints(source/'scene.xml') - present
                    if not any(name.startswith(body+'_') for body in removed_names))
    if not frozen and not has_extra_stove_knobs(old):
        return old, old_data, dict(rebuilt=False, saved_joints=old.njnt, added_joints=[], removed_stove_knobs=[])
    new = _rebuild(source, old)
    copy_saved_physics(old, new)
    new_data = mujoco.MjData(new)
    added = carry_state(old, old_data, new, new_data)
    mujoco.mj_forward(new, new_data)
    still = sorted(set(frozen) - {new.joint(j).name for j in range(new.njnt)})
    if still:
        raise RuntimeError(f'Rebuilt scene still lacks {len(still)} authored fixture joints, e.g. {still[:4]}')
    door_armature = stabilize_door_hinges(new, new_data, added)
    cabinet_contacts = settle_cabinet_fronts(new, new_data, added)
    removed = _removed_bodies(old, new)
    gaps =[(float(np.linalg.norm(new_data.body(old.body(b).name).xpos - old_data.body(old.body(b).name).xpos)),
             old.body(b).name) for b in range(1, old.nbody) if old.body(b).name not in removed]
    worst = max(gaps)
    if worst[0] > tolerance:
        raise RuntimeError(f'Rebuilt scene moves saved body {worst[1]} by {worst[0]*1000:.3f} mm')
    return new, new_data, dict(rebuilt=True, saved_joints=old.njnt, rebuilt_joints=new.njnt,
                               added_joints=added, max_body_gap_m=worst[0], tolerance_m=tolerance,
                               removed_stove_knobs=sorted(removed), door_hinge_armature=door_armature,
                               cabinet_fronts=cabinet_contacts)


# Cabinet doors in the authored assets weigh 55-140 g with ~0.003 kg m^2 about their
# hinge. Frozen, that never mattered; articulated, a robot contact on such a door
# gives accelerations MuJoCo cannot integrate at 2 ms and the episode diverges.
# A real 0.3-0.5 m wooden cabinet door (~2-3 kg) has ~0.1 kg m^2 about its hinge.
DOOR_HINGE_MIN_INERTIA = 0.1


def stabilize_door_hinges(model, data, joints, minimum=DOOR_HINGE_MIN_INERTIA):
    """Raise each added cabinet door hinge's effective inertia to `minimum` (armature).

    Only joints named *doorhinge* (cabinet doors). Appliance doors, drawers, knobs
    and buttons keep their authored dynamics. Joint ranges, damping, collisions and
    the start pose are unchanged. Returns {joint: added armature}.
    """
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv)); mujoco.mj_fullM(model, full, data.qM)
    added = {}
    for name in joints:
        j = model.joint(name)
        if 'doorhinge' not in name or int(j.type[0]) != mujoco.mjtJoint.mjJNT_HINGE: continue
        dof = int(model.jnt_dofadr[j.id])
        need = minimum - float(full[dof, dof])
        if need > 0:
            model.dof_armature[dof] += need; added[name] = round(need, 4)
    if added: mujoco.mj_forward(model, data)
    return added


# Saved v4 scenes had every cabinet front frozen, so MuJoCo never computed a door's contact
# with its own cabinet. Articulated, a closed door is held by its hinge limit AND contacts
# with its own cabinet AND the robot pressing on it; that over-constrained stack diverges.
# The soft default limit (solref 0.02) also let the robot push doors/drawers past closed into
# the neighbouring appliance. Validated on 6 crashed + 3 successful GPT episodes (full replays).
FRONT_LIMIT_SOLREF = (0.004, 1.0)          # 2x the 2 ms timestep: stiffest stable time constant
FRONT_LIMIT_SOLIMP = (0.99, 0.999, 0.001)


def settle_cabinet_fronts(model, data, joints):
    """For added cabinet door hinges (*doorhinge) and cabinet drawer slides (stack*/cab* *slidejoint):
    stiff joint limits; for doors also no collision with their OWN cabinet body (contype bits:
    door 4/1, own cabinet 2/1; every other geom is 1/1 so both still collide with the robot,
    objects, walls and other fixtures). Appliance doors, knobs and buttons are unchanged.
    """
    def subtree(root):
        out = set()
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == root: out.add(b); break
                p = model.body_parentid[p]
        return out
    report = dict(stiff_limits=[], doors_without_own_cabinet_contact=[])
    for name in joints:
        j = model.joint(name).id
        door = 'doorhinge' in name
        drawer = 'slidejoint' in name and name.startswith(('stack', 'cab'))
        if not (door or drawer) or not model.jnt_limited[j]: continue
        model.jnt_solref[j] = FRONT_LIMIT_SOLREF; model.jnt_solimp[j, :3] = FRONT_LIMIT_SOLIMP
        report['stiff_limits'].append(name)
        if not door: continue
        front = subtree(model.jnt_bodyid[j]); cabinet = subtree(model.body_parentid[model.jnt_bodyid[j]]) - front
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]): continue
            if model.geom_bodyid[g] in front: model.geom_contype[g], model.geom_conaffinity[g] = 4, 1
            elif model.geom_bodyid[g] in cabinet and model.geom_contype[g] != 4:
                model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
        report['doors_without_own_cabinet_contact'].append(name)
    if report['stiff_limits']: mujoco.mj_forward(model, data)
    return report
