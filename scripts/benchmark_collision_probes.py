"""Compare full and position-only forward passes on a compiled episode model.

The model must match the checkpoint's controller model (including frozen joints),
not just the authored scene XML. This benchmarks geometric probes, not mj_step or
end-to-end episode generation. No input files are modified.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from cross_episode_sim.control.collision_probe import forward_collision


def benchmark(model, checkpoint, samples):
    full, probe = mujoco.MjData(model), mujoco.MjData(model)
    for data in (full, probe):
        for field in ('qpos', 'qvel', 'ctrl', 'eq_active'):
            getattr(data, field)[:] = checkpoint[field]
        data.time = float(checkpoint['time'])
    initial = full.qpos.copy()
    rng = np.random.default_rng(7)
    elapsed = dict(full_forward=0., collision_forward=0.)
    contacts = []
    for index in range(samples):
        # Nearby configurations exercise changing contacts as well as the saved
        # state; integratePos handles both scalar joints and free quaternions.
        qpos = initial.copy()
        if index:
            mujoco.mj_integratePos(model, qpos, rng.normal(size=model.nv), .002)
        for data, name, update in ((full, 'full_forward', mujoco.mj_forward),
                                   (probe, 'collision_forward', forward_collision)):
            data.qpos[:] = qpos
            start = perf_counter()
            update(model, data)
            elapsed[name] += perf_counter() - start
        for field in ('xpos', 'xmat', 'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat'):
            np.testing.assert_array_equal(getattr(full, field), getattr(probe, field))
        for field in ('geom1', 'geom2', 'dist', 'pos', 'frame', 'includemargin'):
            np.testing.assert_array_equal(getattr(full.contact, field), getattr(probe.contact, field))
        np.testing.assert_array_equal(probe.qpos, qpos)
        assert probe.time == float(checkpoint['time'])
        contacts.append(full.ncon)
    return dict(samples=samples, seconds=elapsed,
                speedup=elapsed['full_forward']/elapsed['collision_forward'],
                contact_count_range=[min(contacts), max(contacts)],
                geometry_and_contacts_exact=True,
                scope='scratch collision probes; not live physics or whole episodes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Compiled controller .mjb model')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    model = mujoco.MjModel.from_binary_path(str(args.model))
    with np.load(args.checkpoint) as checkpoint:
        result = benchmark(model, checkpoint, args.samples)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text+'\n')


if __name__ == '__main__':
    main()
