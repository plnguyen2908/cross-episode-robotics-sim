"""Preserve continuous base yaw when recordings re-label angles modulo 2*pi."""
import numpy as np


def continuous_yaw_commands(qpos, commands, *, yaw_qpos, yaw_actuator):
    """Return command/reference copies and wrap events; never modify live state.

    Only exact full-turn discontinuities in recorded *state* authorize a branch
    change. Intentional turns in command targets are otherwise retained.
    """
    qpos = np.asarray(qpos)
    commands = np.asarray(commands)
    delta = np.diff(qpos[:, yaw_qpos])
    indices = np.flatnonzero(np.abs(delta) > np.pi)
    turns = np.rint(delta[indices] / (2 * np.pi))
    if not np.allclose(delta[indices], turns * (2 * np.pi), atol=1e-4, rtol=0):
        raise ValueError('Recorded yaw contains a non-periodic state discontinuity')
    changes = np.zeros(len(qpos))
    changes[indices + 1] = -turns * (2 * np.pi)
    offsets = np.cumsum(changes)
    continuous_commands = commands.copy()
    continuous_reference = qpos.copy()
    continuous_commands[:, yaw_actuator] += offsets
    continuous_reference[:, yaw_qpos] += offsets
    events = [dict(step=int(i + 1), recorded_jump_radians=float(delta[i]),
                   command_offset_radians=float(offsets[i + 1])) for i in indices]
    return continuous_commands, continuous_reference, events
