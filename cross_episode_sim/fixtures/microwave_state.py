"""Contact-only microwave state; independent of manipulation planners."""
import mujoco
import numpy as np

def next_state(on, start, stop, door_open):
    """RoboCasa Microwave.update_state contact-triggered latch semantics."""
    if door_open:
        return False
    if on and stop:
        return False
    if not on and start:
        return True
    return on

def button_contacts(self, name):
    force = 0.
    touched = False
    for i, c in enumerate(self.data.contact):
        gids = (c.geom1, c.geom2)
        if self.button_ids[name] not in gids:
            continue
        other = gids[1] if gids[0] == self.button_ids[name] else gids[0]
        body = self.model.body(self.model.geom_bodyid[other]).name or ''
        if body.startswith('robot_0/gripper/'):
            touched = True
            if self.model.geom_bodyid[other] in self.pad_bids:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                force += max(0., float(f[0]))
    return touched, force
