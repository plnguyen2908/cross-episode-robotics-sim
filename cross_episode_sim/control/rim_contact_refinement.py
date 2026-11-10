"""Contact-preserving outward alternatives for shallow cookware rim grasps."""
import numpy as np


def outward_rim_contacts(rows, offsets=(0., .003, .006, .009, .012)):
    """Back off along the predicted approach axis without changing orientation.

    Model contact, score and provenance are preserved. These are proposals, not
    executable commands: endpoint, insertion, lift and physical retention still
    require validation. A shallow rim can need a few millimetres of clearance
    for the gripper linkage above the counter while retaining pad contact.
    """
    for row in rows:
        for offset in offsets:
            pose = np.asarray(row['tcp'], float).copy()
            pose[:3, 3] -= float(offset) * pose[:3, 2]
            yield dict(row, tcp=pose.tolist(), contact_backoff=float(offset))
