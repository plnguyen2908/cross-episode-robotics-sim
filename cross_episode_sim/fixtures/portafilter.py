"""Standalone portafilter detach/attach tasks using the composite's exact skills.

    python -m cross_episode_sim.fixtures.portafilter --action detach --output runs/pf_detach
    python -m cross_episode_sim.fixtures.portafilter --action attach --output runs/pf_attach

Attachment starts with an empty portafilter authored on the counter. Loaded
attachment is exercised by the coffee composite after the shared pour skill.
"""

import argparse
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.coffee.machine import PORTAFILTER, numbers


def prepare_attachment(output, manifest):
    """Author the detached starting scene; preserve the socket's constraint frames."""
    path = output / "robocasa_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    names = ("bayonet_axis_a", "bayonet_axis_b", "bayonet_locked")
    manifest["coffee"]["bayonet_constraint_data"] = {
        name: model.eq_data[model.equality(name).id].tolist() for name in names
    }
    tree = ET.parse(path)
    pose = np.array(manifest["coffee"]["portafilter_reference_pose"], dtype=float)
    # The manipulation target includes a 2 mm approach clearance. A free basket
    # dropped through that gap tips on its handle; author it at counter contact.
    pose[2, 3] -= 0.002
    body = tree.find(f".//body[@name='{PORTAFILTER}']")
    body.set("pos", numbers(pose[:3, 3]))
    body.set("quat", numbers(Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)))
    for name in names:
        tree.find(f"equality/*[@name='{name}']").set("active", "false")
    tree.write(path)
    info = next(i for i in manifest["bindings"] if i["body"] == PORTAFILTER)
    info.update(source="kitchen", position=pose[:3, 3].tolist())
    manifest["instruction"] = "Attach the portafilter to the coffee machine and withdraw the empty hand."
    manifest["atomic_initial_state"] = "empty portafilter resting on counter; socket constraints inactive"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("detach", "attach"))
    args, remaining = parser.parse_known_args(argv)
    from cross_episode_sim.tasks.coffee.workflow import main as workflow_main

    return workflow_main(["--atomic-skill", args.action + "-portafilter", *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
