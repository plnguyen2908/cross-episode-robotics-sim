"""Evaluation guard: every authored kitchen fixture joint must be present.

Datagen compiles scenes with only the task-bound cabinet/drawer articulated
(controller/house_scene.py in the datagen checkout froze the rest). A
sensor-only robot cannot tell a frozen cabinet from a working one, so the
evaluation pipeline refuses such a scene (user decision, November 16, 2026).
Rebuild it first with scripts/rearticulate_episode.py. No simulator state is
read beyond joint names; this runs before any robot motion.
"""
import re
from pathlib import Path

import mujoco

FIXTURE_JOINT = re.compile(r'<joint\b[^>]*>')


def authored_fixture_joints(scene_xml):
    """Hinge/slide joint names authored in the scene XML (no robot, room-door or free joints)."""
    names = set()
    for tag in FIXTURE_JOINT.findall(Path(scene_xml).read_text()):
        name = re.search(r'name="([^"]+)"', tag)
        kind = re.search(r'type="([^"]+)"', tag)
        if name is None or (kind is not None and kind.group(1) not in ('hinge', 'slide')):
            continue
        if name.group(1).startswith(('robot', 'room_door_')):
            continue
        names.add(name.group(1))
    return names


def missing_fixture_joints(source):
    source = Path(source)
    model = mujoco.MjModel.from_binary_path(str(source/'action_recording/model.mjb'))
    present = {model.joint(j).name for j in range(model.njnt)}
    return sorted(authored_fixture_joints(source/'scene.xml') - present)


def require_articulated(source):
    missing = missing_fixture_joints(source)
    if missing:
        raise RuntimeError(
            f'{len(missing)} authored fixture joints are frozen in {source}/action_recording/model.mjb '
            f'(e.g. {", ".join(missing[:4])}). Evaluate a rebuilt copy: scripts/rearticulate_episode.py '
            f'--source {source} --output <new folder>')
    return dict(fixture_joints_checked=True)
