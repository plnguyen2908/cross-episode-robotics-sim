"""What the reorder stack needs to know about a robot.

The tools were written against RB-Y1 and name its joints directly -- about 220
references to `right_arm_*`, `torso_*`, `head_0`, `ee_site_r`, `right_finger_act`
and friends. A second embodiment cannot be configured in, only ported, unless
those names come from one place. This is that place.

Profiles are data, not behaviour. Anything that differs structurally rather than
by name -- RB-Y1 has a torso and a pan/tilt head, the Franka has neither -- is
expressed as an empty tuple, so callers branch on "does this robot have a head"
instead of on the robot's identity.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class RobotProfile:
    name: str
    namespace: str
    base_joints: tuple[str, ...]
    base_actuators: tuple[str, ...]
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    tcp_site: str
    gripper_actuator: str
    finger_bodies: tuple[str, ...]
    # Open and close commands for the single gripper actuator, in ctrl units.
    gripper_open: float
    gripper_close: float
    # Widest object the fingers can straddle. Grasp annotations are filtered
    # against this, so a wrong value silently admits ungraspable objects.
    gripper_stroke_m: float
    # Empty when the robot has no such chain, rather than absent: callers test
    # the tuple, so a robot without a torso needs no special case.
    torso_joints: tuple[str, ...] = ()
    torso_actuators: tuple[str, ...] = ()
    head_joints: tuple[str, ...] = ()
    head_camera: str | None = None
    # Folded pose used for driving, in arm_joints order.
    travel_posture: tuple[float, ...] = ()
    ready_elbow_rad: float | None = None
    placement_surface_fractions: tuple[float, ...] = (-.55, -.25, 0., .25, .55)
    # Use a shipped cuRobo model or export the live MuJoCo embodiment.
    curobo_dir: Path | None = None
    curobo_config: str | None = None
    planner_from_mujoco: bool = False
    manipulation_offsets: tuple[tuple[float, float], ...] = ((.60, 0.), (.70, 0.), (.78, 0.))
    pregrasp_standoff_m: float = .10
    grasp_width_mode: str = 'whole_object'
    minimum_contact_width_m: float = 0.0
    preferred_contact_width_m: float = 0.0
    pickup_distance_m: float = .60
    adaptive_grasp_aperture: bool = True
    notes: str = ""

    @property
    def has_torso(self) -> bool:
        return bool(self.torso_joints)

    @property
    def has_head(self) -> bool:
        return bool(self.head_joints)

    @property
    def can_plan_arm(self) -> bool:
        return self.curobo_config is not None or self.planner_from_mujoco

    def prefixed(self, names):
        return tuple(self.namespace + name for name in names)


RBY1M = RobotProfile(
    name="rby1m",
    namespace="robot_0/",
    base_joints=("base_x", "base_y", "base_theta"),
    base_actuators=("base_x_act", "base_y_act", "base_theta_act"),
    arm_joints=tuple(f"right_arm_{i}" for i in range(7)),
    arm_actuators=tuple(f"right_arm_{i + 1}_act" for i in range(7)),
    torso_joints=tuple(f"torso_{i}" for i in range(6)),
    torso_actuators=tuple(f"link{i + 1}_act" for i in range(6)),
    head_joints=("head_0", "head_1"),
    head_camera="head_camera",
    tcp_site="ee_site_r",
    gripper_actuator="right_finger_act",
    finger_bodies=("ee_finger_r1", "ee_finger_r2"),
    gripper_open=-100.0,
    gripper_close=0.0,
    # Measured: fingertips 114.5 mm apart fully open, 14.7 mm closed.
    gripper_stroke_m=0.1145,
    ready_elbow_rad=-1.1,
    travel_posture=(0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0),
    curobo_config="rby1m_right_arm_holobase.yml",
    manipulation_offsets=((.55, -.28), (.62, -.24), (.70, -.18), (.78, 0.)),
)


MOBILE_FRANKA = RobotProfile(
    name="franka_droid",
    namespace="robot_0/",
    base_joints=("base_x", "base_y", "base_theta"),
    base_actuators=("base_x_act", "base_y_act", "base_theta_act"),
    arm_joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    # The Franka model drives each joint through an actuator of the same name.
    arm_actuators=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    # No torso and no pan/tilt head: the wrist camera moves with the arm, so
    # gaze control has nothing to command and the reach logic has no trunk to
    # bend. Both fall out of the empty tuples.
    head_camera="wrist_cam",
    # The hand is a separate model mounted under its own sub-namespace, so these
    # carry a `gripper/` prefix the arm names do not. The validator caught this;
    # it is exactly the kind of drift a profile is meant to make impossible.
    tcp_site="gripper/grasp_site",
    gripper_actuator="gripper/fingers_actuator",
    finger_bodies=("gripper/left_pad", "gripper/right_pad"),
    gripper_open=0.0,
    gripper_close=255.0,
    # Robotiq 2f85 nominal stroke. NOT measured in this scene yet -- verify with
    # the same fingertip probe used for RB-Y1 before trusting grasp filtering.
    gripper_stroke_m=0.085,
    pickup_distance_m=.50,
    placement_surface_fractions=(-1., -.75, -.55, -.25, 0., .25, .55, .75, 1.),
    # Upright FR3 configuration near the straight q=0 pose. Joint 4 cannot
    # exceed -0.1518 rad and joint 6 cannot go below 0.5445 rad in this model.
    # A small limit margin keeps this posture usable by collision-aware planning.
    # Other joints at zero stack the arm over the base for navigation.
    travel_posture=(0.0, 0.0, 0.0, -0.20, 0.0, 0.60, 0.0),
    curobo_dir=None,
    curobo_config=None,
    planner_from_mujoco=True,
    pregrasp_standoff_m=.06,
    grasp_width_mode='contact_region',
    minimum_contact_width_m=0.006,
    preferred_contact_width_m=.03,
    adaptive_grasp_aperture=False,
    notes=(
        "The task adapter exports the actual MuJoCo chain and conservative link "
        "spheres for the shared cuRobo planner. Rebuilt after base motion; full "
        "MuJoCo geometry validates contact paths. Tuck stays inside joint limits."
    ),
)


# Benchmark B3: the same FR3 and hand, stood on a TidyBot++-sized base. Identical
# joint names -- only the base geometry and the arm's mount height differ -- so the
# profile is the mobile Franka's under another name.
FRANKA_TIDYBOT = replace(MOBILE_FRANKA, name="franka_tidybot")


PROFILES = {profile.name: profile
            for profile in (RBY1M, MOBILE_FRANKA, FRANKA_TIDYBOT)}


def profile_for(name: str) -> RobotProfile:
    if name not in PROFILES:
        raise KeyError(f"no robot profile for {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]


def validate(profile: RobotProfile, model) -> dict[str, list[str]]:
    """Check every name in a profile against a loaded model.

    A profile is only useful if it is true. Returns the missing names by kind so
    a caller can fail loudly at start-up rather than at the first reach.
    """
    import mujoco

    def present(kind, names):
        missing = []
        for name in names:
            try:
                getattr(model, kind)(profile.namespace + name)
            except (KeyError, ValueError):
                missing.append(name)
        return missing

    report = {
        "joint": present("joint", profile.base_joints + profile.arm_joints
                         + profile.torso_joints + profile.head_joints),
        "actuator": present("actuator", profile.base_actuators + profile.arm_actuators
                            + profile.torso_actuators + (profile.gripper_actuator,)),
        "site": present("site", (profile.tcp_site,)),
        "body": present("body", profile.finger_bodies),
    }
    if profile.head_camera:
        try:
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                              profile.namespace + profile.head_camera)
        except Exception:  # noqa: BLE001
            report.setdefault("camera", []).append(profile.head_camera)
    return {kind: names for kind, names in report.items() if names}
