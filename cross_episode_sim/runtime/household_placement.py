"""Additional fixture-support preparation, separate from set-table placement."""
import json
from . import artifacts


def support_backend(receptacle, region, aperture_available=False):
    if region == 'interior' and 'cabinet' in receptacle and aperture_available:
        return 'cabinet'
    if 'stove burner' in receptacle:
        return 'burner'
    return 'generic'


# Datagen oracle cabinet stand poses: (distance from the door plane, metres past
# the closed handle on the side away from the hinge). First is the oracle's own.
ORACLE_CABINET_SIDE_DOCKS=((.51,.33),(.59,.33),(.43,.33),(.51,.25),(.59,.25),(.67,.33))


def prepare_support_view(runtime, tool_points, root, kind, aperture=None, *, cabinet_distances=None, cabinet_offsets=None, cabinet_require_observed_support=True, cabinet_side_docks=None):
    """Retain the grasp while exposing the support; no privileged scene input."""
    from ..control.held_view_posture import raise_held_view
    from ..navigation.shaped_docking import RobotGeometry
    from ..perception.held_attachment import observe_held_attachment
    category = 'cookware' if kind == 'burner' else 'bowl'
    geometry=RobotGeometry(runtime.model)
    if kind=='burner':
        m,d=runtime.model,runtime.data
        fingers={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
                 if (m.joint(j).name or '').startswith('robot_0/gripper/')}
        geometry=RobotGeometry(m,gripper_joint_positions=fingers)
    try:
        tool_points, detail = observe_held_attachment(runtime.sensors.capture(),
            geometry, tool_points, category)
    except ValueError as exc:
        # A semantic refresh is optional: the retained pre-grasp RGB-D
        # attachment still goes through the unchanged swept collision checks.
        detail = dict(refreshed=False, reason=str(exc), attachment='retained pregrasp observation')
    artifacts.write_text(root/'held_surface_refresh.json', json.dumps(detail, indent=2))
    print('HELD_SURFACE_REFRESH',json.dumps(detail),flush=True)
    if kind == 'burner':
        from ..control.raised_payload_posture import raise_payload_clearance
        raise_payload_clearance(runtime, tool_points, root/'support_clearance', target_height=1.3)
    else:
        raise_held_view(runtime, tool_points, root/'support_clearance', target_height=1.4)
    if kind == 'cabinet':
        from ..navigation.aperture_dock import dock_to_aperture
        options = {'offsets': cabinet_offsets} if cabinet_offsets is not None else {}
        from ..control.held_view_posture import arm_start_clear
        dock_to_aperture(runtime, aperture, tool_points, root/'aperture_dock',
                         distances=cabinet_distances or (.65,.60,.70,.55),
                         multi_goal=cabinet_distances is not None, require_observed_support=cabinet_require_observed_support,
                         arm_clear=lambda attempt: arm_start_clear(runtime, tool_points, root/f'arm_start_check_{attempt}'),
                         side_docks=cabinet_side_docks, **options)
    return tool_points
