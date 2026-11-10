"""Additional fixture-support preparation, separate from set-table placement."""
import json
from . import artifacts


def support_backend(receptacle, region, aperture_available=False):
    if region == 'interior' and 'cabinet' in receptacle and aperture_available:
        return 'cabinet'
    if 'stove burner' in receptacle:
        return 'burner'
    return 'generic'


def prepare_support_view(runtime, tool_points, root, kind, aperture=None):
    """Retain the grasp while exposing the support; no privileged scene input."""
    from ..control.held_view_posture import raise_held_view
    from ..navigation.shaped_docking import RobotGeometry
    from ..perception.held_attachment import observe_held_attachment
    category = 'cookware' if kind == 'burner' else 'bowl'
    try:
        tool_points, detail = observe_held_attachment(runtime.sensors.capture(),
            RobotGeometry(runtime.model), tool_points, category)
    except ValueError as exc:
        # A semantic refresh is optional: the retained pre-grasp RGB-D
        # attachment still goes through the unchanged swept collision checks.
        detail = dict(refreshed=False, reason=str(exc), attachment='retained pregrasp observation')
    artifacts.write_text(root/'held_surface_refresh.json', json.dumps(detail, indent=2))
    if kind == 'burner':
        from ..control.raised_payload_posture import raise_payload_clearance
        raise_payload_clearance(runtime, tool_points, root/'support_clearance', target_height=1.3)
    else:
        raise_held_view(runtime, tool_points, root/'support_clearance', target_height=1.4)
    if kind == 'cabinet':
        from ..navigation.aperture_dock import dock_to_aperture
        dock_to_aperture(runtime, aperture, tool_points, root/'aperture_dock',
                         distances=(.65,.60,.70,.55))
    return tool_points
