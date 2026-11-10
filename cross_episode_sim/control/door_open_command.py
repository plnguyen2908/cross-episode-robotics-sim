"""One sensor-driven room-door opening configuration for live and replay tests."""
import sys


def door_open_command(grasp, output, *, contact_motion_profile='fast', handle_then_push=False):
    if contact_motion_profile not in {'feedback', 'fast'}:
        raise ValueError('Unknown contact motion profile')
    command = [sys.executable, 'scripts/test_sensor_hinged_cycle.py',
            '--grasp-run', str(grasp), '--output', str(output),
            '--open-only', '--push-after', '15', '--base-push',
            '--adaptive-push-orientation', '--planar-push', '--contact-retreat',
            '--auto-probe-direction', '--probe-direction', '1',
            '--absolute-open-angle', '90', '--orientation-gain', '1',
            '--shared-oracle-contact', '--exact-closed-fingers',
            '--contact-motion-profile', contact_motion_profile]

    if handle_then_push:command.extend(['--handle-then-push','--pull-push-after','45','--oracle-contact-poses'])
    return command
