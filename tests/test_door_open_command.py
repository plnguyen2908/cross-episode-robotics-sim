"""Keep live exploration on the physically qualified original pulling command."""
from cross_episode_sim.control.door_open_command import door_open_command


def test_live_default_matches_run50_successful_continuation():
    command=door_open_command('grasp','opening')
    assert command[1:]==[
        'scripts/test_sensor_hinged_cycle.py',
        '--grasp-run','grasp','--output','opening',
        '--open-only','--push-after','15','--base-push',
        '--adaptive-push-orientation','--planar-push','--contact-retreat',
        '--auto-probe-direction','--probe-direction','1',
        '--absolute-open-angle','90','--orientation-gain','1',
        '--shared-oracle-contact','--exact-closed-fingers',
        '--contact-motion-profile','fast',
    ]


def test_experimental_panel_push_requires_explicit_opt_in():
    assert '--handle-then-push' not in door_open_command('a','b')
    assert '--handle-then-push' in door_open_command('a','b',handle_then_push=True)
