"""Compare shared base-assisted opening against the legacy arm-only option.

Uses qualify_composite_fixture arguments plus --base-push --follow-arm,
--push-speed, --smooth-preparation, and optional strict --direct-push mode.
The production implementation is in fixtures/base_articulation.py.
"""
import sys
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--base-push', action='store_true')
    parser.add_argument('--follow-arm', action='store_true')
    parser.add_argument('--direct-push', action='store_true', help='Try the panel from the post-pull stance; forbid redocking fallback')
    parser.add_argument('--smooth-preparation', action='store_true',
                        help='Use conservative live interpolation/retiming for the approach and repositioning')
    parser.add_argument('--push-speed', type=float, default=.06)
    args, remaining = parser.parse_known_args()
    if not 0 < args.push_speed <= .30:
        parser.error('push speed must be in (0, .30] m/s')
    if args.base_push and not args.follow_arm:
        parser.error('Base assistance requires wrist coordination; add --follow-arm')
    if args.direct_push and not (args.base_push and args.follow_arm):
        parser.error('--direct-push requires --base-push --follow-arm')
    sys.argv = [sys.argv[0], *remaining]
    # Install the controller override at configure time, after GPU selection.
    from cross_episode_sim.env.household import PhysicalCompositeEnvironment
    configure = PhysicalCompositeEnvironment.configure
    def configured(self, *a, **kw):
        if args.smooth_preparation:
            self.config['motion'] = dict(smooth_action=True, smooth_action_profile='conservative')
            (self.output/'config.json').write_text(json.dumps(self.config, indent=2))
        if args.direct_push:
            from scripts.direct_door_push import install as install_direct
            install_direct()
        (self.output/'base_push_experiment.json').write_text(json.dumps(dict(
            base_push=args.base_push, follow_arm=args.follow_arm, direct_push=args.direct_push,
            push_speed_m_s=args.push_speed, smooth_preparation=args.smooth_preparation, scope='isolated cabinet opening experiment'), indent=2))
        result = configure(self, *a, **kw)
        self.controller.base_assisted_articulation = args.base_push
        self.controller.contact_base_speed = args.push_speed
        for binding in self.fixture_bindings.values():
            binding['base_assisted_articulation'] = args.base_push
            binding['contact_base_speed'] = args.push_speed
        return result
    PhysicalCompositeEnvironment.configure = configured
    from scripts.qualify_composite_fixture import main
    raise SystemExit(main())
