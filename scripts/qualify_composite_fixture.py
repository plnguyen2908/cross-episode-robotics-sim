"""Qualify atomic fixture calls in a composite scene, without running its recipe."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture', choices=['microwave', 'fridge', 'room_door', 'cabinet_storage'])
    p.add_argument('--scene-from', type=Path,
                   help='Reuse an authored scene for a controlled controller retest; physics restarts from setup')
    p.add_argument('--retrieve', action='store_true', help='Qualify fridge drink retrieval and serving as well as the door')
    p.add_argument('--fridge-shelf-height', type=float, default=.83)
    p.add_argument('--fridge-asset', help='Native side-by-side fridge asset override')
    p.add_argument('--pick-place-back', action='store_true', help='Pick the fridge drink and place it back on the shelf')
    p.add_argument('--buttons-only', action='store_true', help='Test microwave Start/Stop from its initial closed state')
    p.add_argument('--door-only', action='store_true', help='Qualify cabinet opening and closing without object transfers')
    p.add_argument('--push-close', action='store_true', help='Use physical panel pushing to close')
    strategy = p.add_mutually_exclusive_group()
    strategy.add_argument('--push-contact', action='store_true', help='Push the exposed cabinet panel after the first handle stroke')
    strategy.add_argument('--handle-only', action='store_true', help='Compare retained handle grasps without panel pushing')
    from cross_episode_sim.control.smooth_action import add_arguments
    add_arguments(p)
    args = p.parse_args()
    from cross_episode_sim.generation.gpu_pool import lease_gpu
    from cross_episode_sim.rendering_gpu import resolve_gpus, worker_environment
    with lease_gpu('auto') as gpu:
        os.environ.update(worker_environment(dict(os.environ, MUJOCO_GL='egl'), gpu, resolve_gpus([gpu])[gpu]))
        from cross_episode_sim.generation.composite_pilot import configurations, scene_hash
        from cross_episode_sim.generation.composite_scenes import scene_specification
        from cross_episode_sim.env.household import PhysicalCompositeEnvironment
        from cross_episode_sim.tasks.composites.execution import fixture, transfer, execute_plan
        task = {'fridge': 'prepare_and_serve_meal', 'microwave': 'prepare_and_serve_meal', 'room_door': 'set_table',
                'cabinet_storage': 'clear_and_store'}[args.fixture]
        config = next(c for c in configurations() if c['task'] == task and c['variant'] == 'original')
        if args.fixture == 'room_door':
            config['scene']['room_doors'] = {'room_door_0': 'closed'}
        if args.fridge_asset:
            config['scene']['fridge_asset'] = args.fridge_asset
            config['scene']['fridge_shelf_height'] = args.fridge_shelf_height
        if args.smooth_action or args.optimize_motion_components:
            config['motion'] = {name: value for name, value in vars(args).items()
                                if name.startswith('smooth_action') or name == 'optimize_motion_components'}
        config['scene_spec_sha256'] = scene_hash(config)
        Path('runs').mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=f'qualify_{args.fixture}_', dir='runs')).resolve()
        print(f'OUTPUT={out}', flush=True)
        (out/'config.json').write_text(json.dumps(config, indent=2))
        env = PhysicalCompositeEnvironment(config, out)
        report = dict(success=False, scope='fixture calls only; not a composite success',
                      door_only=args.door_only, push_contact_override=args.push_contact,
                      handle_only=args.handle_only, push_close=args.push_close)
        try:
            if args.scene_from:
                source = args.scene_from.resolve()
                previous = json.loads((source/'config.json').read_text())
                if previous['scene'] != config['scene'] or previous['task'] != config['task']:
                    raise ValueError('Reused scene configuration differs from the requested fixture test')
                manifest = json.loads((source/'task_manifest.json').read_text())
                import cross_episode_sim.env.household as household
                household.build_scene = lambda output, params, specification: manifest
                (out/'task_manifest.json').write_text(json.dumps(manifest, indent=2))
                report['reused_authored_scene'] = str(source)
            env.configure(config['scene'], scene_specification(config))
            if args.push_close:
                if args.fixture not in ('cabinet_storage', 'fridge'):
                    raise ValueError('--push-close requires a cabinet or fridge')
                env.fixture_bindings['fridge' if args.fixture == 'fridge' else 'kitchen_cabinet']['door_push_close'] = True
            if args.push_contact:
                if args.fixture not in ('cabinet_storage', 'fridge'):
                    raise ValueError('--push-contact requires a cabinet or fridge')
                env.fixture_bindings['fridge' if args.fixture == 'fridge' else 'kitchen_cabinet']['door_push_contact'] = True
            if args.handle_only:
                if args.fixture not in ('cabinet_storage', 'fridge'):
                    raise ValueError('--handle-only requires a cabinet or fridge')
                env.fixture_bindings['fridge' if args.fixture == 'fridge' else 'kitchen_cabinet']['door_push_contact'] = False
                env.fixture_bindings['fridge' if args.fixture == 'fridge' else 'kitchen_cabinet']['door_push_close'] = False
            env.save_initial_state(out/'initial_state.npz')
            if args.fixture == 'microwave':
                plan = (fixture('open_door', 'microwave') + fixture('close_door', 'microwave')
                        + fixture('push_button', 'microwave_start') + fixture('push_button', 'microwave_stop'))
                if args.buttons_only:
                    plan=plan[4:]
            elif args.fixture == 'fridge':
                plan = (fixture('open_door', 'fridge')
                        + (transfer('drink_bottle_01', 'fridge', 'interior') if args.pick_place_back else
                           transfer('drink_bottle_01', 'dining_table') if args.retrieve else [])
                        + fixture('close_door', 'fridge'))
            elif args.fixture == 'room_door':
                plan = (fixture('open_door', 'room_door_0_left')
                        + fixture('open_door', 'room_door_0_right'))
            else:
                # Keep the original wider-cabinet scene but qualify only two
                # transfers before attempting the complete storage recipe.
                plan = (fixture('open_door', 'kitchen_cabinet')
                        + transfer('bowl_01', 'kitchen_cabinet', 'interior')
                        + transfer('cup_01', 'kitchen_cabinet', 'interior')
                        + fixture('close_door', 'kitchen_cabinet'))
                report['tested_objects'] = ['bowl_01', 'cup_01']
                report['scope'] = 'one bowl and one cup into cabinet; not a full composite success'
                if args.door_only:
                    plan = (fixture('open_door', 'kitchen_cabinet')
                            + fixture('close_door', 'kitchen_cabinet'))
                    report['tested_objects'] = []
                    report['scope'] = 'cabinet open/close qualification; no object transfers'
            (out/'plan.json').write_text(json.dumps(plan, indent=2))
            report.update(execute_plan(env.interface, plan, args.fixture))
            snapshot = env.snapshot()
            (out/'final_snapshot.json').write_text(json.dumps(snapshot, indent=2))
            if args.fixture == 'fridge' and args.pick_place_back:
                report['final_checks'] = {
                    'drink_in_fridge': snapshot['objects']['drink_bottle_01']['support'] == 'fridge',
                    'empty_hand': snapshot['held'] is None,
                    'fridge_closed': snapshot['fixtures']['fridge']['closed']}
                if not all(report['final_checks'].values()):
                    raise RuntimeError('Fridge pick/place final checks failed')
            if args.fixture == 'cabinet_storage' and not args.door_only:
                report['final_checks'] = {
                    role: snapshot['objects'][role]['support'] == 'kitchen_cabinet'
                          and snapshot['objects'][role]['region'] == 'interior'
                    for role in ('bowl_01', 'cup_01')}
                report['final_checks']['cabinet_closed'] = snapshot['fixtures']['kitchen_cabinet']['closed']
                report['final_checks']['empty_hand'] = snapshot['held'] is None
                if not all(report['final_checks'].values()):
                    raise RuntimeError('Final cabinet storage checks failed')
            report['video'] = str(env.render(out, speedup=1.))
            report['success'] = True
        except Exception as exc:
            report['error'] = str(exc)
        finally:
            env.close()
            (out/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
