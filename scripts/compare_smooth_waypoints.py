"""Matched physical nav + pick/carry/place experiment; no default policy changes."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['baseline','smooth','continuous','fast'],required=True)
    p.add_argument('--source',type=Path,default=ROOT/'runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original')
    p.add_argument('--output',type=Path)
    p.add_argument('--fast-profile',choices=['conservative','target30','quicker','finer'],default='conservative')
    args=p.parse_args()
    output=args.output or Path(tempfile.mkdtemp(prefix='smooth_'+args.variant+'_',dir=ROOT/'runs'))
    output.mkdir(parents=True,exist_ok=True)
    print('OUTPUT='+str(output.resolve()),flush=True)
    from cross_episode_sim.generation.gpu_pool import lease_gpu
    from cross_episode_sim.rendering_gpu import resolve_gpus,worker_environment
    with lease_gpu('auto',waiting=lambda:print('Waiting for idle GPU',flush=True)) as gpu:
        os.environ.update(worker_environment(dict(os.environ,MUJOCO_GL='egl'),gpu,resolve_gpus([gpu])[gpu]))
        os.environ['ROBOCASA_DIR']=str(ROOT/'external/robocasa/robocasa')
        import random
        import numpy as np
        import torch
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        import cross_episode_sim.env.household as household
        from cross_episode_sim.generation.composite_scenes import scene_specification
        source=args.source.resolve()
        config=json.loads((source/'config.json').read_text())
        # This historical ablation installs its own policy below. Do not let
        # production defaults silently turn its baseline into the fast policy.
        config['motion'] = dict(optimize_motion_components=False, smooth_action=False)
        manifest=json.loads((source/'task_manifest.json').read_text())
        # Reuse immutable datagen scene/assets to make both variants identical.
        household.build_scene=lambda *a,**kw:json.loads(json.dumps(manifest))
        env=household.PhysicalCompositeEnvironment(config,output)
        result=dict(variant=args.variant,fast_profile=args.fast_profile,source=str(source),success=False)
        started=time.monotonic()
        try:
            env.configure(config['scene'],scene_specification(config))
            c=env.controller
            c.smooth_waypoint_experiment=args.variant in ('smooth','continuous','fast')
            c.retain_carry_posture_experiment=args.variant in ('continuous','fast')
            c.direct_carry_departure_experiment=args.variant=='fast'
            if args.variant=='fast':
                # Plan checked forward primitives, then simplify and round with tangent heading.
                c.args.base_motion='turn_drive'
                c.forward_curve_experiment=True
                c.centered_carry_experiment=True
                c.args.nav_speed=.45
                c.args.turn_speed=1.
                c.args.motion_slowdown=.3
                c.contact_waypoint_seconds=.10
                c.experimental_joint_speed=1.2
                c.experimental_joint_acceleration=4.
                if args.fast_profile in ('target30','quicker','finer'):
                    c.args.nav_speed=.6
                    c.args.turn_speed=1.2
                    c.args.motion_slowdown=.2
                    c.contact_waypoint_seconds=.08
                    c.experimental_joint_speed=1.5
                    c.experimental_joint_acceleration=8.
                if args.fast_profile in ('quicker','finer'):
                    c.local_joint_retiming_experiment=True
                    c.args.nav_speed=.8
                    c.args.turn_speed=1.5
                    c.args.motion_slowdown=.12
                    c.contact_waypoint_seconds=.06
                    c.experimental_joint_speed=1.8
                    c.experimental_joint_acceleration=10.
                if args.fast_profile=='finer':
                    c.contact_sample_seconds=.01
                    c.forward_curve_control_dt=.01
            env.save_initial_state(output/'initial_state.npz')
            (output/'initial_snapshot.json').write_text(json.dumps(env.snapshot(),indent=2))
            plan=[('navigate',dict(target='food_container_01')),
                  ('pick',dict(target='food_container_01')),
                  ('navigate',dict(target='food_container_01',destination='dining_table')),
                  ('place',dict(receptacle='dining_table'))]
            for skill,arguments in plan:
                print('ACTION',skill,arguments,flush=True)
                event=env.interface.execute_skill(skill,arguments)
                print('RESULT',event,flush=True)
                if not event['success']:raise RuntimeError(str(event))
            final=env.snapshot()
            result['success']=final['objects']['food_container_01']['support']=='dining_table' and final['held'] is None
            result['final_snapshot']=final
            result['physics_wall_seconds']=time.monotonic()-started
            result['simulation_seconds']=float(c.data.time)
            result['calls']=env.interface.history
            result['navigation']=c.report.get('navigation',[])
            result['base_motion']='forward_curve' if getattr(c,'forward_curve_experiment',False) else c.args.base_motion
            if args.variant=='fast':
                assert result['navigation'], 'No navigation validation recorded'
                assert all(n['base_motion']=='forward_curve' and n['holonomic_distance_m']==0
                           for n in result['navigation']), 'Fast navigation must remain forward-facing'
            result['interpolation']=[r for r in c.report.get('stages',[]) if 'smooth_contact' in r or 'smooth_route' in r]
            print('RENDERING',flush=True)
            result['video']=str(env.render(output))
        except Exception as exc:
            import traceback
            traceback.print_exc();result['error']=str(exc)
        finally:
            result['calls']=env.interface.history if hasattr(env,'interface') else []
            env.close()
            result['total_wall_seconds']=time.monotonic()-started
            (output/'comparison.json').write_text(json.dumps(result,indent=2))
            print('COMPLETE',result['success'],flush=True)
        return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
