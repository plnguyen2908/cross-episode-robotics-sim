"""Object-by-object mixed-source grasp qualification using the shared cuRobo controller."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

from cross_episode_sim.manipulation.molmo_objects import assets_root
from cross_episode_sim.paths import MOLMO_OBJECTS_DIR, PACKAGE_DIR, grasp_registry_path

PILOT = ['lightwheel/salt_and_pepper_shaker/SaltShaker001',
         'lightwheel/cookie_dough_ball/CookieDoughBall001',
         'lightwheel/cream_cheese_stick/CreamCheeseStick001']
MOLMO = ['Egg_1','Apple_1','Tomato_1','Potato_1','Mug_1','Cup_1',
         'Salt_Shaker_1','Pepper_Shaker_1','Soap_Bottle_1','Bottle_1']


CONTROLLER_SOURCES = (
    'manipulation/robocasa.py', 'manipulation/grasp_qualification.py', 'manipulation/surface_grasps.py',
    'manipulation/recovery.py', 'manipulation/edge_access.py', 'manipulation/cross_room.py',
    'robot/profile.py', 'robot/embodiment.py', 'robot/arm_model.py', 'robot/arm_planner.py',
    'robot/tidybot_franka.py', 'controller/annotated_grasp.py', 'controller/base.py',
    'controller/transfer.py', 'controller/manipulation.py', 'controller/reorder_chain.py',
    'controller/navigation.py')


def controller_fingerprint():
    return {name: hashlib.sha256((PACKAGE_DIR/name).read_bytes()).hexdigest() for name in CONTROLLER_SOURCES}


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2)); temporary.replace(path)


def entry(source, asset):
    return dict(source=source, asset=asset, key=source+'__'+asset.replace('/', '__'))


def catalog(robosuite_python):
    code = ('import json; from cross_episode_sim.grasping.sweep import discover; '
            'print("ASSET_CATALOG="+json.dumps(discover()[0]))')
    run = subprocess.run([robosuite_python, '-c', code], text=True, capture_output=True, check=True)
    objects = json.loads(next(line.split('=',1)[1] for line in run.stdout.splitlines() if line.startswith('ASSET_CATALOG=')))
    root=assets_root()
    models={p.name.removesuffix('_mesh.xml'):str(p.resolve())
            for p in (root/'objects/thor').rglob('*_mesh.xml')}
    atomic_json(MOLMO_OBJECTS_DIR/'source_index.json',models)
    names=[]; skipped=[]
    for directory in sorted((root/'grasps/droid').iterdir()):
        if not directory.is_dir(): continue
        name=directory.name
        annotation=directory/(name+'_grasps_filtered.npz')
        if name in models and annotation.is_file(): names.append(name)
        else: skipped.append(dict(asset=name,reason='missing model or filtered grasp file'))
    atomic_json(MOLMO_OBJECTS_DIR/'discovery_exclusions.json',skipped)
    # Interleave sources so both collections make progress immediately.
    molmo=[entry('molmo',name) for name in names]
    native=[entry('robocasa',obj['asset']) for obj in objects]
    return [item for i in range(max(len(molmo),len(native)))
            for group in (molmo,native) if i<len(group) for item in [group[i]]]



def register_success(record):
    path = grasp_registry_path(record['model_xml'])
    payload = json.loads(path.read_text()) if path.is_file() else dict(schema_version=1, grasps=[])
    # Per-trial evidence is immutable; repeated resume must not duplicate it.
    if not any(old.get('report') == record['report'] for old in payload['grasps']):
        payload['grasps'].append(record)
    atomic_json(path, payload)
    return str(path)


def status(output, objects, attempts, phase):
    rows = []
    for obj in objects:
        path = output/'objects'/obj['key']/'status.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    result = dict(status=phase, total_objects=len(objects), attempts_per_object=attempts,
        completed_objects=sum(r['complete'] for r in rows),
        finished_trials=sum(len(r['trials']) for r in rows),
        qualified_grasps=sum(r['qualified_grasps'] for r in rows),
        objects_with_qualified_grasps=sum(r['qualified_grasps']>0 for r in rows),
        physical_attempts=sum(t.get('physical_attempted',False) for r in rows for t in r['trials']),
        full_placement_passes=sum(t.get('placement_success',False) for r in rows for t in r['trials']),
        objects=rows)
    atomic_json(output/'summary.json', result)
    # Snapshot for visualization: exactly one row per asset, replaced atomically.
    known={r['key']:r for r in rows}
    current=[]
    for obj in objects:
        r=known.get(obj['key'])
        trials=r['trials'] if r else []
        current.append(dict(**obj, sweep_status=phase,
            status=('setup_failed' if r and r.get('setup_error') else
                    'complete' if r and r['complete'] else
                    'stopped' if r and phase=='stopped' else
                    'in_progress' if r else 'pending'),
            failure_category=(r.get('failure_category') or (trials[-1].get('failure_category') if trials else None)) if r else None,
            finished_trials=len(trials), qualified_grasps=r['qualified_grasps'] if r else 0,
            placement_passes=sum(bool(t.get('placement_success')) for t in trials),
            cross_room_passes=sum(bool(t.get('cross_room_success')) for t in trials),
            edge_push_trials=sum(bool(t.get('edge_push_used')) for t in trials),
            latest_error=(r.get('setup_error') or (trials[-1].get('error') if trials else None)) if r else None))
    atomic_json(output/'current_results.json', current)
    temp=output/'progress.log.tmp'
    temp.write_text(''.join(json.dumps(row)+'\n' for row in current))
    temp.replace(output/'progress.log')

    # This index contains ONLY objects with physically qualified grasps.
    atomic_json(output/'qualified_objects.json', [dict(source=r['source'], asset=r['asset'],
        grasps_path=r['grasps_path'], qualified_grasps=r['qualified_grasps'])
        for r in rows if r['qualified_grasps']])
    lines = ['# Shared cuRobo grasp qualification', '',
        f"{result['completed_objects']}/{len(objects)} objects complete; {result['qualified_grasps']} qualified grasps; "
        f"{result['objects_with_qualified_grasps']} objects with annotations.", '',
        'Grasp qualification requires >=10 cm physical lift and two-second retention. Contact metrics are diagnostic only. Placement is evaluated separately.', '',
        '| Object | Trials | Qualified | Placement passes | Details |', '|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['source']} / {r['asset']} | {len(r['trials'])}/{attempts} | {r['qualified_grasps']} | "
            f"{sum(t.get('placement_success',False) for t in r['trials'])} | [results](objects/{r['key']}/results.md) |")
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    return result


def failure_category(report, log_text='', qualified=False):
    if qualified:
        return None if report.get('success') else 'placement'
    error=report.get('error') or log_text
    selection=report.get('annotation_selection',{})
    if 'Empty filtered grasp annotations' in error: return 'empty_annotations'
    if not report: return 'candidate_generation'
    if selection.get('width_and_approach_candidates') == 0:
        counts=selection.get('filter_rejections',{})
        if counts.get('approach') == selection.get('tested_orientation_variants'):
            return 'approach_unavailable'
        return 'grasp_filtering'
    if not report.get('physical_grasp_attempted'): return 'planning'
    return 'dropped_or_no_lift'


def run_object(obj, gpu, args):
    out = args.output/'objects'/obj['key']; out.mkdir(parents=True, exist_ok=True)
    path = out/'status.json'
    record = json.loads(path.read_text()) if path.exists() else dict(**obj, complete=False,
        trials=[], qualified_grasps=0, grasps_path=str(out/'grasps.json'))
    if record['complete']:
        return record
    print(json.dumps(dict(event='object_started',source=obj['source'],asset=obj['asset'],gpu=gpu)),flush=True)
    initial = out/'initial_scene'
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES=str(gpu))
    if not (initial/'setup.json').is_file():
        with (out/'setup.log').open('w') as log:
            result = subprocess.run([args.robosuite_python, '-m', 'cross_episode_sim.grasping.prepare_recording',
                '--source',obj['source'],'--asset',obj['asset'],'--output',str(initial)],
                env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            detail=(out/'setup.log').read_text(errors='replace')
            record.update(complete=True,setup_error=detail.strip().splitlines()[-1],
                          failure_category='empty_annotations' if 'Empty filtered grasp annotations' in detail else 'setup')
            atomic_json(path,record)
            print(json.dumps(dict(event='setup_failed',source=obj['source'],asset=obj['asset'],log=str(out/'setup.log'))),flush=True)
            (out/'results.md').write_text('# Setup failed\n\nSee setup.log; no physical grasp attempted.\n')
            return record
    annotations = json.loads((out/'grasps.json').read_text()) if (out/'grasps.json').exists() else []
    for index in range(args.attempts):
        if any(t['index']==index for t in record['trials']): continue
        trial = out/f'trial_{index:03d}'
        # Same authored object pose, different grasps; only the initial stance
        # varies slightly to avoid spending all five attempts at one bad reach.
        stance = (.42,.50,.38,.46,.54)[index]
        command = [args.python,'-u','-m','cross_episode_sim.manipulation.robocasa',
            '--recording',str(initial),'--asset',obj['asset'],'--output',str(trial),
            '--grasp-source','molmo' if obj['source']=='molmo' else 'surface',
            '--approach-policy','any-above-table','--stand-off',str(stance),
            '--qualify-grasp','--previous-trials',str(out),
            '--grasp-family',('top','oblique','side','oblique','side')[index]]
        if getattr(args, 'edge_placement', False):
            command.append('--edge-placement')
        if getattr(args, 'cross_room', False):
            command.append('--cross-room')
        if getattr(args, 'thin_edge_fallback', False):
            command.append('--thin-edge-fallback')
        if getattr(args, 'recover_stance', False):
            command.append('--recover-stance')
        trial.mkdir(parents=True, exist_ok=True)
        atomic_json(trial/'command.json',dict(command=command,gpu=gpu,
                    source_hashes=controller_fingerprint()))
        print(json.dumps(dict(event='trial_started',source=obj['source'],asset=obj['asset'],trial=index+1,grasp_family=('top','oblique','side','oblique','side')[index],gpu=gpu,log=str(out/f'trial_{index:03d}.log'))),flush=True)
        start = time.monotonic()
        with (out/f'trial_{index:03d}.log').open('w') as log:
            run = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        report_path = trial/'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        qualified = trial/'qualified_grasp.json'
        selection = report.get('annotation_selection',{})
        row = dict(index=index,recovery_enabled=bool(getattr(args,'recover_stance',False)),grasp_family=('top','oblique','side','oblique','side')[index],physical_attempted=report.get('physical_grasp_attempted',False),
            selected_annotation=selection.get('selected_index'),qualified=qualified.is_file(),
            placement_success=bool(run.returncode == 0 and report.get('success')),
            cross_room_success=bool(run.returncode == 0 and report.get('success') and report.get('cross_room_navigation_completed')),
            edge_push_used=bool(report.get('edge_push_used')),error=report.get('error'),
            returncode=run.returncode,elapsed_seconds=round(time.monotonic()-start,1),
            video=next((str(trial/name) for name in ('robocasa_cross_room.mp4','robocasa_curobo.mp4') if (trial/name).is_file()),None),
            report=str(report_path) if report_path.exists() else None,
            video_status=report.get('video_status'),planning_candidates=selection.get('width_and_approach_candidates'))
        log_text=(out/f'trial_{index:03d}.log').read_text(errors='replace')
        if not report:
            row['error']=log_text.strip().splitlines()[-1] if log_text.strip() else 'Missing report and empty trial log'
        row['failure_category']=failure_category(report,log_text,qualified.is_file())
        if qualified.is_file():
            annotation = json.loads(qualified.read_text())
            annotation['placement_success'] = row['placement_success']
            row['registry_sidecar'] = register_success(annotation)
            annotations.append(annotation)
            atomic_json(out/'grasps.json',annotations)
        record['trials'].append(row)
        record['trials'].sort(key=lambda t:t['index'])
        record['qualified_grasps'] = len(annotations)
        record['complete'] = len(record['trials']) == args.attempts
        if getattr(args, 'stop_after_success', False) and row['placement_success'] and (not getattr(args,'cross_room',False) or row['cross_room_success']):
            record.update(complete=True, stop_reason='full transfer succeeded; remaining attempts unnecessary')
        atomic_json(path,record)
        lines = [f"# {obj['source']} / {obj['asset']}", '',
                 '| Attempt | Physical closure | Qualified hold | Full place | Video | Error |',
                 '|---|---|---|---|---|---|']
        for item in record['trials']:
            video = f"[video]({Path(item['video']).relative_to(out)})" if item['video'] else 'none'
            error = str(item.get('error') or '').replace('|','/').replace('\n',' ')
            lines.append(f"| {item['index']+1} | {item['physical_attempted']} | {item['qualified']} | {item['placement_success']} | {video} | {error} |")
        (out/'results.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(object=obj['asset'],source=obj['source'],**row)),flush=True)
        if record['complete']:
            break
    atomic_json(out/'grasps.json',annotations)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('pilot','combined'),default='pilot')
    p.add_argument('--attempts',type=int,choices=(3,4,5),default=5)
    p.add_argument('--gpus',default='0,1')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--edge-placement',action='store_true')
    p.add_argument('--cross-room',action='store_true')
    p.add_argument('--thin-edge-fallback',action='store_true')
    p.add_argument('--stop-after-success',action='store_true')
    p.add_argument('--recover-stance',action='store_true',help='Use local pickup and loaded placement recovery')
    p.add_argument('--only',action='append',default=[],help='Asset key to retry; repeat for a small validation subset')
    p.add_argument('--python',default=sys.executable,help='Interpreter with cross_episode_sim, MuJoCo and cuRobo')
    p.add_argument('--robosuite-python',default=os.environ.get('ROBOSUITE_PYTHON',sys.executable),
                   help='Interpreter with robosuite and robocasa, used to build starting scenes')
    p.add_argument('--pilot-evidence',type=Path)
    p.add_argument('--outcome-evidence',type=Path, help='Completed drop-only salt retest; starts a fresh sweep')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    # Event history is separate from the replaceable visualization snapshot.
    sys.stdout=(args.output/'events.log').open('a', buffering=1)
    lock=(args.output/'.sweep.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('A sweep is already running in this output directory')
    if args.mode=='combined' and args.outcome_evidence:
        evidence=json.loads((args.outcome_evidence/'status.json').read_text())
        if evidence['status']!='complete' or not any(t['qualified'] for t in evidence['trials']):
            p.error('Outcome evidence must finish with a successful grasp')
    elif args.mode=='combined':
        if not args.pilot_evidence:
            p.error('Combined sweep requires completed native-object pilot evidence')
        pilot=json.loads((args.pilot_evidence/'summary.json').read_text())
        if pilot['attempts_per_object'] != args.attempts:
            p.error('Use the pilot attempt count when reusing its evidence')
        if pilot['status']!='complete' or pilot['completed_objects']!=3 or pilot['objects_with_qualified_grasps']<1:
            p.error('Native pilot must finish all three objects and validate at least one physical grasp before scaling')
    manifest=args.output/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text());objects=meta['objects']
        if meta['attempts']!=args.attempts or meta['mode']!=args.mode:
            p.error('Resume must use the original mode and attempt count')
    else:
        objects=[entry('robocasa',asset) for asset in PILOT] if args.mode=='pilot' else catalog(args.robosuite_python)
        hashes=controller_fingerprint()
        atomic_json(manifest,dict(mode=args.mode,attempts=args.attempts,objects=objects,controller_hashes=hashes, validation_policy='physical_lift_and_retention_v2', grasp_families=['top','oblique','side','oblique','side']))
    if args.mode == 'combined' and not args.outcome_evidence:
        # The pilot has already spent this object's five-attempt budget. Reuse
        # its immutable evidence rather than running five more trials on it.
        for obj in objects:
            source=args.pilot_evidence.resolve()/'objects'/obj['key']
            target=args.output/'objects'/obj['key']
            if (source/'status.json').is_file() and not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                target.symlink_to(source, target_is_directory=True)
    selected=[obj for obj in objects if not args.only or obj['key'] in args.only]
    missing=set(args.only)-{obj['key'] for obj in selected}
    if missing: p.error(f'Unknown asset keys: {sorted(missing)}')
    if args.retry_failed:
        stamp=time.strftime('%Y%m%d_%H%M%S')
        for obj in selected:
            folder=args.output/'objects'/obj['key']; path=folder/'status.json'
            if not path.exists(): continue
            record=json.loads(path.read_text())
            failed=[t for t in record['trials'] if not t.get('qualified')]
            if not failed and not record.get('setup_error'): continue
            history=folder/'history'/stamp;history.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,history/'status.json')
            for t in failed:
                for old in (folder/f"trial_{t['index']:03d}",folder/f"trial_{t['index']:03d}.log"):
                    if old.exists(): shutil.move(str(old),str(history/old.name))
            if record.get('setup_error') and (folder/'setup.log').exists():
                shutil.copy2(folder/'setup.log',history/'setup.log')
            record['trials']=[t for t in record['trials'] if t.get('qualified')]
            record.pop('setup_error',None);record.pop('stop_reason',None);record['complete']=False
            atomic_json(path,record)
    status(args.output,objects,args.attempts,'running')
    gpus=args.gpus.split(','); remaining=iter(selected)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        active={}
        for gpu in gpus:
            obj=next(remaining,None)
            if obj: active[pool.submit(run_object,obj,gpu,args)]=gpu
        while active:
            # Periodic index refresh reads only saved results, never reruns physics.
            finished,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
            status(args.output,objects,args.attempts,'running')
            for done in finished:
                gpu=active.pop(done);done.result()
                obj=next(remaining,None)
                if obj: active[pool.submit(run_object,obj,gpu,args)]=gpu
    status(args.output,objects,args.attempts,'stopped' if args.only else 'complete')
    print(json.dumps(dict(event='subset_finished' if args.only else 'sweep_finished',objects=[o['key'] for o in selected])),flush=True)


if __name__=='__main__': main()
