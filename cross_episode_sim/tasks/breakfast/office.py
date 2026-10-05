"""Seeded office-breakfast scenes using the qualified breakfast controller.

Variations are authored before physics, with explicit independent random streams.
These are independent demo episodes, not the persistent household/day runner.
"""
import argparse
import copy
import hashlib
import json
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.scene import (SUPPORTS, OUTWARD, allocate_sites, bounds,
                             support_bounds, settle_population)
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode, run
from cross_episode_sim.manipulation.edge_access import object_bodies
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.paths import DATA_DIR, source_digests, CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR/'office_breakfast.json'


def factor_rng(config, variant, factor):
    period = config['change_period_days'][factor]
    if type(period) is not int or period < 1:
        raise ValueError('Change periods must be positive integer days')
    epoch = variant['day']//period
    token = f"{config['seed']}:{factor}:{epoch}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], 'little')
    return np.random.default_rng(seed)


def validate_config(config):
    if config['people'] not in (1, 2):
        raise ValueError('Office breakfast supports one or two place settings')
    if not config['variants']:
        raise ValueError('At least one variant is required')
    names = [v['name'] for v in config['variants']]
    if len(names) != len(set(names)) or any(not n or Path(n).name != n or n in ('.','..') for n in names):
        raise ValueError('Variant names must be distinct directory names')
    for key in ('desk_translation_m', 'side_table_translation_m'):
        limits = np.asarray(config[key], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits < 0) or np.any(limits > .20):
            raise ValueError('Furniture translation bounds must be two values in [0, .20] m')
    for key, limit in [('object_yaw_jitter_deg', 10), ('source_fraction_jitter', .08),
                       ('destination_fraction_jitter', .04)]:
        if not 0 <= config[key] <= limit:
            raise ValueError(f'Invalid bounded variation: {key}')
    mode=config.get('source_position_sampling','preferred_jitter')
    if mode not in ('preferred_jitter','uniform_reachable_edge'):
        raise ValueError('Unknown source_position_sampling')
    if mode=='uniform_reachable_edge':
        for key,minimum,maximum in [('source_fraction_range',.05,.95),
                                    ('source_inset_range_m',.015,.12)]:
            limits=np.asarray(config[key],dtype=float)
            if limits.shape!=(2,) or not np.isfinite(limits).all() or not minimum<=limits[0]<limits[1]<=maximum:
                raise ValueError(f'Invalid reachable spawn bounds: {key}')
    lo, hi = config['clutter_count_per_room']
    if not (isinstance(lo,int) and isinstance(hi,int) and 0 <= lo <= hi <= 2):
        raise ValueError('The qualified base scene provides at most two clutter assets per room')
    for v in config['variants']:
        if type(v['day']) is not int or v['day'] < 0:
            raise ValueError('Day must be a nonnegative integer')
        for factor in ('objects', 'furniture', 'clutter'):
            factor_rng(config,v,factor)


def source_candidates(config,rng,count=512):
    """Unordered source sites; actual footprints/support decide acceptance."""
    return np.column_stack((rng.uniform(*config['source_fraction_range'],size=count),
                            rng.uniform(*config['source_inset_range_m'],size=count))).tolist()


def validate_furniture(model, data, names):
    """Reject moved furniture penetrating structural walls/door frames."""
    owners=set()
    for name in names:owners.update(object_bodies(model,name))
    moved=[g for g in range(model.ngeom) if model.geom_bodyid[g] in owners
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    walls=[g for g in range(model.ngeom)
           if (model.body(model.geom_bodyid[g]).name or '').startswith(('wall','doorframe_'))
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    for a in moved:
        for b in walls:
            distance=mujoco.mj_geomDistance(model,data,a,b,.01,None)
            if distance < -.001:
                raise ValueError(f'Furniture intersects structure: {model.geom(a).name} / {model.geom(b).name}')


def add_office_equipment(root, low, high):
    """A fixed monitor, keyboard and mouse on the rear desk strip, with collisions."""
    x=float((low[0]+high[0])/2);y=float(low[1]+.13);top=float(high[2])
    world=root.find('worldbody');names=[]
    def body(name, pos, pieces):
        node=ET.SubElement(world,'body',name=name,pos=' '.join(map(str,pos)))
        for i,(p,size,color) in enumerate(pieces):
            ET.SubElement(node,'geom',name=f'{name}_{i}',type='box',
                          pos=' '.join(map(str,p)),size=' '.join(map(str,size)),
                          rgba=' '.join(map(str,color)),group='1',contype='1',conaffinity='1')
        names.append(name)
    dark=(.09,.10,.12,1);screen=(.08,.27,.42,1)
    body('office_monitor',[x,y,top], [
        ([0,0,.008],[.13,.09,.008],dark),
        ([0,0,.11],[.018,.018,.10],dark),
        ([0,0,.27],[.24,.025,.145],dark),
        ([0,.026,.27],[.22,.002,.125],screen)])
    body('office_keyboard',[x,y+.23,top+.009],[([0,0,0],[.20,.065,.009],dark)])
    body('office_mouse',[x+.27,y+.23,top+.014],[([0,0,0],[.027,.045,.014],dark)])
    return names


def instruction(config):
    count=config['people']
    cleanup=('First clear the book from the office desk to the living-room side table, '
             'and return the misplaced living-room cup to the kitchen counter. ' if config['cleanup'] else '')
    return (f'Prepare breakfast for {count} {"person" if count==1 else "people"} at the office desk. '
            + cleanup + f'Arrange {count} separate place {"setting" if count==1 else "settings"}, '
            'each with one drinking cup and one bowl, in the clear front area of the desk. '
            'Bring syrup or honey from the kitchen and place it beside the setting'
            + ('s' if count>1 else '') + '. Keep suitable items already correctly placed; '
            'fetch only missing items. Leave the computer, keyboard and unrelated decorations '
            'where they are. Finish with an empty gripper.\n')


def prepare(config, variant, output):
    validate_config(config)
    output.mkdir(parents=True,exist_ok=False)
    base=Path(config['base_episode'])
    if not base.is_absolute():base=DATA_DIR/base
    manifest=copy.deepcopy(json.loads((base/'task_manifest.json').read_text()))
    tree=ET.parse(manifest['scene_xml']);root=tree.getroot();world=root.find('worldbody')
    gathering=config.get('workflow')=='gather_fill_serve'
    wanted=['cup_one','bowl_one']+(['cup_two','bowl_two'] if config['people']==2 else [])
    if not gathering:wanted+=['sweet_condiment']
    if gathering and (config['cleanup'] or variant['already_correct']):
        raise ValueError('Gather/fill uses empty vessels; cleanup and already_correct are not applicable')
    if config['cleanup']:wanted=['reading','misplaced_cup']+wanted
    if set(variant['already_correct'])-set(wanted):
        raise ValueError('Already-correct roles must be part of the selected task')
    inactive=[i for i in manifest['bindings'] if i['role'] not in wanted]
    for info in inactive:world.remove(world.find(f"body[@name='{info['body']}']"))
    infos=[i for i in manifest['bindings'] if i['role'] in wanted]
    if gathering:
        for info in infos:
            source=config['initial_sources'][info['role']]
            # Storage placement is authored after restoring its passive joints.
            info['source']=source if source in SUPPORTS else 'kitchen'
            info['initial_source']=source
            info['vessel_type']='cup' if info['role'].startswith('cup') else 'bowl'
    objects=factor_rng(config,variant,'objects');furniture=factor_rng(config,variant,'furniture')
    clutter=factor_rng(config,variant,'clutter')
    sampled={'objects':{},'furniture':{},'clutter':{},'already_correct':variant['already_correct']}
    # Preserve native fixture IDs and floor topology; 'dining' is the historical
    # room slot now presented as the office. Translate table and its chairs together.
    for room,key in [('dining','desk_translation_m'),('living','side_table_translation_m')]:
        limit=np.asarray(config[key]);delta=furniture.uniform(-limit,limit) if variant['furniture'] else np.zeros(2)
        moved=[]
        for body in world.findall('body'):
            name=body.get('name','')
            if name==SUPPORTS[room] or (room=='dining' and name.startswith('chair_') and 'dining_room' in name):
                pos=np.fromstring(body.get('pos','0 0 0'),sep=' ');pos[:2]+=delta
                body.set('pos',' '.join(map(str,pos)));moved.append(name)
        sampled['furniture'][room]={'translation_xy_m':delta.tolist(),'bodies':moved}
    background=[]
    for room in SUPPORTS:
        pool=[i for i in manifest['background'] if i['room']==room]
        if gathering:
            # Exact counts: no other vessels, including decorative cup assets.
            for info in pool:world.remove(world.find(f"body[@name='{info['body']}']"))
            sampled['clutter'][room]=[]
            continue
        count=int(clutter.integers(config['clutter_count_per_room'][0],config['clutter_count_per_room'][1]+1)) if variant['clutter'] else len(pool)
        selected=set(clutter.choice(len(pool),size=count,replace=False).tolist()) if count else set()
        for index,info in enumerate(pool):
            if index in selected:background.append(info)
            else:world.remove(world.find(f"body[@name='{info['body']}']"))
        sampled['clutter'][room]=[i['body'] for i in background if i['room']==room]
    for info in infos:
        # Each output owns its annotations. Native mesh paths remain pinned to
        # the same installed assets as the source scene.
        target=output/f"{info['role']}_grasps.npz";shutil.copy2(info['grasp_path'],target)
        info['grasp_path']=str(target.resolve())
        yaw=float(objects.uniform(-config['object_yaw_jitter_deg'],config['object_yaw_jitter_deg'])) if variant['objects'] else 0.
        body=world.find(f"body[@name='{info['body']}']")
        q=np.fromstring(body.get('quat','1 0 0 0'),sep=' ')
        q=(Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_quat(q,scalar_first=True)).as_quat(scalar_first=True)
        body.set('quat',' '.join(map(str,q)))
        info['orientation_policy']='seeded yaw about verified supported pose'
        sampled['objects'][info['role']]={'yaw_delta_deg':yaw}
    scene=output/'robocasa_scene.xml';tree.write(scene)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    validate_furniture(model,data,[name for f in sampled['furniture'].values() for name in f['bodies']])
    low,high=support_bounds(model,data,SUPPORTS['dining'])
    props=add_office_equipment(root,low,high);tree.write(scene)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    reserved={room:[] for room in SUPPORTS}
    reserved['dining']=[bounds(model,data,name) for name in props]
    source_sites={'reading':.25,'misplaced_cup':.75,'cup_one':.10,'bowl_one':.27,'cup_two':.44,'bowl_two':.61,'sweet_condiment':.78}
    destination_sites={'reading':.25,'misplaced_cup':.93,'cup_one':.10,'bowl_one':.28,'cup_two':.70,'bowl_two':.88,'sweet_condiment':.49}
    requests=[]
    for info in infos:
        role=info['role']
        if role in variant['already_correct']:info['source']=info['destination']
        for room_key,field,preferences,jitter in [
                ('source','position',source_sites,'source_fraction_jitter'),
                ('destination','destination_position',destination_sites,'destination_fraction_jitter')]:
            if role in variant['already_correct'] and field=='position':continue
            if (field=='position' and variant['objects'] and
                    config.get('source_position_sampling')=='uniform_reachable_edge'):
                sampled['objects'][role]['spawn_policy']='uniform_reachable_edge'
                requests.append(dict(body=info['body'],room=info[room_key],field=field,info=info,
                    candidates=source_candidates(config,objects),sample_record=sampled['objects'][role]))
                continue
            delta=float(objects.uniform(-config[jitter],config[jitter])) if variant['objects'] else 0.
            preferred=preferences[role]+delta
            sampled['objects'][role][field+'_preferred_fraction']=preferred
            requests.append(dict(body=info['body'],room=info[room_key],field=field,info=info,preferred=preferred))
        # Gather destinations are allocated across all kitchen surfaces after
        # storage authoring, with space for the hand and base approach. Do not
        # reserve a random main-counter gap here before comparing alternatives.
    if gathering:
        for request in requests:request['check_static']=True
    reserved=allocate_sites(model,data,requests,reserved)
    for info in infos:
        if gathering and info['source']=='kitchen':info['filling_position']=list(info['position'])
        if info['role'] in variant['already_correct']:info['position']=list(info['destination_position'])
        world.find(f"body[@name='{info['body']}']").set('pos',' '.join(map(str,info['position'])))
    # Reserve equipment before choosing clutter; use the same physical support
    # and footprint checks as the original task generator.
    allocate_sites(model,data,[dict(body=i['body'],room=i['room'],field='position',info=i,
        preferred=float(clutter.uniform(.1,.9)) if variant['clutter'] else (.15 if i['kind']=='pickable' else .85),inset=.40)
        for i in background],reserved)
    for info in background:world.find(f"body[@name='{info['body']}']").set('pos',' '.join(map(str,info['position'])))
    tree.write(scene)
    manifest.update(task='office_breakfast',seed=config['seed'],scene_xml=str(scene.resolve()),
        bindings=infos,background=background,room_labels={'kitchen':'kitchen','dining':'office','living':'living room'},
        fixed_props=props,robot_spawn=[float((low[0]+high[0])/2),float(high[1]+.80),float(-np.pi/2)],
        instruction=instruction(config),variation=copy.deepcopy(variant),
        variation_controls=copy.deepcopy(config),sampled_variation=sampled,
        source_episode=str(base),episode_relationship='independent seeded demo; not persistent daily history',
        physical_validation='pending',population_support_validation='pending')
    (output/'instruction.txt').write_text(manifest['instruction'])
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        settle_population(scene,infos,background)
    except Exception as exc:
        manifest.update(population_support_validation='failed',preparation_error=str(exc))
        (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2));raise
    manifest['population_support_validation']='passed'
    for info in infos:
        sampled['objects'][info['role']]['settled_position']=list(info['position'])
    from cross_episode_sim.tasks.breakfast import episode, scene
    manifest['code_sha256']=source_digests(sys.modules[__name__],scene,episode)
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


class OfficeBreakfastEpisode(BreakfastEpisode):
    video_filename='office_breakfast.mp4'

    def __init__(self,args,selection,manifest):
        super().__init__(args,selection,manifest)
        prop_ids={self.model.body(name).id for name in manifest['fixed_props']}
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] in prop_ids:
                self.model.geom_group[gid]=1
        self.report.update(scope='office breakfast with seeded scene variations',
                           variation=manifest['variation'],sampled_variation=manifest['sampled_variation'])

    def tick(self,seconds):
        self.review_phase=getattr(self,'review_phase','').replace('DINING','OFFICE')
        return super().tick(seconds)

    def describe_stage(self,stage):
        return super().describe_stage(stage).replace('dining table','office desk')

    def object_label(self):
        # Replay restores active_object per frame, but not annotation_asset.
        # Resolve the caption from the recorded object, not the last grasp.
        obj=getattr(self,'object_name',None)
        for info in self.manifest['bindings']:
            if info['body']==obj:
                return f"{info['role']} ({info['asset'].rsplit('/',1)[-1]})"
        return super().object_label()

    def transfer_role(self,role):
        if self.verify_role(role):
            self.record(task_role=role,goal_already_satisfied=True,skipped_transfer=True)
            return
        return super().transfer_role(role)

    def prepare_destination(self):
        # Static equipment must participate in placement-footprint exclusion,
        # in addition to the normal cuRobo/actual-mesh collision world.
        temporary=[dict(body=n,task_object=False) for n in self.manifest['fixed_props']]
        background=self.manifest['background']
        self.manifest['background']=background+temporary
        try:return super().prepare_destination()
        finally:self.manifest['background']=background

    def run_test(self,validate_only=False):
        steps=[dict(operation='validate_population',arguments={})]
        operations={'validate_population':Operation(self.validate_population,lambda:True),
                    'transfer':Operation(self.transfer_role,self.verify_role)}
        if not validate_only:
            steps += [dict(operation='transfer',arguments={'role':i['role']}) for i in self.manifest['bindings']]
        def goal():
            if validate_only:return {'population_and_empty_routes_valid':True}
            return {**{i['role']:self.verify_role(i['role']) for i in self.manifest['bindings']},
                    'empty_hand':not self.attached and not getattr(self, 'holding_loaf', False)}
        return 0 if CompositeEpisode(self,operations,goal).run('office_breakfast',steps) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',default='office_baseline')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--resume-prepared',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text());validate_config(config)
    variant=next(v for v in config['variants'] if v['name']==args.variant)
    output=args.output.resolve()
    manifest=json.loads((output/'task_manifest.json').read_text()) if args.resume_prepared else prepare(config,variant,output)
    if args.resume_prepared and (output/'report.json').exists():
        raise FileExistsError('Refusing to rerun an episode with existing physics results')
    if args.prepare_only:
        print(json.dumps({'prepared':str(output),'population':manifest['population_support_validation']}));return 0
    if manifest.get('population_support_validation')!='passed':
        raise ValueError('Only a scene with passed population validation can execute')
    code=run(output,manifest,validate_only=args.validate_only,episode_class=OfficeBreakfastEpisode)
    manifest['physical_validation']=('routes_only_passed' if args.validate_only else 'passed') if code==0 else 'failed'
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    return code


if __name__=='__main__':raise SystemExit(main())
