"""One continuous gather -> simulated-human fill -> serve breakfast episode.

The controller is an oracle harness. New episodes use free rigid food objects;
legacy recordings retain their explicitly labelled visual proxy mode.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.scene import SUPPORTS, OUTWARD, bounds, support_bounds
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode, run
from cross_episode_sim.tasks.breakfast.office import OfficeBreakfastEpisode, prepare as prepare_office
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.paths import source_digests, CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR/'breakfast_gather_fill_serve.json'


def vessel_roles(people):
    if type(people) is not int or people not in (1, 2):
        raise ValueError('This qualified scene supports one or two breakfasts')
    return ['cup_one', 'bowl_one'] + (['cup_two', 'bowl_two'] if people == 2 else [])


def validate_gather_config(config):
    roles=vessel_roles(config['people'])
    if config.get('workflow')!='gather_fill_serve':
        raise ValueError('Expected gather_fill_serve workflow')
    if set(config['initial_sources'])!=set(roles):
        raise ValueError('Specify exactly one initial source for each requested vessel')
    if any(s not in ('kitchen','dining','living','drawer','cabinet') for s in config['initial_sources'].values()):
        raise ValueError('Unknown initial source')
    if not 0 < config['human_wait_seconds'] <= 60:
        raise ValueError('Human wait must be in (0,60] simulated seconds')
    if config['content_representation'] not in ('attached_visual_proxy', 'physical_objects'):
        raise ValueError('Unknown breakfast content representation')
    if not 0 < config['spill_grace_seconds'] <= 2:
        raise ValueError('Invalid spill observation debounce')
    if config['content_representation']=='attached_visual_proxy' and not 20 <= config['spill_tilt_deg'] < 90:
        raise ValueError('Invalid visual-content tilt rule')


def instruction(people, physical=False, place='office desk'):
    return (f'Prepare breakfast for {people} people at the {place}. '
        f'There are exactly {people} cups and {people} bowls in the household. '
        'Find the empty vessels and gather them on clear kitchen work surfaces: '
        'counters, islands, kitchen tables, or a switched-off stovetop. '
        f'Open and close storage as needed. An empty vessel already at the {place.split()[-1]} '
        f'must come back for filling. Withdraw, leave your gripper empty, and wait '
        + ('for the human to put small food pieces in the bowls and small edible pieces in the cups. '
           if physical else 'for the human to fill every bowl with food and every cup with a drink. ') +
        'After observing the filled contents, carry those same vessels to the '
        f'{place} and arrange one cup and one bowl per person. Finish with '
        'the contents retained, storage closed, and your gripper empty.\n')


def add_contents(scene, manifest):
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    tree=ET.parse(scene)
    for info in manifest['bindings']:
        node=tree.find(f".//body[@name='{info['body']}']")
        lo,hi=bounds(model,data,info['body'])
        center=(lo+hi)/2;center[2]=hi[2]-.012
        rotation=data.body(info['body']).xmat.reshape(3,3)
        local=rotation.T@(center-data.body(info['body']).xpos)
        radius=float(min(hi[0]-lo[0],hi[1]-lo[1])*.27)
        names=[]
        pieces=[(0.,0.,0.)] if info['vessel_type']=='cup' else [
            (radius*.6*np.cos(a),radius*.6*np.sin(a),0.) for a in np.linspace(0,2*np.pi,7,endpoint=False)]
        for i,offset in enumerate(pieces):
            name=f"breakfast_contents_{info['role']}_{i}";names.append(name)
            cup=info['vessel_type']=='cup'
            ET.SubElement(node,'geom',name=name,type='cylinder' if cup else 'ellipsoid',
                pos=' '.join(map(str,local+offset)),
                size=f'{radius} .003' if cup else f'{radius*.38} {radius*.32} .006',
                rgba='0.3 0.13 0.04 0' if cup else '0.92 0.65 0.22 0',
                contype='0',conaffinity='0',mass='0',group='1')
        info['content_geoms']=names;info['contents']='empty'
    tree.write(scene)


def prepare(config, output):
    validate_gather_config(config)
    variant=config['variants'][0]
    manifest=prepare_office(config,variant,output)
    # Keep all phase targets, instead of mutating final goals during gathering.
    for info in manifest['bindings']:
        info['serving_position']=list(info['destination_position'])
        info['serving_room']='dining'
    physical = config['content_representation'] == 'physical_objects'
    place='office desk' if config.get('table_setting','office')=='office' else 'dining table'
    manifest.update(task='breakfast_gather_fill_serve',instruction=instruction(config['people'], physical, place),
        gathering_config=copy.deepcopy(config),supports=dict(SUPPORTS),outward=dict(OUTWARD),
        dynamic_fixtures=[],storage={},human_events=[],
        content_model=dict(representation='attached_visual_proxy',physical_food=False,
            spill_tilt_deg=config.get('spill_tilt_deg',55.),spill_grace_seconds=config['spill_grace_seconds']))
    from cross_episode_sim.tasks.breakfast.storage import prepare_storage
    try:
        prepare_storage(manifest)
        from cross_episode_sim.manipulation.kitchen_surfaces import discover_filling_surfaces, allocate_filling_targets
        model=mujoco.MjModel.from_xml_path(manifest['scene_xml']);data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        surfaces=discover_filling_surfaces(model,data,manifest['supports']['kitchen'])
        allocate_filling_targets(model,data,manifest,surfaces)
        for site,surface in surfaces.items():
            manifest['supports'][site]=surface['body']
            manifest['outward'][site]=surface['outward']
        if physical:
            from cross_episode_sim.manipulation.physical_contents import add_physical_contents
            add_physical_contents(Path(manifest['scene_xml']),manifest)
        else:
            add_contents(Path(manifest['scene_xml']),manifest)
    except Exception as exc:
        manifest.update(population_support_validation='failed',preparation_error=str(exc))
        (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
        raise
    manifest['physical_validation']='pending'
    from cross_episode_sim.manipulation import food_catalog, kitchen_surfaces, physical_contents
    from cross_episode_sim.tasks.breakfast import episode, storage
    manifest.setdefault('code_sha256',{}).update(source_digests(
        sys.modules[__name__],storage,episode,kitchen_surfaces,physical_contents,food_catalog))
    (output/'gather_config.json').write_text(json.dumps(config,indent=2))
    (output/'instruction.txt').write_text(manifest['instruction'])
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def content_state_at(events, time, roles):
    states={r:'empty' for r in roles}
    for event in events:
        if event['time']<=time+1e-8:
            states.update(event['states'])
    return states


class GatherBreakfast(OfficeBreakfastEpisode):
    video_filename='breakfast_gather_fill_serve.mp4'

    def __init__(self,args,selection,manifest):
        self.content_states={i['role']:'empty' for i in manifest['bindings']}
        self.content_events=[];self._tilt_since={};self.phase='INITIALIZE'
        self.physical_content_items=manifest.get('physical_contents', [])
        args.dynamic_objects=list(dict.fromkeys([*args.dynamic_objects,
            *(item['body'] for item in self.physical_content_items)]))
        super().__init__(args,selection,manifest)
        from cross_episode_sim.manipulation.kitchen_surfaces import register_filling_surfaces
        register_filling_surfaces(self)
        if self.physical_content_items:
            from cross_episode_sim.manipulation.physical_contents import initialize_contents
            initialize_contents(self)
        self.content_gids={i['role']:[self.model.geom(n).id for n in i['content_geoms']]
                           for i in manifest['bindings']}
        self._content_up={i['role']:Rotation.from_quat(i['initial_quaternion'],scalar_first=True).as_matrix().T[:,2]
                          for i in manifest['bindings']}
        self._next_content_check=0.
        self.report.update(scope='gather empty vessels, human fill, serve in office',
            human_events=[],content_events=self.content_events,content_model=manifest['content_model'],
            inventory_source='private simulator oracle; no VLM perception claim')

    def contents_visible(self, states):
        if getattr(self,'physical_content_items',[]):
            # Replay records every free body's real pose, including spills.
            return
        for role,gids in self.content_gids.items():
            self.model.geom_rgba[gids,3]=1. if states[role]=='filled' else 0.

    def render_video_frame(self,stage,time):
        self.contents_visible(content_state_at(self.content_events,time,self.content_states))
        return super().render_video_frame(stage,time)

    def tick(self,seconds):
        super().tick(seconds)
        if not hasattr(self,'content_gids'):return
        # Evaluate every recorded sample, not just the end of a long motion.
        self.contents_visible(self.content_states)

    def before_step(self):
        super().before_step()
        if not hasattr(self,'content_gids'):return
        if self.data.time < self._next_content_check:return
        self._next_content_check=float(self.data.time)+.025
        config=self.manifest['gathering_config']
        if getattr(self,'physical_content_items',[]):
            from cross_episode_sim.manipulation.physical_contents import retained
            for item in self.physical_content_items:
                role=item['role']
                if self.content_states[role]!='filled':continue
                info=next(i for i in self.manifest['bindings'] if i['role']==role)
                if retained(self.model,self.data,item,info['body']):
                    self._tilt_since.pop(role,None);continue
                since=self._tilt_since.setdefault(role,float(self.data.time))
                if self.data.time-since>=config['spill_grace_seconds']:
                    self.content_states[role]='spilled'
                    self.content_events.append(dict(time=float(self.data.time),event='physical_contents_spilled',
                        states={role:'spilled'},object=item['body']))
                    raise RuntimeError(f'Physical contents left vessel: {role}/{item["asset"]}')
            return
        for info in self.manifest['bindings']:
            role=info['role']
            if self.content_states[role]!='filled':continue
            up=float(self.data.body(info['body']).xmat.reshape(3,3)[2]@self._content_up[role])
            if up>=np.cos(np.radians(config['spill_tilt_deg'])):
                self._tilt_since.pop(role,None);continue
            since=self._tilt_since.setdefault(role,float(self.data.time))
            if self.data.time-since>=config['spill_grace_seconds']:
                self.content_states[role]='spilled'
                self.content_events.append(dict(time=float(self.data.time),event='visual_contents_spilled',states={role:'spilled'}))
                self.contents_visible(self.content_states)
                raise RuntimeError(f'Contents spilled under declared tilt rule: {role}')

    def current_room(self, info):
        support=self.assignment(objects=[info['body']])[info['body']]
        return next(room for room,body in self.task_supports.items() if body==support)

    def is_filling_site(self, site):
        if site not in getattr(self, 'filling_sites', ('kitchen',)):
            return False
        surface = getattr(self, 'filling_surface_info', {}).get(site, {})
        if surface.get('kind') in ('stove', 'cooktop'):
            from cross_episode_sim.manipulation.kitchen_surfaces import stove_is_off
            return stove_is_off(self.model, self.data, surface['body'])
        return True

    def prepare_destination(self):
        site = self.object_info[self.object_name]['destination']
        if site in getattr(self, 'filling_sites', ()) and not self.is_filling_site(site):
            raise RuntimeError('Stovetop filling requires all burners off')
        return super().prepare_destination()

    def configure_phase(self, phase):
        self.phase=phase;self.report['breakfast_phase']=phase
        self.report['executed_placement_targets']={}
        for info in self.manifest['bindings']:
            info['source']=self.current_room(info)
            info['destination']=info.get('filling_site','kitchen') if phase=='GATHER' else info['serving_room']
            info['destination_position']=list(info['filling_position'] if phase=='GATHER' else info['serving_position'])
        self.record(breakfast_phase=phase,inventory={i['role']:dict(location=i['source'],contents=self.content_states[i['role']]) for i in self.manifest['bindings']})

    def phase_ready(self, phase):
        return self.phase==phase

    def validate_population(self):
        self.tick(2.)
        expected=vessel_roles(self.manifest['gathering_config']['people'])
        if sorted(self.content_states)!=sorted(expected) or len(self.objects)!=len(expected):
            raise RuntimeError('Scene does not contain exactly the requested vessels')
        for info in self.manifest['bindings']:
            if self.current_room(info)!=info['initial_source']:
                raise RuntimeError(f"Incorrect initial support: {info['role']}")
        from cross_episode_sim.tasks.breakfast.storage import storage_closed
        if not storage_closed(self):raise RuntimeError('Breakfast storage must start closed')
        self.report['population_validation']=dict(exact_vessel_count=len(expected),supported=True,
            full_episode_success=False)
        self.tuck_for_navigation()

    def transfer_role(self, role):
        info=next(i for i in self.manifest['bindings'] if i['role']==role)
        room=self.current_room(info);info['source']=room
        if self.phase=='SERVE' and self.content_states[role]!='filled':
            raise RuntimeError(f'Cannot serve an empty or spilled vessel: {role}')
        if self.phase=='GATHER' and self.is_filling_site(room):
            # Any supported, separated counter site is a valid filling station.
            info['filling_position']=self.data.body(info['body']).xpos.tolist()
            info['destination_position']=list(info['filling_position'])
            info['filling_site']=room;info['destination']=room
            self.record(task_role=role,skipped_transfer=True,reason='already at kitchen filling surface')
            return
        if self.phase=='SERVE' and self.verify_role(role):return
        from cross_episode_sim.tasks.breakfast.storage import retrieve_from_storage
        if room in self.manifest['storage']:
            retrieve_from_storage(self,info)
        else:
            BreakfastEpisode.transfer_role(self,role)

    def verify_role(self,role):
        if self.phase=='SERVE' and self.content_states[role]!='filled':return False
        if self.phase=='SERVE' and getattr(self,'physical_content_items',[]):
            from cross_episode_sim.manipulation.physical_contents import contents_supported
            if not all(contents_supported(self,item) for item in self.physical_content_items if item['role']==role):
                return False
        return BreakfastEpisode.verify_role(self,role)

    def all_at_counter(self):
        return (not self.attached and not getattr(self,'holding_loaf',False)
            and all(self.is_filling_site(self.current_room(i)) for i in self.manifest['bindings']))

    def human_fill(self):
        if self.phase!='GATHER' or not self.all_at_counter():
            raise RuntimeError('Human filling requires all vessels on kitchen filling surfaces and an empty hand')
        self.tuck_for_navigation()
        base=self.base_pose()
        self.task_navigate(base[:2]+np.array([0.,-.25]),False,face=float(base[2]))
        self.review_phase='WAIT_FOR_HUMAN: FILL BOWLS AND CUPS'
        self.stage='wait for human filling'
        self.record(human_wait_started=True)
        self.tick(self.manifest['gathering_config']['human_wait_seconds'])
        if not self.all_at_counter() or not self.in_default_travel_posture(False):
            raise RuntimeError('Filling station changed or robot is not safely tucked')
        if getattr(self,'physical_content_items',[]):
            return self.fill_physical_contents()
        event=dict(time=float(self.data.time),actor='simulated_human',event='fill_vessels',
            representation='attached_visual_proxy',states={r:'filled' for r in self.content_states},
            vessels=[dict(role=i['role'],body=i['body'],before='empty',after='filled',
                content='drink' if i['vessel_type']=='cup' else 'food',
                vessel_position=self.data.body(i['body']).xpos.tolist(),geoms=i['content_geoms'])
                for i in self.manifest['bindings']])
        self.content_states.update(event['states']);self.content_events.append(event)
        self.report['human_events'].append(event)
        (self.output/'human_events.json').write_text(json.dumps(self.report['human_events'],indent=2))
        self.contents_visible(self.content_states)
        self.record(human_event=event)
        self.tick(1.)

    def fill_physical_contents(self):
        from cross_episode_sim.manipulation.physical_contents import activate_contents, contents_supported
        if any(state != 'empty' for state in self.content_states.values()):
            raise RuntimeError('Physical filling is a one-time human event for empty vessels')
        activate_contents(self)
        self.content_states.update({role:'filling' for role in self.content_states})
        event=dict(time=float(self.data.time),actor='simulated_human',event='fill_vessels',
            representation='physical_objects', states=dict(self.content_states),
            vessels=[dict(role=item['role'],object=item['body'],asset=item['asset'],
                object_scale=item['object_scale'],mass_kg=item['mass_kg'],rescaled=False)
                for item in self.physical_content_items])
        self.content_events.append(event);self.report['human_events'].append(event)
        self.record(human_event=event)
        self.stage='settle physical food in vessels';self.tick(2.)
        support={item['role']:contents_supported(self,item) for item in self.physical_content_items}
        self.report['physical_fill_support']=support
        if not all(support.values()):
            raise RuntimeError(f'Physical filling did not settle inside vessels: {support}')
        self.content_states.update({role:'filled' for role in self.content_states})
        self.content_events.append(dict(time=float(self.data.time),event='physical_contents_settled',
                                       states=dict(self.content_states)))
        (self.output/'human_events.json').write_text(json.dumps(self.report['human_events'],indent=2))
        self.record(physical_food_settled=support)

    def kitchen_world_geoms(self):
        from cross_episode_sim.manipulation.physical_contents import active_content_bodies
        carried=active_content_bodies(self)
        return [gid for gid in super().kitchen_world_geoms() if self.model.geom_bodyid[gid] not in carried]

    def loaded_payload_vertices(self):
        from cross_episode_sim.manipulation.physical_contents import active_content_bodies
        from cross_episode_sim.manipulation.edge_access import collision_vertices
        carried=active_content_bodies(self)
        vertices=self.bread_vertices()
        return np.concatenate([vertices,collision_vertices(self.model,self.data,carried)]) if carried else vertices

    def loaded_payload_bodies(self):
        from cross_episode_sim.manipulation.physical_contents import active_content_bodies
        return self.bread_bids | active_content_bodies(self)

    def tuck_loaded_for_navigation(self):
        # Rim grips need the same orientation-preserving carry while empty.
        # Filling changes the contents, not the stability of that grasp.
        return self.tuck_vessel_for_navigation()

    def project_payload_contents(self, probe, reference=None):
        """Project contained objects into a scratch state only, never live physics."""
        from cross_episode_sim.manipulation.physical_contents import active_content_bodies
        reference = self.data if reference is None else reference
        if probe is reference:return
        contents=active_content_bodies(self)
        if not contents:return
        live=reference.body(self.object_name);future=probe.body(self.object_name)
        rotation=future.xmat.reshape(3,3) @ live.xmat.reshape(3,3).T
        for item in self.physical_content_items:
            if self.model.body(item['body']).id not in contents:continue
            food=reference.body(item['body']);joint=probe.joint(item['joint'])
            joint.qpos[:3]=future.xpos+rotation@(food.xpos-live.xpos)
            joint.qpos[3:]=Rotation.from_matrix(rotation@food.xmat.reshape(3,3)).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model,probe)

    def navigation_penetration(self,data,carrying):
        from cross_episode_sim.manipulation.physical_contents import active_content_bodies
        contents=active_content_bodies(self)
        if not contents:return super().navigation_penetration(data,carrying)
        if carrying and data is not self.data:
            # Future route probes rigidly project the currently contained load.
            # Live physics is never constrained to the vessel this way.
            self.project_payload_contents(data)
        robot, _, object_mask = self.contact_body_masks()
        payload = object_mask.copy()
        payload[list(contents)] = True
        contacts = data.contact
        a = self.model.geom_bodyid[contacts.geom1]
        b = self.model.geom_bodyid[contacts.geom2]
        ra, rb, pa, pb = robot[a], robot[b], payload[a], payload[b]
        ma, mb = ra | (carrying & pa), rb | (carrying & pb)
        selected = ((ma & ~rb & ~pb) | (mb & ~ra & ~pa))
        selected &= ~(self._contact_floor_mask[a] | self._contact_floor_mask[b])
        return float(np.max(.001 - contacts.dist[selected], initial=0.))

    def filled(self):
        return bool(self.report['human_events']) and all(v=='filled' for v in self.content_states.values())

    def final_goal(self):
        from cross_episode_sim.tasks.breakfast.storage import storage_closed
        return {**{i['role']:self.verify_role(i['role']) for i in self.manifest['bindings']},
            'human_fill_event_completed':self.filled(),'storage_closed':storage_closed(self),
            'empty_hand':not self.attached and not getattr(self,'holding_loaf',False)}

    def run_test(self,validate_only=False):
        self.report.update(validation_only=bool(validate_only),full_task_requested=not validate_only)
        ops={'validate_population':Operation(self.validate_population,lambda:True),
             'phase':Operation(self.configure_phase,self.phase_ready),
             'transfer':Operation(self.transfer_role,self.verify_role),
             'human_fill':Operation(self.human_fill,self.filled)}
        steps=[dict(operation='validate_population',arguments={})]
        if not validate_only:
            for phase in ('GATHER','SERVE'):
                steps.append(dict(operation='phase',arguments={'phase':phase}))
                steps.extend(dict(operation='transfer',arguments={'role':i['role']}) for i in self.manifest['bindings'])
                if phase=='GATHER':steps.append(dict(operation='human_fill',arguments={}))
        goal=(lambda:dict(initial_population_valid=True)) if validate_only else self.final_goal
        return 0 if CompositeEpisode(self,ops,goal).run('breakfast_gather_fill_serve',steps) else 1


def prepare_episode(output, seed=None, people=None, sources=None, config_path=DEFAULT_CONFIG,
                    randomize=False, floor_objects=0, episode_label=None, table_objects=0, table_setting=None):
    """Author and settle a breakfast episode in `output`; returns its manifest.

    `sources='storage'` starts one mug in a drawer and one bowl in a cabinet.
    `randomize` also moves the desk and side table and scatters table clutter;
    `floor_objects` adds that many loose objects on the floor (rooms in turn).
    """
    config=json.loads(Path(config_path).read_text())
    if table_setting:config['table_setting']=table_setting
    if randomize:
        config['variants'][0].update(objects=True,furniture=True,clutter=True)
        config['clutter_count_per_room']=[1,2]
    if people is not None:config['people']=people
    if seed is not None:config['seed']=seed
    roles=vessel_roles(config['people'])
    config['initial_sources']={r:config['initial_sources'][r] for r in roles}
    if sources=='storage':
        config['initial_sources'].update(cup_one='drawer',bowl_one='cabinet')
    config['storage_profile']=sources or config.get('storage_profile','rooms')
    manifest=prepare(config,Path(output).resolve())
    if floor_objects or episode_label or table_objects:
        from cross_episode_sim.tasks.floor_objects import add_floor_objects, scatter_objects
        scene=Path(manifest['scene_xml'])
        if table_objects:
            # Tabletop objects on the dining and side tables, clear of every
            # vessel's start and its place setting.
            serving=[i['serving_position'][:2] for i in manifest['bindings']]+[i['position'][:2] for i in manifest['bindings']]
            tops=[dict(room=r,support=manifest['supports'][r]) for r in ('dining','living')]
            manifest['table_objects']=scatter_objects(scene,(tops*table_objects)[:table_objects],config['seed']+1,
                                                      avoid_xy=serving,avoid_radius=.22,prefix='table')
        if floor_objects:
            rooms=[('kitchen','dining','living')[i%3] for i in range(floor_objects)]
            keep=[manifest['robot_spawn'][:2]]+[i['position'][:2] for i in manifest['bindings']]
            manifest['floor_objects']=add_floor_objects(scene,rooms,config['seed'],avoid_xy=keep)
        if episode_label:manifest['episode_label']=episode_label
        (Path(output).resolve()/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--people',type=int,choices=(1,2))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--sources',choices=('rooms','storage'),default=None)
    parser.add_argument('--randomize',action='store_true',help='Also move the desk and side table and scatter table clutter')
    parser.add_argument('--floor-objects',type=int,default=0,help='Loose objects on the floor, one room after another')
    parser.add_argument('--table-objects',type=int,default=0,help='Loose objects on the dining and side tables')
    parser.add_argument('--table-setting',choices=('office','dining'),help='office: fixed monitor, keyboard and mouse on the desk; dining: none')
    parser.add_argument('--episode-label',help='Banner shown in the videos, e.g. "History 1 (given): breakfast"')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    manifest=prepare_episode(output,args.seed,args.people,args.sources,args.config,
                             args.randomize,args.floor_objects,args.episode_label,args.table_objects,args.table_setting)
    print(json.dumps({'prepared':str(output),'initial_sources':{i['role']:i['initial_source'] for i in manifest['bindings']},
        'content_model':manifest['content_model']}),flush=True)
    if args.prepare_only:return 0
    result=run(output,manifest,args.validate_only,episode_class=GatherBreakfast)
    authored=json.loads((output/'task_manifest.json').read_text())
    authored['physical_validation']=('initialization_only_passed' if args.validate_only else 'passed') if result==0 else 'failed'
    (output/'task_manifest.json').write_text(json.dumps(authored,indent=2))
    return result


if __name__=='__main__':raise SystemExit(main())
