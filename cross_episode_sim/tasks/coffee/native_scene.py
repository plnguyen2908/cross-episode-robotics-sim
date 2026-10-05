"""Author task 2 in the existing three-room house with a labelled hopper retrofit."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.tasks.breakfast.scene import SUPPORTS, OUTWARD, site_pose, bounds
from cross_episode_sim.fixtures.blender_lid import GENERATED, LID, handle_annotations
from cross_episode_sim.manipulation.physical_contents import cavity_geometry
from cross_episode_sim.tasks.coffee.grounds import HOPPER, add_hopper, add_grounds, grain_inventory, make_dosing_box, box_grasp_annotations
from cross_episode_sim.paths import DATA_DIR, source_digests, CONFIG_DIR

COFFEE='coffee_machine_main_group_main'
DEFAULT_CONFIG=CONFIG_DIR/'coffee_ground_brew_serve.json'


def prepare(output, config):
    container=config.get('dosing_container','mug')
    if container not in ('mug','small_box'):
        raise ValueError('Dosing container must be mug or small_box')
    box_cavity=None
    if not 0 < config['minimum_hopper_fraction'] <= 1:
        raise ValueError('Invalid required coffee capture fraction')
    if not 0 <= config['maximum_spill_fraction'] <= 1-config['minimum_hopper_fraction']+1e-9:
        raise ValueError('Invalid coffee spill limit')
    if not 0 < config['brew_seconds'] <= 120 or not 0 < config['pour_hold_seconds'] <= 5:
        raise ValueError('Invalid coffee timing')
    if not .025 <= config.get('pour_lip_clearance_m',.035) <= .06:
        raise ValueError('Pour lip clearance must be between 25 and 60 mm')
    angles=config['pour_angles_deg']
    if not angles or angles[0]!=0 or not 90<=angles[-1]<=135 or any(a>=b for a,b in zip(angles,angles[1:])):
        raise ValueError('Pour angles must increase from zero to 90–135 degrees')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    base=DATA_DIR/config['base_episode']
    old=json.loads((base/'task_manifest.json').read_text())
    tree=ET.parse(base/'robocasa_scene.xml');root=tree.getroot();world=root.find('worldbody')
    selected=[copy.deepcopy(next(i for i in old['bindings'] if i['role']==r)) for r in ('cup_one','cup_two')]
    keep={i['body'] for i in selected}
    for info in old['bindings']+old.get('background',[]):
        if info['body'] not in keep:
            node=world.find(f"body[@name='{info['body']}']")
            if node is not None:world.remove(node)
    for role,source,info in zip(('coffee_mug','grounds_cup'),('dining','living'),selected):
        for name in info.get('content_geoms',[]):
            for parent in root.iter('body'):
                for geom in list(parent.findall('geom')):
                    if geom.get('name')==name:parent.remove(geom)
        annotation=output/f'{role}_grasps.npz';shutil.copy2(info['grasp_path'],annotation)
        info.update(role=role,source=source,destination='kitchen',grasp_path=str(annotation),
                    content_geoms=[],task_object=True)
        if role=='grounds_cup':
            node=world.find(f"body[@name='{info['body']}']")
            if container=='small_box':
                box_cavity=make_dosing_box(node)
                box_grasp_annotations(annotation)
                info.update(asset='CoffeeDosingBox',key='custom__CoffeeDosingBox',
                            vessel_type='box',annotation_source='geometry_hypotheses',
                            evidence=[],contents='coffee grounds',
                            setup=dict(asset='CoffeeDosingBox',source='authored_primitives',
                                       model_xml=str(output/'robocasa_scene.xml'),object_scale=[1.,1.,1.],
                                       physics_executed=False,grasp_executed=False))
            original=Rotation.from_quat(np.fromstring(node.get('quat','1 0 0 0'),sep=' '),scalar_first=True)
            rotation=Rotation.from_euler('z',config.get('dosing_yaw_deg',180.),degrees=True)*original
            node.set('quat',' '.join(map(str,rotation.as_quat(scalar_first=True))))
    machine=world.find(f"body[@name='{COFFEE}']")
    if machine is None:raise ValueError('Source house has no native coffee machine')
    position=np.fromstring(machine.get('pos'),sep=' ');position[0]=config['machine_x']
    machine.set('pos',' '.join(map(str,position)))
    for site in machine.findall('site'):
        if 'coffee_liquid' in site.get('name',''):site.set('rgba','.35 .16 .06 0')
    hopper=add_hopper(world,config['hopper_xy'],.922,1.022)
    # Visible mounting/feed housing connects the added compartment to the native
    # appliance. Brewing is still a task-state surrogate; no internal grinder.
    mount=world.find(f"body[@name='{HOPPER}']")
    end=config['machine_x']-.10;start=config['hopper_xy'][0]+.062
    if end>start:
        for group in ('0','1'):
            ET.SubElement(mount,'geom',name='coffee_hopper_mount_'+group,type='box',
                pos=f'{(start+end)/2-config["hopper_xy"][0]} 0 .018',
                size=f'{(end-start)/2} .025 .016',rgba='.24 .25 .27 1',group=group,
                contype='1' if group=='0' else '0',conaffinity='1' if group=='0' else '0')
            # The appliance is farther back than the inlet: complete the
            # mounting elbow so the housing actually reaches its front tray.
            rear=position[1]-.12
            if rear>config['hopper_xy'][1]:
                ET.SubElement(mount,'geom',name='coffee_hopper_mount_elbow_'+group,type='box',
                    pos=f'{end-.02-config["hopper_xy"][0]} {(rear-config["hopper_xy"][1])/2} .018',
                    size=f'.02 {(rear-config["hopper_xy"][1])/2+.025} .016',
                    rgba='.24 .25 .27 1',group=group,
                    contype='1' if group=='0' else '0',conaffinity='1' if group=='0' else '0')
    native=ET.parse(GENERATED).getroot()
    lid=copy.deepcopy(native.find(f"worldbody/body[@name='{LID}']"))
    disk=lid.find("geom[@name='skill_blender_lid_g1']")
    disk_center=np.fromstring(disk.get('pos'),sep=' ')
    disk_half=np.fromstring(disk.get('size'),sep=' ')[1]
    anchor=np.r_[hopper['center_xy'],hopper['top_z']+disk_half+.001]-disk_center
    lid.set('pos',' '.join(map(str,anchor)));world.append(lid)
    root.find('asset').extend(copy.deepcopy(a) for a in native.find('asset')
                              if a.get('name','').startswith('skill_blender_lid_'))
    scene=output/'robocasa_scene.xml';tree.write(scene)
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    for info,fraction in zip(selected,(.55,.60)):
        point=site_pose(m,d,info['body'],info['source'],fraction,.05)
        world.find(f"body[@name='{info['body']}']").set('pos',' '.join(map(str,point)))
        info['position']=point
    tree.write(scene)
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    dose=selected[1];cavity=box_cavity if box_cavity is not None else cavity_geometry(m,d,dose['body'])
    grains=add_grounds(world,d.body(dose['body']).xpos,cavity,config['particle_count'],
                       d.body(dose['body']).xmat.reshape(3,3) if box_cavity is not None else None)
    # A native dispensing site is the horizontal target, not a guessed countertop spot.
    name=next(m.site(i).name for i in range(m.nsite) if m.site(i).name.startswith('coffee_machine_')
              and m.site(i).name.endswith('receptacle_place_site'))
    target=d.site(name).xpos.copy()
    from cross_episode_sim.manipulation.edge_access import object_bodies
    groups=m.geom_group.copy();m.geom_group[:]=5
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] in object_bodies(m,COFFEE) and (m.geom_contype[gid] or m.geom_conaffinity[gid]):m.geom_group[gid]=4
    ray=target.copy();ray[2]+=.02
    dist=mujoco.mj_ray(m,d,ray,np.array([0.,0.,-1.]),np.array([0,0,0,0,1,0],dtype=np.uint8),True,-1,None)
    m.geom_group[:]=groups
    if dist<0:raise ValueError('No physical dispenser tray below the native mug site')
    mug=selected[0];mug_low,_=bounds(m,d,mug['body'])
    tray_z=float(ray[2]-dist);target[2]=tray_z+.002-(mug_low[2]-d.body(mug['body']).xpos[2])
    mug['destination']='coffee_machine';mug['destination_position']=target.tolist()
    mug_cavity=cavity_geometry(m,d,mug['body'])
    # Visible coffee is explicitly a non-colliding brewing-state marker, unlike
    # the freely simulated grounds. Never advertise it as simulated liquid.
    visual_pos=d.body(mug['body']).xmat.reshape(3,3).T@np.r_[mug_cavity['center_xy'],mug_cavity['rim_z']-.02]
    ET.SubElement(world.find(f"body[@name='{mug['body']}']"),'site',name='task_coffee_surface',
                  type='cylinder',pos=' '.join(map(str,visual_pos)),
                  size=f'{mug_cavity["radius"]*.85} .001',rgba='.20 .08 .03 0',group='2')
    dose['destination_position']=[*hopper['center_xy'],hopper['top_z']+.12]
    annotation=handle_annotations(output)
    lid_info=dict(role='grounds_lid',body=LID,asset='Blender008_Lid reused as hopper lid',
        source='hopper',destination='kitchen',task_object=True,position=anchor.tolist(),
        destination_position=[*config['lid_parking_xy'],.94],grasp_path=str(annotation.resolve()),
        setup=dict(asset='Blender008_Lid',source='robocasa_fixture',model_xml=str(GENERATED.resolve()),object_scale=[1.,1.,1.]),
        recording=selected[0]['recording'],annotation_source='geometry_hypotheses')
    tree.write(scene)
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    mujoco.mj_step(m,d,nstep=1000)
    spec=dict(machine=COFFEE,hopper=hopper,dosing_body=dose['body'],dosing_cavity=cavity,
              grains=grains,lid_body=LID,lid_anchor=anchor.tolist(),lid_disk_center=disk_center.tolist(),
              dispenser_target=target.tolist(),tray_z=tray_z,**config)
    inv=grain_inventory(m,d,spec)
    if len(inv['vessel'])!=len(grains):raise ValueError(f'Initial grounds did not settle in dosing container: {inv}')
    for info in selected+[lid_info]:
        info['position']=d.body(info['body']).xpos.tolist()
        info['initial_quaternion']=d.body(info['body']).xquat.tolist()
    # Scene authoring ends here. Runtime may never write these free-body poses.
    for node in world.findall('body'):
        name=node.get('name')
        if name in keep|{LID}|set(grains):
            node.set('pos',' '.join(map(str,d.body(name).xpos)))
            node.set('quat',' '.join(map(str,d.body(name).xquat)))
    tree.write(scene)
    supports={**SUPPORTS,'hopper':HOPPER,'coffee_machine':COFFEE,
              'kitchen_right_counter':'counter_right_main_group_main'}
    manifest=dict(task='task_02_ground_coffee_brew_serve',seed=config['seed'],scene_xml=str(scene),
        bindings=selected+[lid_info],background=[],supports=supports,
        outward={**OUTWARD,'hopper':[0.,-1.],'coffee_machine':[0.,-1.],'kitchen_right_counter':[0.,-1.]},
        dynamic_fixtures=grains,robot_spawn=old['robot_spawn'],fixed_props=old.get('fixed_props',[]),
        coffee=spec,population_support_validation='pending',physical_validation='pending',
        instruction='Prepare coffee at the kitchen coffee machine and serve it at the office desk. '
        f'Remove the grounds lid. Fetch the {"small grounds box" if container=="small_box" else "dosing cup"} '
        'from the living room and pour its grounds into the coffee machine’s attached intake. '
        'Set the empty container on an open kitchen counter and replace the lid. '
        'Fetch the mug from the office, put it under the dispenser, press Start, wait for brewing, '
        'and return the coffee mug to the office desk. Finish with an empty gripper and the grounds lid closed.',
        simulation_contract=dict(grounds=f'{len(grains)} coarse free rigid granules; no particle pose resets during execution',
            hopper='custom side-mounted receiving compartment added to native RoboCasa coffee machine',
            brewing='contact-triggered task state and visual coffee, not fluid or thermal simulation'))
    # Check physical initial support for every task object, including hopper lid.
    for info in manifest['bindings']:
        own=object_bodies(m,info['body']);support=object_bodies(m,supports[info['source']])
        if not any((m.geom_bodyid[c.geom1] in own and m.geom_bodyid[c.geom2] in support)
                   or (m.geom_bodyid[c.geom2] in own and m.geom_bodyid[c.geom1] in support) for c in d.contact):
            raise ValueError(f'Initial object has no support contact: {info["role"]}')
    manifest['population_support_validation']='passed'
    from cross_episode_sim.tasks.coffee import grounds, native_task
    manifest['code_sha256']=source_digests(sys.modules[__name__],grounds,native_task)
    (output/'task_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'instruction.txt').write_text(manifest['instruction']+'\n')
    (output/'coffee_config.json').write_text(json.dumps(config,indent=2))
    return manifest
