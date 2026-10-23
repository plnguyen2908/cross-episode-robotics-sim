#!/usr/bin/env python3
"""Sample and render base houses, independently of task variation sampling."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cross_episode_sim.generation.base_layout import sample_layout,build_scene,kitchen_bounds
from cross_episode_sim.generation.gpu_pool import lease_gpu
from cross_episode_sim.rendering_gpu import worker_environment,resolve_gpus


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,default=Path('runs/composite_pilot_iw5cmrpp/set_table_01_original/task_manifest.json'),help='Existing authored RoboCasa house whose kitchen/assets are retained')
    p.add_argument('--output',type=Path)
    p.add_argument('--count',type=int,default=3,help='Number of accepted base layouts')
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--sampler',choices=['cells','partition','growth'],default='cells')
    p.add_argument('--rooms-min',type=int,default=7)
    p.add_argument('--rooms-max',type=int,default=9)
    p.add_argument('--min-room-side',type=float,default=5.)
    p.add_argument('--loop-probability',type=float,default=.7)
    p.add_argument('--max-attempts',type=int,default=250)
    p.add_argument('--clearance-radius',type=float,default=.35,help='Static navigation footprint radius in metres; not swept-arm certification')
    p.add_argument('--gpu',default='auto')
    args=p.parse_args()
    if args.count<1 or not 4<=args.rooms_min<=args.rooms_max<=12 or args.clearance_radius<.3:
        p.error('Require count >= 1, 4 <= rooms-min <= rooms-max <= 12, clearance-radius >= .3')
    manifest=json.loads(args.source_manifest.read_text())
    source=Path(manifest['scene_xml'])
    if not source.is_file():p.error(f'Missing source scene: {source}')
    Path('runs').mkdir(exist_ok=True)
    out=args.output or Path(tempfile.mkdtemp(prefix='base_layout_samples_',dir='runs'))
    if args.output:out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT='+str(out.resolve()),flush=True)
    with lease_gpu(args.gpu) as gpu:
        os.environ.update(worker_environment(dict(os.environ,MUJOCO_GL='egl'),gpu,resolve_gpus([gpu])[gpu]))
        import mujoco
        import numpy as np
        from scipy.ndimage import label,distance_transform_edt
        from types import SimpleNamespace
        from PIL import Image,ImageDraw
        from tqdm import tqdm
        from cross_episode_sim.controller.house_scene import make_house
        from cross_episode_sim.manipulation.locomanip import RoboCasaLocomanip
        bounds=kitchen_bounds(source);rng=np.random.default_rng(args.seed)
        accepted=[];rejected=[]
        bar=tqdm(total=args.count,desc='Valid base layouts')
        for attempt in range(args.max_attempts):
            seed=int(rng.integers(0,2**31-1));count=int(rng.integers(args.rooms_min,args.rooms_max+1))
            folder=out/f'sample_{len(accepted)+1:03d}'
            try:
                if args.sampler=='growth':
                    from cross_episode_sim.generation.growth_layout import sample_growth_layout
                    layout=sample_growth_layout(seed,bounds,count,args.min_room_side,args.loop_probability)
                elif args.sampler=='partition':
                    from cross_episode_sim.generation.partition_layout import sample_partition_layout
                    layout=sample_partition_layout(seed,bounds,count,args.min_room_side,args.loop_probability)
                else:
                    layout=sample_layout(seed,bounds,count,args.min_room_side,args.loop_probability)
                folder.mkdir(exist_ok=True)
                build_scene(source,folder/'scene.xml',layout)
                model,data,_=make_house(folder/'scene.xml',[b['body'] for b in manifest['bindings']],manifest['robot_spawn'][:2],manifest['robot_spawn'][2],robot='franka_tidybot')
                from cross_episode_sim.generation.circulation import validate_furnished_model
                circulation=validate_furnished_model(model,data,layout)
                c=SimpleNamespace(model=model,data=data,bread_bids=set(),output=folder)
                nav=RoboCasaLocomanip._base_nav_map(c)
                clearance=distance_transform_edt(nav.occupancy)/50.
                labels,_=label(clearance>=args.clearance_radius)
                membership={}
                for room in layout['rooms']:
                    ids=[rid for rid,name in nav.room_ids_to_name.items() if name==room['floor_body'] or room['name']=='kitchen' and name=='floor_backing_room_main']
                    mask=np.isin(nav.room_map,ids)
                    components,areas=np.unique(labels[mask],return_counts=True)
                    membership[room['name']]=[int(v) for v,area in zip(components,areas) if v and area>=100]
                common=set(membership['kitchen'])
                for ids in membership.values():common &= set(ids)
                if not common:raise ValueError('Furnished footprint clearance disconnects rooms: '+str(membership))
                # Check actual robot initial collisions except intended floor contacts.
                bad=[]
                for contact in data.contact:
                    if contact.dist>=-.003:continue
                    names=[model.body(model.geom_bodyid[g]).name for g in contact.geom]
                    if any(n.startswith('robot_0/') for n in names) and not all(n.startswith('robot_0/') for n in names) and not any(n.startswith('floor_') for n in names):
                        bad.append(names)
                if bad:raise ValueError('Initial robot collision: '+str(bad[:3]))
                validation=dict(valid_base_layout=True,scope='partition + shared-wall topology + furnished static footprint clearance + initial robot contacts',
                                clearance_radius_m=args.clearance_radius,common_free_components=sorted(common),
                                room_local_circulation=circulation,furniture_door_sweep_envelope_checked=True,physical_task_rollout=False,appliance_swept_motion_validated=False,closed_door_operation_validated=False)
                (folder/'validation.json').write_text(json.dumps(validation,indent=2))
                model.vis.global_.offwidth=1600;model.vis.global_.offheight=1000
                opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.sitegroup[:]=0
                room_bounds=np.asarray([r['bounds'] for r in layout['rooms']]);low=room_bounds[:,:2].min(0);high=room_bounds[:,2:].max(0)
                cam=mujoco.MjvCamera();cam.lookat[:]=[*(low+high)/2,.4];cam.distance=float(max(high-low)*1.4+4);cam.azimuth=90
                with mujoco.Renderer(model,height=1000,width=1600) as renderer:
                    for filename,elevation in [('overview.png',-58),('top_view.png',-90)]:
                        cam.elevation=elevation;renderer.update_scene(data,camera=cam,scene_option=opt)
                        image=Image.fromarray(renderer.render())
                        draw=ImageDraw.Draw(image);draw.rectangle((0,0,1600,46),fill='#15202b')
                        draw.text((18,15),f'Seed {seed} | {count} rooms | {len(layout["kitchen_dining_paths"])} kitchen-to-dining routes | static clearance checked',fill='white')
                        image.save(folder/filename)
                # Annotated plan helps identify the single dining room and routes.
                plan=Image.new('RGB',(1100,850),'#f4f1eb');draw=ImageDraw.Draw(plan)
                scale=min(1000/(high[0]-low[0]),740/(high[1]-low[1]))
                def screen(x,y):return (50+(x-low[0])*scale,800-(y-low[1])*scale)
                for room in layout['rooms']:
                    x0,y0,x1,y1=room['bounds'];color={'kitchen':'#f0d0a3','dining':'#b3d8bf','living':'#bdd5ec','bedroom':'#d7c7e7'}[room['room_type']]
                    draw.polygon([screen(*point) for point in room['floor_polygon']],fill=color,outline='#393939',width=3)
                    draw.text(screen(*room['center']),room['name'].replace('_',' '),fill='black',anchor='mm')
                for edge in layout['connections']:
                    x,y=screen(*edge['center']);draw.ellipse((x-7,y-7,x+7,y+7),fill='#e98035')
                plan.save(folder/'plan.png')
                accepted.append(dict(seed=seed,room_count=count,directory=folder.name,paths=layout['kitchen_dining_paths']))
                bar.update();bar.set_postfix(attempts=attempt+1)
                print(f'Accepted {folder}: seed={seed}, rooms={count}',flush=True)
            except ValueError as exc:
                rejected.append(dict(seed=seed,room_count=count,reason=str(exc)))
            (out/'index.json').write_text(json.dumps(dict(stage='base_layout_before_task_variations',master_seed=args.seed,source_manifest=str(args.source_manifest.resolve()),accepted=accepted,rejected=rejected),indent=2))
            if len(accepted)==args.count:break
        bar.close()
        if len(accepted)!=args.count:raise RuntimeError(f'Only {len(accepted)}/{args.count} layouts accepted; see {out}/index.json')
    print('DONE '+str(out.resolve()),flush=True)


if __name__=='__main__':main()
