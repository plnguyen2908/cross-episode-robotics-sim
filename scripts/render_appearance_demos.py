"""Render appearance variants on one previously validated house."""
import argparse,json,os,sys,tempfile,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cross_episode_sim.generation.appearance import PRESETS,apply_appearance
from cross_episode_sim.generation.gpu_pool import lease_gpu
from cross_episode_sim.rendering_gpu import worker_environment,resolve_gpus

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,default=Path('runs/growth_layout_clear_routes_52/sample_001'))
 p.add_argument('--manifest',type=Path,default=Path('runs/composite_pilot_iw5cmrpp/set_table_01_original/task_manifest.json'))
 p.add_argument('--output',type=Path);p.add_argument('--gpu',default='auto');args=p.parse_args()
 out=args.output or Path(tempfile.mkdtemp(prefix='appearance_demos_',dir='runs'))
 if args.output:out.mkdir(parents=True,exist_ok=False)
 print('OUTPUT='+str(out.resolve()),flush=True)
 m=json.loads(args.manifest.read_text());layout=json.loads((args.source/'base_layout.json').read_text())
 with lease_gpu(args.gpu) as gpu:
  os.environ.update(worker_environment(dict(os.environ,MUJOCO_GL='egl'),gpu,resolve_gpus([gpu])[gpu]))
  import mujoco,numpy as np
  from PIL import Image,ImageDraw
  from cross_episode_sim.controller.house_scene import make_house
  from cross_episode_sim.generation.circulation import validate_furnished_model
  reference=None;records=[]
  for theme in ['original',*PRESETS]:
   folder=out/theme;folder.mkdir();scene=folder/'scene.xml';shutil.copyfile(args.source/'scene.xml',scene)
   shutil.copyfile(args.source/'room_doors.json',folder/'room_doors.json')
   changes=apply_appearance(scene,{} if theme=='original' else {'preset':theme},m['bindings'])
   model,data,_=make_house(scene,[b['body'] for b in m['bindings']],m['robot_spawn'][:2],m['robot_spawn'][2],robot='franka_tidybot')
   fields=['body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_type','geom_contype','geom_conaffinity','geom_friction','jnt_pos','jnt_axis','jnt_range','qpos0','mesh_vert','mesh_face']
   signature={k:np.array(getattr(model,k),copy=True) for k in fields}
   if reference is None:reference=signature
   else:
    for k in fields:assert np.array_equal(reference[k],signature[k]),'Appearance changed physics: '+k
   checks=validate_furnished_model(model,data,layout)
   model.vis.global_.offwidth=1600;model.vis.global_.offheight=1000
   opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.sitegroup[:]=0
   bounds=np.asarray([r['bounds'] for r in layout['rooms']]);lo=bounds[:,:2].min(0);hi=bounds[:,2:].max(0)
   cam=mujoco.MjvCamera();cam.lookat[:]=[*(lo+hi)/2,.4];cam.distance=float(max(hi-lo)*1.4+4);cam.azimuth=90;cam.elevation=-58
   with mujoco.Renderer(model,height=1000,width=1600) as renderer:
    renderer.update_scene(data,camera=cam,scene_option=opt);Image.fromarray(renderer.render().copy()).save(folder/'overview.png')
   with mujoco.Renderer(model,height=600,width=800) as renderer:
    from types import SimpleNamespace
    from cross_episode_sim.robot.observations import render_onboard_observations,depth_rgb
    observations=render_onboard_observations(renderer,data,SimpleNamespace(namespace='robot_0/'),opt)
    np.save(folder/'navigation_depth_m.npy',observations['navigation_depth_m'])
    np.save(folder/'wrist_depth_m.npy',observations['wrist_depth_m'])
    frames=[Image.fromarray(observations['navigation_rgb']),Image.fromarray(depth_rgb(observations['navigation_depth_m'])),Image.fromarray(observations['wrist_rgb']),Image.fromarray(depth_rgb(observations['wrist_depth_m']))]
    panel=Image.new('RGB',(1600,1280),'#202831');draw=ImageDraw.Draw(panel)
    for i,frame in enumerate(frames):
     frame.save(folder/(['navigation_cam.png','navigation_depth.png','wrist_cam.png','wrist_depth.png'][i]))
     panel.paste(frame,(800*(i%2),640*(i//2)+40));draw.text((800*(i%2)+15,640*(i//2)+15),theme+' | '+['Navigation RGB','Navigation depth: 0-8 m','Wrist RGB','Wrist depth: 0-8 m'][i],fill='white')
    panel.save(folder/'onboard.png')
   record=dict(theme=theme,changes=changes,geometry_and_physics_equal_to_original=True,room_local_circulation=checks,source=str(args.source.resolve()),physical_rollout=False)
   (folder/'manifest.json').write_text(json.dumps(record,indent=2));records.append(record);print('Rendered '+theme,flush=True)
  (out/'index.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
