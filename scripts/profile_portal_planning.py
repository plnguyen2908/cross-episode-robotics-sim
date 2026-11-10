import json,time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np,mujoco
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.portal_crossing import PortalCrossing
from cross_episode_sim.navigation.open_aperture import observed_open_aperture
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.navigation.local_scan import ScanConfig
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.run;a=np.load(root/'door_exploration/wall_evidence.npz');g=ObservedNavigationMap(MapConfig(resolution=float(a['resolution']),height_resolution=float(a['resolution'])));g.evidence=a['evidence'];g.floor_views=a['floor'];g.origin=a['origin']
r=json.loads((root/'door_exploration/report.json').read_text());opening=r['opening_results'][0];aperture=observed_open_aperture(g,opening['doorway_center_map'],opening['portal_normal_map']);rows=[json.loads(x) for x in (root/'door_exploration/progress.jsonl').read_text().splitlines()];start=next(x for x in rows if x.get('status')=='refresh_open_portal' or x.get('status')=='cross_observed_portal');xy=start['estimated_base_xy'];yaw=start['estimated_base_yaw'];T=np.eye(4);T[:2,:2]=[[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]];T[:2,3]=xy
m=mujoco.MjModel.from_binary_path(str(root/'merged_recording/model.mjb'));geometry=RobotGeometry(m)
cross=PortalCrossing(g,geometry,aperture['center'],aperture['normal'],xy,transit_grid=g)
t=time.perf_counter();result=cross.step(NS(timestamp=0.,arm=NS(position=np.array(ScanConfig().stowed_arm))),NS(accepted=True,T_map_base=T),0.);dt=time.perf_counter()-t
out=dict(seconds=dt,result=result,events=cross.dock.events,note='Offline planning against final observed map; not original pre-crossing map or physical execution.')
args.output.write_text(json.dumps(out,indent=2));print(dt,result)
