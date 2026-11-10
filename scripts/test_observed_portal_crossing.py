"""Saved-state physical test of observed portal crossing; reset/scoring only are private."""
import argparse
from pathlib import Path
from dataclasses import replace
import numpy as np,json,mujoco,h5py
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import ShapedDocking,RobotGeometry
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--portals',type=int,nargs='+',default=[0,1]);parser.add_argument('--refresh-scan',action='store_true');parser.add_argument('--no-post-open-scan',action='store_true');parser.add_argument('--room-handoff',action='store_true');args=parser.parse_args()
p=args.source;m=mujoco.MjModel.from_binary_path(str(p/'state_recording/model.mjb'));d=mujoco.MjData(m)
with h5py.File(p/'state_recording/states.h5') as f:mujoco.mj_setState(m,d,f['states'][-1],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d)
s=np.load(p/'navigation_map.npz');g=ObservedNavigationMap(MapConfig(**json.loads((p/'summary.json').read_text())['config']));g.evidence=s['evidence'].copy();g.floor_views=s['floor_views'].copy();g.origin=s['origin_xy'].copy()
T=np.load(p/'slam/estimated_base_poses.npy')[-1];mapper=create_mapper(fuse_dense=False);control=OnboardBaseActuator(m,d);dock=ShapedDocking(g,RobotGeometry(m))

from cross_episode_sim.navigation.portal_crossing import PortalCrossing
from cross_episode_sim.recording.deferred import StateVideoRecorder
from scripts.demo_local_scan import ContactAudit
r=json.loads((p/'report.json').read_text());center=np.mean([r['portals'][i]['position'][:2] for i in args.portals],axis=0);normal=r['opening_results'][0]['portal_normal_map'][:2]
cross=PortalCrossing(g,RobotGeometry(m),center,normal,T[:2,3]);out=args.output;out.mkdir(parents=True,exist_ok=False);audit=ContactAudit(m);start=d.time
if args.no_post_open_scan:
 from cross_episode_sim.navigation.portal_crossing import OpenedPortalHandoff
 g.refresh_opened_portal(center,timestamp=float(d.time))
 g.inherited_support_poses=[np.r_[P[:2,3],np.arctan2(P[1,0],P[0,0])] for P in np.load(p/'slam/estimated_base_poses.npy')]
 cross=OpenedPortalHandoff(g,RobotGeometry(m),center,normal,after_timestamp=float(d.time))
with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,out/'state_recording') as recorder:
 transit_grid=g
 if args.refresh_scan:
  transit_grid.refresh_opened_portal(center,timestamp=float(d.time))
  stale_occupied=int((transit_grid.evidence>=2).sum())
  from cross_episode_sim.navigation.local_scan import LocalScanSkill
  from cross_episode_sim.navigation.portal_crossing import fresh_portal_map
  floor=np.load(p/'floor_observations.npz');g.record_floor_samples(floor['points'],floor['counts'])
  scanner=LocalScanSkill();g=fresh_portal_map(g,np.load(p/'slam/estimated_base_poses.npy'))
  for i in range(1501):
   if i%5==0:
    packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp);tracking=replace(tracking,T_map_base=T@tracking.T_map_base)
    if tracking.accepted:
     g.update(packet.cameras['navigation'],tracking);g.update_support(packet.cameras['wrist'],tracking,geometry=True)
     transit_grid.update(packet.cameras['navigation'],tracking)
     g.inherited_support_poses.append(np.r_[tracking.T_map_base[:2,3],np.arctan2(tracking.T_map_base[1,0],tracking.T_map_base[0,0])])
   decision=scanner.step(packet,tracking,float(d.time))
   if i%100==0:print('refresh',i*.02,decision.status,flush=True)
   if decision.complete:break
   if scanner.terminal:raise RuntimeError(decision.status)
   control.step(0.,decision.yaw_rps,diagnostic_observer=audit.observe);recorder.record('refresh_open_door')
  control.step(0.,0.,diagnostic_observer=audit.observe);recorder.record('refresh_complete')
  g.save(out/'refreshed_map')
  transit_grid.save(out/'refreshed_global_map')
  print(json.dumps(dict(stage='global_doorway_refresh',occupied_voxels_before=stale_occupied,occupied_voxels_after=int((transit_grid.evidence>=2).sum()))),flush=True)
  cross=PortalCrossing(g,RobotGeometry(m),center,normal,tracking.T_map_base[:2,3],transit_grid=transit_grid if args.room_handoff else None)
 for i in range(3000):
  if i%5==0:
   packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp);tracking=replace(tracking,T_map_base=T@tracking.T_map_base)
   if tracking.accepted:
    g.update(packet.cameras['navigation'],tracking)
    if args.no_post_open_scan:g.update_support(packet.cameras['wrist'],tracking,geometry=True)
  cmd=cross.step(packet,tracking,float(d.time))
  if i%50==0:print(i*.02,cmd,flush=True)
  if cmd['status']=='portal_crossed' or cross.follower.terminal:break
  control.step(cmd['forward'],cmd['yaw'],diagnostic_observer=audit.observe);recorder.record(cmd['status'])
 report=dict(status=cmd['status'],simulated_seconds=d.time-start,contacts=list(audit.events.values()),events=cross.follower.events)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
