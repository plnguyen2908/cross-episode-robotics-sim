"""Recorded-state test of mapped destination -> local dock -> physical placement, with live-pose retries.

Reset/scoring are private. The landmark is the earlier exploration observation;
planning consumes only that semantic memory, RGB-D, SLAM and proprioception.
"""
import argparse,json,sys,os
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.runtime.logging import TimestampedStream


def main():
 sys.stdout=TimestampedStream(sys.stdout);sys.stderr=TimestampedStream(sys.stderr)
 p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--transfer',type=int,default=6);p.add_argument('--observe-only',action='store_true')
 p.add_argument('--qwen-docking',action='store_true');p.add_argument('--retries',type=int,default=0);p.add_argument('--max-candidates',type=int,default=0);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 rec=a.episode/f'transfer_{a.transfer:03d}/carry/state_recording';pick=a.episode/f'transfer_{a.transfer-1:03d}/pick'
 m=mujoco.MjModel.from_binary_path(str(rec/'model.mjb'));d=mujoco.MjData(m)
 with h5py.File(rec/'states.h5') as f:state=f['states'][-1]
 mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
 from cross_episode_sim.evaluation.qwen_component_score import ComponentScore
 score=ComponentScore(m,d);runtime=EpisodeRuntime(a.output);runtime.bind(m,d)
 runtime.qwen_docking=a.qwen_docking
 os.environ['SENSOR_SKILL_EXECUTION']='1'
 with runtime.activate(),episode_sensors(m,d) as sensors:
  from cross_episode_sim.navigation.shaped_docking import RobotGeometry
  from cross_episode_sim.control.observed_grasp_planner import observed_target_mask
  from cross_episode_sim.control.observed_payload import ObservedPayload
  geometry=RobotGeometry(m);x=np.load(pick/'input.npz');r=json.loads((pick/'physics/report.json').read_text())
  mask=observed_target_mask(x['points'],x['target'],r['world_stats']['support_z'])
  with h5py.File(pick/'physics/state_recording/states.h5') as f:
   i=next(i for i,phase in enumerate(f['phases']) if phase==b'close_gripper');grasp=f['states'][i]
  gd=mujoco.MjData(m);mujoco.mj_setState(m,gd,grasp,mujoco.mjtState.mjSTATE_INTEGRATION)
  tcp=geometry.tcp_pose(np.array([gd.joint(n).qpos[0] for n in geometry.joint_names]))
  tool=(x['points'][mask]-tcp[:3,3])@tcp[:3,:3]
  from cross_episode_sim.slam import create_mapper
  from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
  from cross_episode_sim.code_policy.recorded_context import restore_start_context
  mapper=create_mapper(fuse_dense=False);packet=ObservedPayload(geometry,tool).filter_packet(sensors.capture())
  tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
  grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
  grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
  anchor=restore_start_context(grid,rec,float(d.time));runtime.observed_navigation_map=grid
  source=a.output/'source';(source/'private_checkpoint').mkdir(parents=True)
  artifacts.save_array(source/'private_checkpoint/state.npy',state);artifacts.save_array(source/'observed_payload_tool.npy',tool)
  grid.save(source);mapper.save(source/'slam')
  artifacts.write_text(source/'report.json',json.dumps(dict(initialization_valid=True,transit_success=True)))
  memory=json.loads((a.episode/'exploration/report.json').read_text())['targets']
  tables=[]
  for item in memory:
   if item['label']!='dining table':continue
   item=dict(item);item['position']=(np.linalg.inv(anchor)@np.r_[item['position'],1.])[:3].tolist();item['surface_points']=[];tables.append(item)
  from cross_episode_sim.mapping.continuity import save_memory
  save_memory(source,dict(rooms=[],tables=tables))
  from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
  pipeline=SensorHouseholdPipeline.__new__(SensorHouseholdPipeline)
  pipeline.runtime=runtime;pipeline.output=a.output;pipeline.stages=[];pipeline.config=dict(task='set_table');pipeline.continue_on_failure=False
  perception=a.output/'perception';dock=a.output/'dock';placed=a.output/'place'
  try:
   def attempt(start,suffix):
    perception=a.output/('perception'+suffix);dock=a.output/('dock'+suffix);placed=a.output/('place'+suffix)
    pipeline.stage('observe_known_table'+suffix,[sys.executable,'scripts/test_m2t2_place.py','--dock',str(start),'--observe-only','--receptacle','dining table','--semantic-receptacle','dining table','--output',str(perception)],perception/'report.json','observation_success')
    evidence=json.loads(artifacts.read_text(perception/'report.json'));evidence['station']='dining table';artifacts.write_text(perception/'report.json',json.dumps(evidence))
    pipeline.stage('dock_known_table'+suffix,[sys.executable,'scripts/test_m2t2_place_redock_astar.py','--start',str(start),'--perception',str(perception),'--continuation-map',str(start),'--output',str(dock)],dock/'report.json','docking_success')
    pipeline.stage('place'+suffix,[sys.executable,'scripts/test_m2t2_place.py','--dock',str(dock),'--receptacle','dining table','--semantic-receptacle','dining table','--output',str(placed)]+(['--max-candidates',str(a.max_candidates)] if a.max_candidates else []),placed/'report.json','execution_success')
   try:attempt(source,'')
   except RuntimeError:
    # Same recovery as the production place handler: live-pose handoff, then
    # observe, dock and place again. Never restores the earlier state.
    if not a.retries:raise
    if getattr(runtime,'sensors',None) is None:runtime.sensors=sensors
    if getattr(runtime,'mapper',None) is None:runtime.mapper=mapper
    previous=source
    for retry in range(1,a.retries+1):
     start=pipeline.checkpoint(a.output/f'retry_{retry}_start',previous)
     artifacts.save_array(start/'observed_payload_tool.npy',tool)
     artifacts.write_text(start/'report.json',json.dumps(dict(initialization_valid=True,scope='placement retry handoff from the live pose',retry=retry)))
     print(f'PLACE_RETRY {retry} of {a.retries}',flush=True);previous=start
     try:attempt(start,f'_retry_{retry}');break
     except RuntimeError:
      if retry==a.retries:raise
   private=score.placement(d,'place');result=dict(private_placement=private,stages=pipeline.stages,scope='recorded failed carry, same production perception/dock/place; earlier observed landmark only')
   artifacts.write_text(a.output/'report.json',json.dumps(result,indent=2));print(json.dumps(result),flush=True)
   return 0 if private['success'] else 1
  finally:
   (a.output/'metrics').mkdir(exist_ok=True)
   artifacts.write_text(a.output/'metrics/report.json',json.dumps(runtime.metrics,indent=2))

if __name__=='__main__':raise SystemExit(main())
