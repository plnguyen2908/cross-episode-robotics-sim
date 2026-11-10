"""Physical regression of a frozen-map corner, using recorded sensor evidence."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import h5py,mujoco,numpy as np
from cross_episode_sim.navigation.fixed_map import FixedRoomWaypointFollower,freeze_map
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.route_memory import RouteMemory
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.recording.deferred import StateVideoRecorder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recover-region',action='store_true',help='Include the pipeline observed-shape departure when tracking leaves the circular transit region')
    p.add_argument('--at-seconds',type=float,default=None,help='Reset at this offset in the recorded carry, before the failed corner')
    p.add_argument('--min-progress',type=float,default=.5)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record=a.run/'transfer_02_bowl/carry/state_recording'
    events=[json.loads(l) for l in (record/'diagnostics.jsonl').read_text().splitlines()]
    event=next(e for e in reversed(events) if e['accepted'] and e['pose'] is not None)
    if a.at_seconds is not None:
        stamp=events[0]['timestamp']+a.at_seconds
        event=min((e for e in events if e['accepted'] and e['pose'] is not None),key=lambda e:abs(e['timestamp']-stamp))
    anchor=np.asarray(event['pose']);saved=np.load(record/event['map'])
    grid=ObservedNavigationMap(MapConfig(resolution=float(saved['resolution']),minimum_floor_views=int(saved['minimum_floor_views'])))
    grid.origin=saved['origin'];grid.floor_views=saved['floor'];grid.evidence=saved['occupied'][:,:,None].astype(np.int8)*2
    graph=json.loads((a.run/'success_graph.json').read_text());memory=RouteMemory()
    memory.poses=[np.array(p) for p in graph['poses']];memory.nodes=[p[:2,3] for p in memory.poses];memory.edges=graph['edges']
    memory.stations={k:[{**s,'pose':np.array(s['pose'])} for s in rows] for k,rows in graph['stations'].items()}
    grid.inherited_support_poses=[np.r_[p[:2,3],np.arctan2(p[1,0],p[0,0])] for p in memory.poses];freeze_map(grid)
    # Private physical reset only. No true scene pose is supplied to following.
    model=mujoco.MjModel.from_binary_path(str(record/'model.mjb'));data=mujoco.MjData(model)
    with h5py.File(record/'states.h5') as f:
        index=np.argmin(abs(f['times'][:]-event['timestamp']))
        if abs(float(f['times'][index])-event['timestamp'])>.051:raise ValueError('Unsynchronized recording')
        mujoco.mj_setState(model,data,f['states'][index],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    monitor=PlannedMotionMonitor();packet=None;tracking=None
    base=OnboardBaseActuator(model,data,motion_guard=lambda f,w,dt:monitor.check(packet,tracking,float(data.time),f,w))
    config=HouseExplorationConfig(max_seconds=40,arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos))
    follower=FixedRoomWaypointFollower(grid,config);follower.set_destinations([])
    follower.route_memory=memory;follower.route_destination='dining table';follower.route_anchor=np.eye(4)
    departure=None
    if a.recover_region:
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry,ShapedDocking
        from cross_episode_sim.navigation.observed_undock import ObservedUndock
        from cross_episode_sim.navigation.transit_recovery import outside_transit_region,resume_transit
        # Only projected occupancy was recorded. Extrude it conservatively for
        # this diagnostic; never infer clear height bands from missing data.
        shape_grid=ObservedNavigationMap(grid.config)
        shape_grid.origin=grid.origin.copy();shape_grid.floor_views=grid.floor_views.copy()
        layers=int(np.ceil((grid.config.robot_height-grid.config.min_obstacle_height)/grid.config.height_resolution))
        shape_grid.evidence=np.repeat(grid.evidence,layers,axis=2)
        shape_grid.inherited_support_poses=grid.inherited_support_poses
        freeze_map(shape_grid)
        geometry=RobotGeometry(model)
        departure_dock=ShapedDocking(shape_grid,geometry)
    mapper=create_mapper(fuse_dense=False);start=float(data.time);wall=time.monotonic();distance=0.;status='budget_exhausted'
    with OnboardSensorInterface(model,data) as sensors,StateVideoRecorder(model,data,a.output/'state_recording') as recorder:
        for tick in range(40*50):
            if tick%5==0:
                packet=sensors.capture(include_robot_mask=False);local=mapper.update(packet.cameras['navigation'],packet.timestamp)
                tracking=replace(local,T_map_base=anchor@local.T_map_base) if local.accepted else local
                if tracking.accepted:distance=float(np.linalg.norm(tracking.T_map_base[:2,3]-anchor[:2,3]))
                recorder.record_map(grid,tracking,'frozen_corner')
            if a.recover_region and departure is None and outside_transit_region(follower,tracking):
                departure=ObservedUndock(departure_dock)
            if departure is not None:
                command=departure.step(packet,tracking,follower.departure_region(packet,tracking),float(data.time))
                if departure.error:status=departure.error;break
                if command['status']=='undock_complete':
                    departure=None;resume_transit(follower,tracking,float(data.time))
            else:command=follower.step(packet,tracking,float(data.time))
            base.step(command['forward'],command['yaw'])
            if tick%50==0:print(json.dumps(dict(simulated_seconds=float(data.time-start),distance_m=distance,**command)),flush=True)
            if distance>a.min_progress:status='corner_passed';break
            if follower.terminal:status=follower.terminal;break
        # Brake the commanded base rather than ending with a moving controller.
        for _ in range(30):base.step(0.,0.);recorder.record('brake')
    report=dict(success=status=='corner_passed',status=status,estimated_displacement_m=distance,plans=follower.plans,
        simulated_seconds=float(data.time-start),wall_seconds=time.monotonic()-wall,events=follower.events,
        scope='Physical corner progress using frozen observed occupancy and SLAM; not a completed carry or task')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
