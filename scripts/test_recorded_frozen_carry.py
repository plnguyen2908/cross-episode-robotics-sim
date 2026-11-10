"""Physical carry regression using saved observed 2D map and route graph."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import h5py,mujoco,numpy as np
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.fixed_map import freeze_map,map_digest,FixedRoomWaypointFollower
from cross_episode_sim.navigation.route_memory import RouteMemory
from cross_episode_sim.navigation.carry_departure import make_carry_departure
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.navigation.house_exploration import HouseExplorationConfig
from cross_episode_sim.navigation.transit_recovery import outside_transit_region,resume_transit
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.execution_monitor import PlannedMotionMonitor
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.slam.planar_motion import PlanarVisualMotion
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.evaluation.grasp_contact import contact_root_lookup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--transfer',default='transfer_02_bowl');p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--destination',default='dining table')
    p.add_argument('--overlap-departure',action='store_true',help='Isolated monotonic departure experiment; does not modify baseline')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record=a.run/a.transfer/'carry/state_recording'
    event=json.loads((record/'diagnostics.jsonl').read_text().splitlines()[0]);anchor=np.array(event['pose'])
    with np.load(record/event['map']) as x:
        grid=ObservedNavigationMap(MapConfig(resolution=float(x['resolution']),minimum_floor_views=int(x['minimum_floor_views'])))
        grid.origin=x['origin'].copy();grid.floor_views=x['floor'].copy();grid.evidence=x['occupied'][:,:,None].astype(np.int8)*2
    saved=json.loads((a.run/'success_graph.json').read_text());memory=RouteMemory()
    memory.poses=[np.array(v) for v in saved['poses']];memory.nodes=[v[:2,3] for v in memory.poses]
    memory.edges=saved['edges'];memory.stations={k:[{**r,'pose':np.array(r['pose'])} for r in rows] for k,rows in saved['stations'].items()}
    grid.inherited_support_poses=[np.r_[v[:2,3],np.arctan2(v[1,0],v[0,0])] for v in memory.poses]
    digest=freeze_map(grid)
    # Private physical reset. Policy inputs below are saved sensor estimates,
    # fresh onboard packets and robot-only calibration.
    model=mujoco.MjModel.from_binary_path(str(record/'model.mjb'));data=mujoco.MjData(model)
    with h5py.File(record/'states.h5') as f:mujoco.mj_setState(model,data,f['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    roots=[int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
    lookup=contact_root_lookup(model.body_parentid,roots)
    pads={model.body(n).id for n in ('robot_0/gripper/left_pad','robot_0/gripper/right_pad')}
    def privately_held():
        held=set()
        for c in data.contact:
            b1,b2=map(int,model.geom_bodyid[[c.geom1,c.geom2]])
            if b1 in pads and lookup[b2]>=0:held.add(int(lookup[b2]))
            if b2 in pads and lookup[b1]>=0:held.add(int(lookup[b1]))
        return held
    initial_held=privately_held();packet=None;tracking=None
    monitor=PlannedMotionMonitor()
    base=OnboardBaseActuator(model,data,motion_guard=lambda f,w,dt:monitor.check(packet,tracking,float(data.time),f,w))
    config=HouseExplorationConfig(max_seconds=a.seconds,arm_posture=tuple(base.controller.robot_view.get_move_group('arm').joint_pos))
    follower=FixedRoomWaypointFollower(grid,config)
    follower.set_destinations([r['pose'][:2,3] for r in memory.stations[a.destination]])
    follower.route_memory=memory;follower.route_destination=a.destination;follower.route_anchor=np.eye(4)
    calibration=SimpleNamespace(geometry=RobotGeometry(model))
    def make_departure():
        result=make_carry_departure(grid,calibration,known_map_only=True)
        if a.overlap_departure:
            from cross_episode_sim.navigation.overlap_departure import LocalOverlapDeparture
            result=LocalOverlapDeparture(result)
        return result
    departure=make_departure();departed=False
    mapper=create_mapper(fuse_dense=False);motion=PlanarVisualMotion()
    start=float(data.time);wall=time.monotonic();status='budget_exhausted';departures=[]
    with OnboardSensorInterface(model,data) as sensors,StateVideoRecorder(model,data,a.output/'state_recording') as recorder:
        for tick in range(round(a.seconds*50)):
            if tick%5==0:
                packet=sensors.capture(include_robot_mask=False);local=mapper.update(packet.cameras['navigation'],packet.timestamp)
                tracking=replace(local,T_map_base=motion.observe(anchor@local.T_map_base)) if local.accepted else local
                recorder.record_map(grid,tracking,'frozen_carry')
            if not tracking.accepted:status='tracking_lost';break
            if departed and tick%5==0 and outside_transit_region(follower,tracking):
                departure=make_departure();departed=False
            if not departed:
                region=follower.departure_region(packet,tracking)
                if departure.follower is None and not outside_transit_region(follower,tracking):departed=True
                else:
                    command=departure.step(packet,tracking,region,float(data.time))
                    if departure.error:status=departure.error;break
                    departed=command['status']=='undock_complete'
                    if departed:
                        departures.extend(departure.docking.events);resume_transit(follower,tracking,float(data.time))
            if departed:command=follower.step(packet,tracking,float(data.time))
            if tick%50==0:print(json.dumps(dict(time=float(data.time-start),pose=tracking.T_map_base[:2,3].tolist(),**command)),flush=True)
            if departed and command['status']=='arrived_frontier':status='known_destination_reached';break
            if follower.terminal:status=follower.terminal;break
            base.step(command['forward'],command['yaw']);motion.command(command['forward'],command['yaw'],.02)
        for _ in range(25):base.step(0.,0.);recorder.record('brake')
    retained=bool(initial_held & privately_held())
    report=dict(success=status=='known_destination_reached' and retained,status=status,private_hold_retained=retained,
        fixed_map_unchanged=map_digest(grid)==digest,wall_seconds=time.monotonic()-wall,
        simulated_seconds=float(data.time-start),departure_events=departures+departure.docking.events,route_events=follower.events,
        departure_error=departure.error,guard_detail=monitor.detail,
        scope='Saved physical carry start, frozen observed 2D map and same-episode routes; not full task')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
