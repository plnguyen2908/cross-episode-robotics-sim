"""Whole-house exploration test. Room polygons are loaded ONLY after execution."""
import argparse,json,time,signal,inspect
from contextlib import nullcontext
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.house_exploration import HouseExplorer,HouseExplorationConfig
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.robot.observations import depth_rgb
from scripts.demo_local_scan import setup,ContactAudit,plot_grid


def interior_mask(vertices,points,margin=.25):
    from matplotlib.path import Path as PolygonPath
    vertices=np.asarray(vertices);points=np.asarray(points)
    inside=PolygonPath(vertices).contains_points(points)
    distance=np.full(len(points),np.inf)
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        edge=b-a
        if np.dot(edge,edge)==0:continue
        t=np.clip((points-a)@edge/np.dot(edge,edge),0,1)
        distance=np.minimum(distance,np.linalg.norm(points-a-t[:,None]*edge,axis=1))
    return inside&(distance>=margin)


def score(out,source,truth,initial,grid):
    from matplotlib.path import Path as PolygonPath
    layout=json.loads((source/'base_layout.json').read_text())
    y,x=np.mgrid[:grid.floor_views.shape[0],:grid.floor_views.shape[1]]
    local=grid.origin+(np.stack((x,y),-1)+.5)*grid.config.resolution
    world=local@initial[:2,:2].T+initial[:2,3]
    observed=grid.floor_views>=grid.config.minimum_floor_views
    scores=[]
    for room in layout['rooms']:
        poly=PolygonPath(np.asarray(room['floor_polygon']))
        cells=poly.contains_points(world.reshape(-1,2)).reshape(observed.shape)
        visits=interior_mask(room['floor_polygon'],truth[:,:2])
        vertices=np.asarray(room['floor_polygon'],dtype=float)
        area=float(abs(np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1)))*.5)
        # Unknown room area outside the allocated map must count as unexplored.
        coverage=float(min(1.,(observed&cells).sum()*grid.config.resolution**2/max(area,1e-12)))
        scores.append(dict(name=room['name'],interior_visit_seconds=float(visits.sum()*.1),floor_coverage=coverage,area_m2=area))
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=600.)
    p.add_argument('--planner',choices=['custom','tare'],default='custom')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    m,d,doors=setup(a.source);bridge=OnboardBaseActuator(m,d);audit=ContactAudit(m)
    slam=RGBDMapper();grid=ObservedNavigationMap(MapConfig(pixel_stride=6));policy=HouseExplorer(grid,HouseExplorationConfig(max_seconds=a.seconds))
    planner_context=nullcontext()
    if a.planner=='tare':
        from cross_episode_sim.navigation.tare import TarePlanner
        from cross_episode_sim.navigation.tare_exploration import TareExplorer
        planner_context=TarePlanner(a.output/'tare')
        policy=TareExplorer(grid,planner_context,HouseExplorationConfig(max_seconds=a.seconds))
    # This private transform is kept only by the diagnostic scorer.
    bid=m.body('robot_0/base').id;initial=np.eye(4);initial[:3,:3]=d.xmat[bid].reshape(3,3);initial[:3,3]=d.xpos[bid]
    start=float(d.time);wall=time.monotonic();rows=[];truth=[];poses=[];stable=0;last_status=None
    interrupted=SimpleNamespace(value=False)
    signal.signal(signal.SIGINT,lambda *_:setattr(interrupted,'value',True))
    (a.output/'policy_source.py').write_text(inspect.getsource(type(policy)))
    (a.output/'shared_follower_source.py').write_text(inspect.getsource(HouseExplorer))
    (a.output/'configuration.json').write_text(json.dumps(asdict(policy.config),indent=2))
    with planner_context,OnboardSensorInterface(m,d) as sensors,imageio.get_writer(str(a.output/'exploration.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,open(a.output/'progress.jsonl','w',buffering=1) as progress:
        for tick in tqdm(range(round((a.seconds+2)*50)),desc=f'Whole-house exploration ({a.planner})'):
            if tick%5==0:
                packet=sensors.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp);grid.update(packet.cameras['navigation'],tracking)
            if interrupted.value:policy.terminal='interrupted'
            try:
                command=policy.step(packet,tracking,float(d.time))
            except (RuntimeError, EOFError, ConnectionError, TimeoutError) as exc:
                if a.planner!='tare':raise
                policy.terminal='tare_process_failure'
                (a.output/'failure.json').write_text(json.dumps(dict(error=str(exc)),indent=2))
                command=policy._stop(policy.terminal)
            if tick%5==0:
                row=dict(time=float(d.time-start),**command,tracking=tracking.status,accepted=tracking.accepted,estimated_position=tracking.T_map_base[:2,3].tolist() if tracking.accepted else None,goal=policy.goal.tolist() if policy.goal is not None else None,observed_floor_m2=float((grid.floor_views>=2).sum()*.01))
                rows.append(row);progress.write(json.dumps(row)+'\n')
                poses.append(tracking.T_map_base if tracking.accepted else np.full((4,4),np.nan))
                truth.append(np.r_[d.xpos[bid,:2].copy(),float(d.time)])
                panel=Image.fromarray(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0));draw=ImageDraw.Draw(panel)
                draw.rectangle((0,0,1280,30),fill='white');draw.text((8,8),f'{d.time-start:.1f}s | {command["status"]} | scans {policy.scans} | floor {row["observed_floor_m2"]:.1f}m2',fill='black');video.append_data(np.asarray(panel))
                if tick%500==0:panel.save(a.output/'latest_frame.png')
                if (command['status']=='local_clearance_stop' or policy.terminal) and not (a.output/'first_stop_observation.npz').exists():
                    c=packet.cameras['navigation'];np.savez_compressed(a.output/'first_stop_observation.npz',rgb=c.rgb,depth=c.depth_m,mask=c.robot_mask,K=c.K,T_base_optical=c.T_base_optical,estimated_pose=tracking.T_map_base)
                    # Private diagnostic snapshot, never read by policy.
                    np.savez_compressed(a.output/'first_stop_physics.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),time=float(d.time))
                if tick%250==0 or command['status']!=last_status:
                    tqdm.write(json.dumps(row));last_status=command['status']
                if tick%1000==0:
                    grid.save(a.output/'latest_map');np.save(a.output/'estimated_poses.npy',poses)
            if policy.terminal and not (a.output/'terminal_state.npz').exists():
                spec=mujoco.mjtState.mjSTATE_INTEGRATION
                state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
                np.savez_compressed(a.output/'terminal_state.npz',state=state,time=float(d.time),state_spec=int(spec))
                c=packet.cameras['navigation']
                np.savez_compressed(a.output/'terminal_observation.npz',rgb=c.rgb,depth=c.depth_m,mask=c.robot_mask,K=c.K,T_base_optical=c.T_base_optical,estimated_pose=tracking.T_map_base if tracking.accepted else np.full((4,4),np.nan),timestamp=packet.timestamp)
            bridge.step(command['forward'],command['yaw'],diagnostic_observer=audit.observe)
            stable=stable+1 if policy.terminal and bridge.stationary() else 0
            if stable>=5:break
    truth=np.asarray(truth);np.save(a.output/'diagnostic_positions.npy',truth);np.save(a.output/'estimated_poses.npy',poses)
    report=grid.save(a.output);slam.save(a.output/'slam');plot_grid(a.output,title=f'Whole-house exploration ({a.planner})')
    room_scores=score(a.output,a.source,truth,initial,grid)
    report.update(status=policy.terminal,room_scores=room_scores,rooms_visited=sum(s['interior_visit_seconds']>=1. for s in room_scores),room_count=len(room_scores),
        coverage_success=all(s['interior_visit_seconds']>=1. and s['floor_coverage']>=.35 for s in room_scores),
        external_contact_events=list(audit.events.values()),physics_contact_samples=audit.samples,simulated_seconds=float(d.time-start),wall_seconds=time.monotonic()-wall,
        scans=policy.scans,plans=policy.plans,estimated_travel_m=policy.traveled,door_joints_initialized_open=doors,
        note='Room geometry used ONLY after policy termination to score coverage. Observed floor planning plus separate robot footprint support prior; live depth braking guard. No oracle route.')
    report['physical_success']=report['coverage_success'] and not audit.events
    report['planner']=a.planner
    report['last_guard_detail']=policy.stop_detail
    report['last_scan_nearest_point']=policy.scanner.nearest_point
    report['last_scan_nearest_distance']=policy.scanner.nearest
    if a.planner=='tare':
        report['tare_sensor_model_limitation']='Upstream uses omnidirectional LiDAR coverage heuristics. Input rays are actual RGB-D; full rotations provide angular observations, but this is not a calibrated RGB-D coverage model.'
    np.save(a.output/'ego_support_prior.npy',policy.support)
    (a.output/'events.json').write_text(json.dumps(policy.events,indent=2));(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
