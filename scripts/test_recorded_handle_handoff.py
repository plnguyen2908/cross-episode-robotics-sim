"""Development test: restore a failed pregrasp view, then use live sensor skills."""
import argparse,json,time
from pathlib import Path
import h5py,mujoco,numpy as np
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.control.live_door_open import LiveDoorOpener
from cross_episode_sim.navigation.portal_approach import PortalApproach
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.perception.molmo_pointing import MolmoPointer
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.slam import create_mapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--portal',default='portal_001');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record=a.source/'state_recording'
    events=[json.loads(x) for x in (record/'diagnostics.jsonl').read_text().splitlines()]
    event=next(e for e in reversed(events) if e.get('accepted') and e.get('pose') is not None)
    # Private reset only. All execution consumes sensor estimates below.
    m=mujoco.MjModel.from_binary_path(str(record/'model.mjb'));d=mujoco.MjData(m)
    with h5py.File(record/'states.h5') as f:
        index=int(np.argmin(abs(f['times'][:]-event['timestamp'])))
        if abs(float(f['times'][index])-event['timestamp'])>.051:raise ValueError('No synchronized saved estimate')
        mujoco.mj_setState(m,d,f['states'][index],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m,d)
    report=json.loads((a.source/'report.json').read_text())
    target=np.asarray(next(t for t in report['portals'] if t['portal']==a.portal)['position'])
    with np.load(a.source/'wall_evidence.npz') as f:
        resolution=float(f['resolution']);grid=ObservedNavigationMap(MapConfig(resolution=resolution,height_resolution=resolution))
        grid.evidence=f['evidence'].copy();grid.floor_views=f['floor'].copy();grid.origin=f['origin'].copy()
    runtime=EpisodeRuntime(a.output);runtime.bind(m,d);runtime.observed_navigation_map=grid
    wall=time.monotonic();start=float(d.time);result=dict(success=False,reason='view_recovery_budget')
    with runtime.activate(),episode_sensors(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recorder:
        mapper=create_mapper(fuse_dense=False);runtime.mapper.map_alignment=np.asarray(event['pose'])
        approach=PortalApproach(grid,target,geometry=RobotGeometry(m));approach.align=True
        opener=LiveDoorOpener(runtime,MolmoPointer(),a.output/'doors')
        base=OnboardBaseActuator(m,d);packet=None
        for tick in range(50*30):
            if packet is None or d.time-packet.timestamp>=.1-1e-8:
                packet=sensors.capture(include_robot_mask=False)
                tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
            command=approach.step(packet,tracking,float(d.time))
            if tick%50==0:print(json.dumps(dict(stage='view_alignment',simulated_seconds=float(d.time-start),**command)),flush=True)
            if command['status']=='handle_view_ready':
                base.step(0.,0.);result=opener.open(sensors,tracking,target)
                if result.get('recoverable') and approach.retry_view(tracking):continue
                break
            base.step(command['forward'],command['yaw']);recorder.record(command['status'])
    result.update(wall_seconds=time.monotonic()-wall,source=str(a.source),scope='saved failed-view recovery; not full pipeline')
    (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
