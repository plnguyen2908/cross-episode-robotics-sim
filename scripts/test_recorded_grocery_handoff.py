"""Replay a grocery handoff using its observed target, never scene object poses.

The recorded simulator state is only the private test reset. Rebuild current
RGB-D geometry and reanchor the saved target with the recorded SLAM pose.
"""
import argparse,json,sys
from pathlib import Path
import h5py,mujoco,numpy as np
from PIL import Image
from cross_episode_sim.runtime.episode import EpisodeRuntime,episode_sensors
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.code_policy.recorded_context import restore_start_context
from cross_episode_sim.code_policy.grocery_redocking import GroceryRedocking


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--navigation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--observe-only',action='store_true')
    p.add_argument('--target-observation',type=Path,help='Reprocessed sensor-only navigation observation in the original estimated map frame')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    record=a.navigation/'state_recording'
    model=mujoco.MjModel.from_binary_path(str(record/'model.mjb'));data=mujoco.MjData(model)
    with h5py.File(record/'states.h5') as f:state=f['states'][-1]
    mujoco.mj_setState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(model,data)
    selected=json.loads((a.navigation/'report.json').read_text())['selected_object']
    if a.target_observation:selected=json.loads(a.target_observation.read_text())
    runtime=EpisodeRuntime(a.output);runtime.bind(model,data)
    with runtime.activate(),episode_sensors(model,data) as sensors:
        mapper=create_mapper(fuse_dense=False)
        grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4))
        packet=sensors.capture();tracking=mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
        anchor=restore_start_context(grid,record,float(data.time))
        point=np.linalg.inv(anchor)@np.r_[selected['position'],1.]
        runtime.observed_navigation_map=grid;runtime.qwen_docking=True
        dock=GroceryRedocking(runtime,sensors,mapper)
        dock.target_label=selected['label'];dock.target_point=point[:3];dock.preserve_target=True
        print('OBSERVED_TARGET_BASE',point.tolist(),flush=True)
        for key,camera in packet.cameras.items():Image.fromarray(camera.rgb).save(a.output/(key+'.png'))
        from cross_episode_sim.control.sensor_appliance_backend import camera_points
        np.savez(a.output/'observed_geometry.npz',points=np.concatenate([camera_points(c)[0] for c in packet.cameras.values()]),target=point[:3])
        packet,tracking,point,camera=dock.observe(selected['label'])
        (a.output/'confirmation.json').write_text(json.dumps(dict(target=point.tolist(),camera=camera,front=dock.observed_support_front),indent=2))
        if a.observe_only:return 0
        result=dock.move(selected['label'],'pick',.55,a.output/'dock')
        print('REDOCK_RESULT',json.dumps(result),flush=True)
        if not result['success']:return 1
        packet,tracking,point,camera=dock.observe(selected['label'])
        source=a.output/'source';source.mkdir();(source/'private_checkpoint').mkdir()
        state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        artifacts.save_array(source/'private_checkpoint/state.npy',state)
        grid.save(source);mapper.save(source/'slam')
        artifacts.write_text(source/'report.json',json.dumps(dict(initialization_valid=True,selected_object=dict(label=selected['label'],position=point.tolist()))))
        result=runtime.run([sys.executable,'scripts/execute_household_pick.py','--source',str(source),'--output',str(a.output/'pick')],sys.stdout)
        return result.returncode


if __name__=='__main__':raise SystemExit(main())
