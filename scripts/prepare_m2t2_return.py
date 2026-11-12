"""Physically return an empty arm to navigation posture using observed obstacles."""
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.recording.deferred import StateVideoRecorder
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from cross_episode_sim.runtime.episode import episode_sensors as OnboardSensorInterface,load_checkpoint
from cross_episode_sim.slam import create_mapper
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
from cross_episode_sim.navigation.map_continuation import restore_observed_map
from cross_episode_sim.navigation.local_scan import ScanConfig
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--place',type=Path,required=True)
    p.add_argument('--observed-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m,d=load_checkpoint(a.place)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    c=MolmoSpacesController(m,d);hold={k:c.robot_view.get_move_group(k).noop_ctrl.copy() for k in ('base','arm','gripper')}
    if np.max(hold['gripper'])>5:raise RuntimeError('Return requires released gripper command')
    grid=ObservedNavigationMap(MapConfig(resolution=.05,height_resolution=.025,pixel_stride=4,extent_m=12))
    restore_observed_map(grid,a.observed_map);slam=create_mapper()
    start=d.time;success=False;error=None
    with OnboardSensorInterface(m,d) as sensors,StateVideoRecorder(m,d,a.output/'state_recording') as recording:
        def observe():
            packet=sensors.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
            grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
            recording.record_map(grid,tracking,'return_posture')
            return packet,tracking
        try:
            packet,tracking=observe()
            if not tracking.accepted:raise RuntimeError('No accepted sensor pose for return')
            # Use the existing fixture stow backend: fresh local RGB-D at
            # manipulation resolution, calibrated robot geometry, checked
            # withdrawal waypoints and native cuRobo joint-goal planning.
            # A coarse navigation occupancy voxel is not an arm obstacle mesh.
            from types import SimpleNamespace
            from cross_episode_sim.control.fixture_posture import stow_after_fixture
            posture_result=stow_after_fixture(SimpleNamespace(model=m,data=d),sensors,
                recording,a.output,np.asarray(ScanConfig().stowed_arm),
                arm_boxes=True,withdrawal_directions=([0.,0.,1.],))
            success=True
        except Exception as exc:
            error=str(exc)
            print('RETURN_PLAN_FAILURE',error,flush=True)
    checkpoint=a.output/'private_checkpoint';checkpoint.mkdir();artifacts.save_model(m,str(checkpoint/'model.mjb'),None)
    state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);artifacts.save_array(checkpoint/'state.npy',state)
    grid.save(a.output);slam.save(a.output/'slam')
    from cross_episode_sim.mapping.continuity import load_memory,save_memory
    save_memory(a.output,load_memory(a.observed_map) if a.observed_map else dict(rooms=[],tables=[]))
    result=dict(success=success,error=error,simulated_seconds=float(d.time-start),reason='Checked return to currently required navigation posture; posture retention not yet supported by pickup navigation')
    result['planning_methods']=locals().get('posture_result',{}).get('planning_methods',[])
    result['obstacle_source']='fresh_onboard_rgbd_1cm'
    artifacts.write_text(a.output/'report.json',json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
