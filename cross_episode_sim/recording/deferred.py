"""Record actual simulation states without rendering; replay states, not actions.

This harness-only data is never an observation supplied to a policy.
"""
from pathlib import Path
import json
import time
import numpy as np
import mujoco


class StateVideoRecorder:
    def __init__(self, model, data, output, fps=10):
        import h5py
        self.model,self.data=model,data
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.fps=fps;self.last=None;self.count=0;self.wall_seconds=0.
        self.map_stamp=None;self.map_path=None;self.map_frames=None
        self.diagnostics=(self.output/'diagnostics.jsonl').open('w',buffering=1)
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.buffer=np.empty(mujoco.mj_stateSize(model,self.spec))
        from cross_episode_sim.runtime.episode import current_runtime
        runtime=current_runtime()
        if runtime is not None and runtime.artifact_root is not None:
            # One compiled-model recording per episode, not another 500 MB
            # checkpoint at every skill boundary. Stage videos share its inode.
            shared=runtime.artifact_root/'recording_model.mjb'
            if not shared.exists():mujoco.mj_saveModel(model,str(shared),None)
            import os
            os.link(shared,self.output/'model.mjb')
        else:mujoco.mj_saveModel(model,str(self.output/'model.mjb'),None)
        self.file=h5py.File(self.output/'states.h5','w')
        n=len(self.buffer)
        self.states=self.file.create_dataset('states',(0,n),maxshape=(None,n),chunks=(64,n),dtype='f8')
        self.times=self.file.create_dataset('times',(0,),maxshape=(None,),chunks=(64,),dtype='f8')
        self.phases=self.file.create_dataset('phases',(0,),maxshape=(None,),chunks=(64,),dtype=h5py.string_dtype())

    def record(self, phase='', force=False):
        stamp=float(self.data.time)
        if self.last is not None and (stamp<=self.last or (not force and stamp-self.last<1/self.fps-1e-8)):
            return
        start=time.monotonic()
        mujoco.mj_getState(self.model,self.data,self.buffer,self.spec)
        i=self.count
        self.states.resize(i+1,axis=0);self.times.resize(i+1,axis=0);self.phases.resize(i+1,axis=0)
        self.states[i]=self.buffer;self.times[i]=stamp;self.phases[i]=phase
        self.last=stamp;self.count+=1
        self.wall_seconds+=time.monotonic()-start

    def record_map(self, grid, tracking, phase='', selected_tcp=None):
        """Small diagnostic snapshots; no render or inference on this path."""
        stamp=float(self.data.time)
        self.record(phase)
        if self.map_path is None or (grid.frames!=self.map_frames and stamp-self.map_stamp>=1.):
            self.map_path=f'map_{self.count:06d}.npz'
            np.savez(self.output/self.map_path,origin=grid.origin,floor=grid.floor_views,
                     occupied=(grid.evidence>=2).any(axis=2),resolution=grid.config.resolution,
                     minimum_floor_views=grid.config.minimum_floor_views)
            self.map_stamp=stamp;self.map_frames=grid.frames
        self.diagnostics.write(json.dumps(dict(timestamp=stamp,map=self.map_path,
            accepted=bool(tracking.accepted),status=str(tracking.status),
            pose=None if tracking.T_map_base is None else tracking.T_map_base.tolist(),
            phase=phase,tcp=None if selected_tcp is None else np.asarray(selected_tcp).tolist()))+'\n')

    def close(self):
        self.file.flush();self.file.close();self.diagnostics.close()
        (self.output/'metadata.json').write_text(json.dumps(dict(fps=self.fps,frames=self.count,state_spec=int(self.spec),recording_wall_seconds=self.wall_seconds,rendered=False),indent=2))

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def render_recording(source, output, width=640, height=480, diagnostic_output=None):
    """Render isolated recorded states. Never step physics or rerun control."""
    import h5py
    import imageio.v2 as imageio
    from cross_episode_sim.sensors import OnboardSensorInterface
    from cross_episode_sim.robot.observations import depth_rgb
    from contextlib import nullcontext
    from types import SimpleNamespace
    from cross_episode_sim.navigation.pipeline_video import PipelineVideoPanel
    from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig
    source=Path(source);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'metadata.json').read_text());start=time.monotonic()
    model=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));data=mujoco.MjData(model)
    events_path=source/'diagnostics.jsonl'
    events=[json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    index=0;map_file=None;grid=None;panel=PipelineVideoPanel()
    diagnostic=(imageio.get_writer(str(diagnostic_output),fps=meta['fps'],codec='libx264',pixelformat='yuv420p') if diagnostic_output and events else nullcontext(None))
    with h5py.File(source/'states.h5','r') as recorded, OnboardSensorInterface(model,data,width=width,height=height) as sensors, imageio.get_writer(str(output),fps=meta['fps'],codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer,diagnostic as diagnostic_writer:
        for state in recorded['states']:
            mujoco.mj_setState(model,data,state,meta['state_spec']);mujoco.mj_forward(model,data)
            packet=sensors.capture(include_robot_mask=False)
            writer.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()]))
            if diagnostic_writer is not None:
                while index+1<len(events) and events[index+1]['timestamp']<=data.time:index+=1
                event=events[index]
                if event['map']!=map_file:
                    saved=np.load(source/event['map']);map_file=event['map']
                    grid=ObservedNavigationMap(MapConfig(resolution=float(saved['resolution']),minimum_floor_views=int(saved['minimum_floor_views'])))
                    grid.origin=saved['origin'];grid.floor_views=saved['floor'];grid.evidence=saved['occupied'][:,:,None].astype(np.int8)*2
                tracking=SimpleNamespace(accepted=event['accepted'],status=event['status'],T_map_base=None if event['pose'] is None else np.array(event['pose']))
                diagnostic_writer.append_data(panel.render(grid,tracking,packet,event['phase'],event['tcp']))
    result=dict(video=str(output),frames=meta['frames'],wall_seconds=time.monotonic()-start,source='saved physical states; no action replay')
    (source/'render_report.json').write_text(json.dumps(result,indent=2))
    return result


def render_episode(directory):
    """Render completed or failed stage recordings after episode execution ends."""
    directory=Path(directory);results=[];start=time.monotonic()
    for metadata in sorted(directory.rglob('state_recording/metadata.json')):
        source=metadata.parent
        try:
            info=json.loads(metadata.read_text())
            if not info['frames']:
                results.append(dict(recording=str(source),status='empty'));continue
            result=render_recording(source,source.parent/'rollout.mp4',
                                    diagnostic_output=source.parent/'diagnostic.mp4')
            # Restore the point-cloud visualization too, but only after motion.
            # This uses saved observed M2T2 input, never scene meshes.
            physics=source.parent
            if (physics/'chosen.json').exists() and (physics/'diagnostic.mp4').exists():
                import subprocess,sys
                kind='pick' if physics.name=='physics' else 'place'
                input_dir=physics.parent if kind=='pick' else physics
                if (input_dir/'input.npz').exists():
                    script=Path(__file__).resolve().parents[2]/'scripts/render_m2t2_geometry.py'
                    with (physics/'offline_geometry.log').open('w') as log:
                        proc=subprocess.run([sys.executable,str(script),'--'+kind,str(input_dir),'--video',str(physics/'diagnostic.mp4')],stdout=log,stderr=subprocess.STDOUT)
                    result['geometry_render_exit_code']=proc.returncode
            results.append(dict(recording=str(source),status='rendered',**result))
        except Exception as exc:
            # Rendering failure must not overwrite the physical task score.
            results.append(dict(recording=str(source),status='failed',error=str(exc)))
        print(json.dumps(results[-1]),flush=True)
    report=dict(wall_seconds=time.monotonic()-start,recordings=results)
    (directory/'offline_render_report.json').write_text(json.dumps(report,indent=2))
    return report
