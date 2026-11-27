"""One simulator and renderer per episode; checkpoints are for restart only."""
from contextlib import contextmanager,nullcontext,redirect_stdout,redirect_stderr
from contextvars import ContextVar
from pathlib import Path
import json,runpy,sys,time,subprocess
import numpy as np
import mujoco

_current=ContextVar('sensor_episode_runtime',default=None)


class EpisodeRuntime:
    def __init__(self,artifact_root=None):
        self.artifact_root=Path(artifact_root).resolve() if artifact_root is not None else None
        self.artifacts={};self.mapper=None
        self.model=None;self.data=None;self.sensors=None;self.sensor_config=None
        self.metrics={'checkpoint_restores':0,'renderer_initializations':0,'stages':[]}
    def bind(self,model,data):
        if self.model is not None and (model is not self.model or data is not self.data):
            raise RuntimeError('Stage tried to replace the live episode')
        self.model,self.data=model,data
    def close(self):
        from cross_episode_sim.slam.gpu_sift import sift_cache_info
        self.metrics['sift']=sift_cache_info()
        if self.sensors is not None:self.sensors.close();self.sensors=None
        if self.artifact_root is not None and self.mapper is not None and getattr(self.mapper,'map_localization_events',None):
            (self.artifact_root/'map_localization.json').write_text(json.dumps(self.mapper.map_localization_events))
        from .artifacts import export_final_diagnostics
        export_final_diagnostics(self)
        if self.artifact_root is not None and hasattr(self,'route_memory'):
            memory=self.route_memory
            # One final diagnostic export, never a checkpoint/reload per skill.
            (self.artifact_root/'success_graph.json').write_text(json.dumps(dict(
                provenance='accepted onboard SLAM poses',poses=[p.tolist() for p in memory.poses],
                edges=memory.edges,stations={k:[{**v,'pose':v['pose'].tolist()} for v in values] for k,values in memory.stations.items()})))
        if hasattr(self,'arm_planners'):self.arm_planners.clear()
    @contextmanager
    def activate(self):
        token=_current.set(self)
        try:yield self
        finally:_current.reset(token);self.close()
    def run(self,command,log):
        """Run migrated stage code in-process, preserving live physics/resources."""
        if command[:2]==['bash','scripts/test_m2t2_pick.sh']:
            source,output=command[2:4]
            stages=[
                [sys.executable,'scripts/capture_m2t2_input.py','--source',source,'--output',output],
                [sys.executable,'scripts/m2t2_worker.py','--input',str(Path(output)/'input.npz'),'--output',str(Path(output)/'predictions.json'),'--checkpoint',__import__('cross_episode_sim.control.m2t2_policy',fromlist=['default_checkpoint']).default_checkpoint()],
                [sys.executable,'scripts/test_m2t2_pick.py','--source',source,'--input',str(Path(output)/'input.npz'),'--predictions',str(Path(output)/'predictions.json'),'--output',str(Path(output)/'physics')]]
            for cmd in stages:
                # M2T2's dependency environment has no simulator; model weights
                # reside in its configured persistent gRPC worker.
                result=self.run(cmd,log)
                if result.returncode:return result
            return subprocess.CompletedProcess(command,0)
        if len(command)<2 or command[0]!=sys.executable:
            raise ValueError(f'Unmigrated persistent stage: {command[:2]}')
        from cross_episode_sim.perception.rpc import call_metrics
        rpc_before=call_metrics()
        previous=sys.argv;start=time.monotonic();code=0
        try:
            sys.argv=command[1:]
            with redirect_stdout(log),redirect_stderr(log):
                try:runpy.run_path(command[1],run_name='__main__')
                except SystemExit as exc:
                    if exc.code not in (None,0):code=exc.code if isinstance(exc.code,int) else 1
                except Exception:
                    import traceback
                    traceback.print_exc(file=log);code=1
        finally:
            sys.argv=previous
            rpc_after=call_metrics()
            rpc_delta={key:{field:value-rpc_before.get(key,{}).get(field,0) for field,value in stats.items()} for key,stats in rpc_after.items()}
            # runpy module globals can form cycles through local functions.
            # Release stage-local planners; live simulator/cameras stay owned here.
            import gc
            cleanup_start=time.monotonic();gc.collect()
            self.metrics['stages'].append({'script':command[1],'wall_seconds':time.monotonic()-start,'cleanup_seconds':time.monotonic()-cleanup_start,'model_rpc':rpc_delta})
        return subprocess.CompletedProcess(command,code)


def load_checkpoint(directory):
    directory=Path(directory);runtime=_current.get()
    from . import artifacts
    state=artifacts.load_array(directory/'private_checkpoint/state.npy')
    if runtime is not None and runtime.model is not None:
        # Never rewind an active episode: a stage always continues from the live
        # state. A handoff saved before later motion (a memory attempt that fell
        # back, a back-off, an arm settle) is only behind it; the stage uses its
        # files (observed target, map, payload) with the live robot, as a live
        # run does. It used to raise 'Checkpoint is not the current live state'
        # (same-layout batch 20261126_152342: 24 pick/observe_receptacle stages).
        if len(state)!=mujoco.mj_stateSize(runtime.model,mujoco.mjtState.mjSTATE_INTEGRATION):
            raise RuntimeError(f'Checkpoint is from a different model: {directory}')
        if (abs(float(state[0])-runtime.data.time)>1e-7
                or not np.allclose(state[1:1+runtime.model.nq],runtime.data.qpos,rtol=0,atol=1e-9)):
            print('CHECKPOINT_BEHIND_LIVE_STATE',json.dumps(dict(checkpoint=str(directory)[-90:],saved_time=round(float(state[0]),3),
                  live_time=round(float(runtime.data.time),3),max_qpos_diff=round(float(np.abs(state[1:1+runtime.model.nq]-runtime.data.qpos).max()),4))),flush=True)
        return runtime.model,runtime.data
    model=mujoco.MjModel.from_binary_path(str(directory/'private_checkpoint/model.mjb'))
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(model,data)
    if runtime is not None:
        runtime.bind(model,data);runtime.metrics['checkpoint_restores']+=1
    return model,data


def episode_sensors(model,data,**kwargs):
    from cross_episode_sim.sensors import OnboardSensorInterface
    runtime=_current.get()
    if runtime is None:return OnboardSensorInterface(model,data,**kwargs)
    runtime.bind(model,data)
    if runtime.sensors is None:
        runtime.sensors=OnboardSensorInterface(model,data,**kwargs)
        runtime.sensor_config=kwargs;runtime.metrics['renderer_initializations']+=1
    elif runtime.sensor_config!=kwargs:raise ValueError('Camera configuration changed inside episode')
    return nullcontext(runtime.sensors)


def current_runtime():return _current.get()
