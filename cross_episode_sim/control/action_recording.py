"""Exact initial state and executed commands for post-generation action replay."""
from bisect import bisect_left
import json
import mujoco
import numpy as np


class ActionRecording:
    def __init__(self, output):
        self.output=output/'action_recording'
        self.times=[];self.commands=[];self.positions=[];self.forces=[];self.stages=[]
        self.labels=[];self.initialized=False;self.boundaries=[]

    def sample(self,c):
        if not self.initialized:
            self.output.mkdir(parents=True,exist_ok=True)
            mujoco.mj_saveModel(c.model,str(self.output/'model.mjb'),None)
            kind=mujoco.mjtState.mjSTATE_INTEGRATION
            state=np.empty(mujoco.mj_stateSize(c.model,kind))
            mujoco.mj_getState(c.model,c.data,state,kind)
            np.savez_compressed(self.output/'initial_state.npz',state=state,state_type=int(kind))
            self.initialized=True
        t=float(c.data.time)
        # Discard commands from speculative branches restored by a checkpoint.
        if self.times and t<=self.times[-1]+1e-10:
            index=bisect_left(self.times,t-1e-10)
            for values in (self.times,self.commands,self.positions,self.forces,self.stages):del values[index:]
        label=getattr(c,'stage','initialization')
        if label not in self.labels:self.labels.append(label)
        self.times.append(t);self.commands.append(c.data.ctrl.copy())
        self.positions.append(c.data.qpos.copy());self.forces.append(c.model.actuator_forcerange.copy())
        self.stages.append(self.labels.index(label))

    def boundary(self,c,event):
        self.boundaries.append(dict(skill=event['skill'],arguments=event['arguments'],
            success=event['success'],start_time=event['start_time'],end_time=event['end_time'],
            qpos=c.data.qpos.tolist()))

    def save(self,c):
        if not self.initialized:return
        np.savez_compressed(self.output/'actions.npz',time=self.times,ctrl=self.commands,
                            qpos=self.positions,forcerange=self.forces,stage=self.stages)
        (self.output/'metadata.json').write_text(json.dumps(dict(stages=self.labels,
            calls=self.boundaries,source_success=c.report.get('success',False),
            namespace=c.profile.namespace,gripper_actuator=c.profile.gripper_actuator,
            sample_semantics='commands immediately before each physics step; no replanning during replay'),indent=2))
