"""Sensor-only M2T2 proposal and execution tools for a pre-docked trial."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from cross_episode_sim.runtime import artifacts


class PickSession:
    def __init__(self,runtime,sensors,output,target,tool_budget=12):
        self.runtime,self.sensors,self.output=runtime,sensors,Path(output)
        self.target=target;self.calls=[];self.tool_budget=tool_budget;self.finished=False
        self.proposals=None;self.private_reports=[];self.source=None

    def log(self,name,args,result,started):
        event=dict(tool=name,arguments=args,result=result,wall_seconds=time.monotonic()-started)
        self.calls.append(event)
        (self.output/'tool_calls.json').write_text(json.dumps(self.calls,indent=2))
        print('POLICY_TOOL',name,'completed',round(event['wall_seconds'],2),'seconds',flush=True)
        return result

    def budget(self):
        if len(self.calls)>=self.tool_budget:raise RuntimeError('Tool-call budget exhausted')

    def state(self):
        """Return {joints: seven measured angles, tcp_base: 4x4 calibrated TCP matrix}. Position is tcp_base[i][3], i=0,1,2."""
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry
        self.budget();start=time.monotonic();packet=self.sensors.capture(include_robot_mask=False)
        geometry=RobotGeometry(self.runtime.model);q=packet.arm.position
        result=dict(joints=q.tolist(),tcp_base=geometry.tcp_pose(q).tolist())
        return self.log('get_robot_state',{},result,start)

    def images(self):
        packet=self.sensors.capture(include_robot_mask=False)
        return [Image.fromarray(packet.cameras[k].rgb) for k in ('navigation','wrist')]

    def snapshot(self,folder):
        import mujoco
        folder.mkdir(parents=True,exist_ok=True)
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;m,d=self.runtime.model,self.runtime.data
        state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
        artifacts.save_model(m,str(folder/'private_checkpoint/model.mjb'),None)
        artifacts.save_array(folder/'private_checkpoint/state.npy',state)
        self.runtime.observed_navigation_map.save(folder)
        self.mapper.save(folder/'slam')

    def propose(self,target):
        """Observe the named visible bowl and return M2T2 grasp candidates (IDs/scores/TCPs). No motion."""
        from cross_episode_sim.perception.molmo_pointing import MolmoPointer
        from cross_episode_sim.control.sensor_appliance_backend import pixel_point,camera_points
        self.budget();start=time.monotonic()
        if target!=self.target:raise ValueError('This component trial supports only the prompted target category')
        folder=self.output/f'proposal_{len(self.calls):03d}'
        packet=self.sensors.capture();camera=packet.cameras['wrist']
        points=MolmoPointer().point(Image.fromarray(camera.rgb),target)['points']
        if not points:return self.log('propose_grasps',dict(target=target),dict(error='Target not visible',candidates=[]),start)
        _,_,valid=camera_points(camera);target_base=pixel_point(camera,valid,points[0])
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not tracking.accepted:raise RuntimeError('Current SLAM localization unavailable')
        target_map=tracking.T_map_base[:3,:3]@target_base+tracking.T_map_base[:3,3]
        self.snapshot(folder)
        artifacts.write_text(folder/'report.json',json.dumps(dict(selected_object=dict(label=target,position=target_map.tolist()))))
        with (folder/'perception.log').open('w',buffering=1) as log:
            for command in ([sys.executable,'scripts/capture_m2t2_input.py','--source',str(folder),'--output',str(folder)],
                            [sys.executable,'scripts/m2t2_worker.py','--input',str(folder/'input.npz'),'--output',str(folder/'predictions.json'),'--checkpoint','remote']):
                result=self.runtime.run(command,log)
                if result.returncode:raise RuntimeError('Perception backend failed; no action executed')
        predictions=json.loads(artifacts.read_text(folder/'predictions.json'))
        self.proposals=predictions;self.source=folder
        rows=[dict(id=i,score=round(row.get('score',row.get('confidence')),3),tcp_position_base=np.asarray(row['tcp'])[:3,3].round(3).tolist(),approach_axis_base=np.asarray(row['tcp'])[:3,2].round(3).tolist(),contact_base=np.asarray(row['contact']).round(3).tolist()) for i,row in enumerate(predictions['candidates'][:8])]
        return self.log('propose_grasps',dict(target=target),dict(candidates=rows),start)

    def execute(self,candidate_ids,approach_offset=.06):
        """Execute a COMPLETE pickup: approach, close fingers, lift off counter, and hold. No separate lift command is needed. Backend ranks selected M2T2 variants and uses cuRobo. Offset is meters, 0.02–0.15. Returns execution_success, observed_gripper_not_empty, phase (verify_hold after lift), simulated_seconds; not a task score. Consumes the current proposals. Inspect fresh images afterwards before deciding whether to finish or request new proposals."""
        self.budget();start=time.monotonic()
        if self.proposals is None:raise ValueError('Call propose_grasps first')
        if not .02<=approach_offset<=.15:raise ValueError('approach_offset must be 0.02 to 0.15 meters')
        if not isinstance(candidate_ids,list) or not 1<=len(candidate_ids)<=24:raise ValueError('Supply 1–24 candidate IDs')
        if any(type(i) is not int or i<0 or i>=min(24,len(self.proposals['candidates'])) for i in candidate_ids):raise ValueError('Unknown candidate ID')
        folder=self.output/f'execute_{len(self.calls):03d}';folder.mkdir()
        prediction=folder/'selected.json'
        artifacts.write_text(prediction,json.dumps({**self.proposals,'candidates':[self.proposals['candidates'][i] for i in candidate_ids]}))
        # Current live state, never a rollback between attempts.
        self.snapshot(self.source)
        with (folder/'execution.log').open('w',buffering=1) as log:
            result=self.runtime.run([sys.executable,'scripts/test_m2t2_pick.py',
                '--source',str(self.source),'--input',str(self.source/'input.npz'),
                '--predictions',str(prediction),'--output',str(folder/'physics'),
                '--approach-offset',str(approach_offset),'--max-candidates','24'],log)
        report_path=folder/'physics/report.json'
        if artifacts.exists(report_path):
            report=json.loads(artifacts.read_text(report_path));self.private_reports.append(report)
            public={k:report[k] for k in ('execution_success','observed_gripper_not_empty','measured_pad_gap_m','phase','simulated_seconds') if k in report}
            public['planning_rejections']=[dict(stage=row.get('stage'),reason=row.get('reason',row.get('error'))) for row in report.get('planning_rejections',[])[:6]]
        else:public=dict(execution_success=False,error='Motion backend returned no report')
        public['backend_returncode']=result.returncode
        # Old camera-derived candidates are no longer valid after physical motion.
        self.proposals=None
        return self.log('execute_grasp',dict(candidate_ids=candidate_ids,approach_offset=approach_offset),public,start)

    def finish(self):
        """End this attempt. This is an agent decision, not a success score."""
        self.budget();self.finished=True
        return self.log('finish',{},dict(finished=True),time.monotonic())

    def functions(self):
        return dict(get_robot_state=self.state,propose_grasps=self.propose,execute_grasp=self.execute,finish=self.finish)
