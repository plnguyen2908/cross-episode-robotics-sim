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
        self._redocking=None;self._skill_interface=None
        self._observed_pick_target=None

    def move(self,target,purpose,distance,name,lateral_offset=None):
        from .redocking import SensorRedocking
        self.budget();started=time.monotonic()
        if self._redocking is None:self._redocking=SensorRedocking(self.runtime,self.sensors,self.mapper)
        self.proposals=None  # Base motion invalidates all camera-frame grasps.
        options={} if lateral_offset is None else dict(lateral_offset=lateral_offset)
        result=self._redocking.move(target,purpose,distance,self.output/f'{name}_{len(self.calls):03d}',**options)
        return self.log(name,dict(target=target,purpose=purpose,distance=distance),result,started)

    def navigate(self,target,distance=.85):
        """Local visual approach to a visible target, stopping at distance meters. Not exploration or cross-room travel. May still need redock before manipulation."""
        return self.move(target,'approach',distance,'navigate')

    def redock(self,target,purpose='pick',distance=None):
        """Locally align closer and face a visible object/handle/receptacle using RGB-D and SLAM. Default pick distance 0.55 m, other purposes 0.65 m. Use after navigate if too far or poorly aligned. Invalidates old grasps; request fresh proposals afterwards. Does not grasp/place/open."""
        return self.move(target,purpose,distance,'redock')

    def skill_interface(self):
        from .skill_interface import QwenSkillInterface
        from cross_episode_sim.skills.m2t2_interface import M2T2SkillInterface,SkillOutcome
        if self._skill_interface is None:
            def pick(target,progress):
                progress('perceive','running');proposals=self.propose(target)
                ids=[row['id'] for row in proposals['candidates']]
                if not ids:return SkillOutcome(False,'target_not_visible')
                progress('pick','running');result=self.execute(ids)
                return SkillOutcome(result.get('execution_success',False),'')
            def nav(target,progress,**kwargs):
                if kwargs:return SkillOutcome(False,'This component supports local target approach only')
                result=self.navigate(target)
                return SkillOutcome(result['success'],result.get('reason',''))
            self._skill_interface=QwenSkillInterface(M2T2SkillInterface({'pick':pick,'navigate':nav}),self.redock)
        return self._skill_interface

    def list_skills(self):
        """List all atomic schemas and availability. All non-nav/explore schemas have redock=True, redock_target and redock_distance options. This component registers pick and local navigate only."""
        return self.skill_interface().list_skills()

    def execute_skill(self,name,arguments):
        """Execute an available skill. Example: execute_skill('pick', {'target':'bowl','redock':True}). Redock precedes fresh grasp perception. Unavailable skills never use an oracle fallback."""
        self.budget();started=time.monotonic()
        result=self.skill_interface().execute_skill(name,arguments)
        return self.log('execute_skill',dict(name=name,arguments=arguments),result,started)

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
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not tracking.accepted:raise RuntimeError('Current SLAM localization unavailable')
        T=tracking.T_map_base
        _,_,valid=camera_points(camera)
        target_base=None
        # Carry the observed target across camera/view changes, never a simulator ID.
        tracked=getattr(self._redocking,'target_point',None)
        tracked_label=getattr(self._redocking,'target_label',None)
        if tracked_label!=target or tracked is None:
            memory=self._observed_pick_target
            tracked=memory[1] if memory is not None and memory[0]==target else None
        if tracked is not None:
            local=T[:3,:3].T@(tracked-T[:3,3])
            optical=np.linalg.inv(camera.T_base_optical)@np.r_[local,1.]
            if optical[2]>0:
                uv=camera.K@optical[:3];uv=uv[:2]/uv[2]
                if 0<=uv[0]<camera.depth_m.shape[1] and 0<=uv[1]<camera.depth_m.shape[0]:
                    measured=pixel_point(camera,valid,uv)
                    if np.linalg.norm(measured-local)<.15:target_base=measured
        if target_base is None:
            points=MolmoPointer().point(Image.fromarray(camera.rgb),target)['points']
            if not points:return self.log('propose_grasps',dict(target=target),dict(error='Target not visible',candidates=[]),start)
            candidates=[pixel_point(camera,valid,p) for p in points]
            target_base=min(candidates,key=lambda p:np.linalg.norm(T[:3,:3]@p+T[:3,3]-tracked)) if tracked is not None else candidates[0]
        target_map=T[:3,:3]@target_base+T[:3,3]
        if getattr(getattr(self,'approach_options',None),'source_receptacle',None):
            self._observed_pick_target=(target,target_map.copy())
        self.snapshot(folder)
        artifacts.write_text(folder/'report.json',json.dumps(dict(selected_object=dict(label=target,position=target_map.tolist()))))
        with (folder/'perception.log').open('w',buffering=1) as log:
            for command in ([sys.executable,'scripts/capture_m2t2_input.py','--source',str(folder),'--output',str(folder)],
                            [sys.executable,'scripts/m2t2_worker.py','--input',str(folder/'input.npz'),'--output',str(folder/'predictions.json'),'--checkpoint','remote']):
                result=self.runtime.run(command,log)
                if result.returncode:raise RuntimeError('Perception backend failed; no action executed')
        predictions=json.loads(artifacts.read_text(folder/'predictions.json'))
        # Match the existing motion backend's robot-relative ranking before
        # presenting a small candidate set. Near-duplicate high-confidence
        # grasps should not hide other reachable rim directions from Qwen.
        from cross_episode_sim.navigation.shaped_docking import RobotGeometry
        geometry=RobotGeometry(self.runtime.model)
        shoulder=geometry.fk(packet.arm.position)[self.runtime.model.body('robot_0/fr3_link1').id][:3,3]
        ranked=sorted(predictions['candidates'],key=lambda row:np.linalg.norm(np.asarray(row['tcp'])[:3,3]-shoulder)-.10*row['confidence'])
        distinct=[];remaining=[]
        for row in ranked:
            tcp=np.asarray(row['tcp'])
            duplicate=any(np.linalg.norm(tcp[:3,3]-np.asarray(old['tcp'])[:3,3])<.025 and np.linalg.norm(tcp[:3,:3]-np.asarray(old['tcp'])[:3,:3])<.4 for old in distinct)
            (remaining if duplicate or len(distinct)>=8 else distinct).append(row)
        predictions['candidates']=distinct+remaining
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
        self.last_pick_output = folder/'physics'
        prediction=folder/'selected.json'
        artifacts.write_text(prediction,json.dumps({**self.proposals,'candidates':[self.proposals['candidates'][i] for i in candidate_ids]}))
        # Current live state, never a rollback between attempts.
        self.snapshot(self.source)
        with (folder/'execution.log').open('w',buffering=1) as log:
            result=self.runtime.run([sys.executable,'scripts/test_m2t2_pick.py',
                '--source',str(self.source),'--input',str(self.source/'input.npz'),
                '--predictions',str(prediction),'--output',str(folder/'physics'),
                '--approach-offset',str(approach_offset),'--max-candidates','24',
                *self.pick_execution_options(folder)],log)
        report_path=folder/'physics/report.json'
        if artifacts.exists(report_path):
            report=json.loads(artifacts.read_text(report_path));self.private_reports.append(report)
            public={k:report[k] for k in ('execution_success','alignment_completed','alignment_only','front_wrist','observed_gripper_not_empty','measured_pad_gap_m','phase','simulated_seconds') if k in report}
            public['error']=next((row['error'] for row in reversed(report.get('events',[])) if row.get('error')),None)
            public['planning_rejections']=[dict(stage=row.get('stage'),reason=row.get('reason',row.get('error'))) for row in report.get('planning_rejections',[])[:6]]
        else:public=dict(execution_success=False,error='Motion backend returned no report')
        public['backend_returncode']=result.returncode
        # Old camera-derived candidates are no longer valid after physical motion.
        self.proposals=None
        return self.log('execute_grasp',dict(candidate_ids=candidate_ids,approach_offset=approach_offset),public,start)

    def pick_execution_options(self,folder):return []

    def finish(self):
        """End this attempt. This is an agent decision, not a success score."""
        self.budget();self.finished=True
        return self.log('finish',{},dict(finished=True),time.monotonic())

    def functions(self):
        return dict(get_robot_state=self.state,propose_grasps=self.propose,execute_grasp=self.execute,
            navigate=self.navigate,redock=self.redock,list_skills=self.list_skills,
            execute_skill=self.execute_skill,finish=self.finish)
