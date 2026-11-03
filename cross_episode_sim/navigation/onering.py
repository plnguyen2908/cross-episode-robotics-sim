"""Native OneRING policy process and sensor-checked discrete base actions."""
import json,math,os,subprocess
from pathlib import Path
import numpy as np
from .house_exploration import HouseExplorer,HouseExplorationConfig
from .local_scan import ScanConfig

class OneRingProcess:
    def __init__(self,checkpoint,output,device='cuda:0',seed=0,max_steps=600,root=None):
        root=Path(root or Path(__file__).resolve().parents[2]).resolve()
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.log=open(self.output/'worker.log','w');self.count=0
        env=os.environ.copy();env['PYTHONPATH']=str(root/'external/OneRING')
        self.process=subprocess.Popen([str(root/'.venv-ring/bin/python'),'-u',str(root/'scripts/onering_worker.py'),'--checkpoint',str(Path(checkpoint).resolve()),'--device',device,'--seed',str(seed),'--max-steps',str(max_steps)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,cwd=root)
        try:self.metadata=self._read()
        except Exception:self.close();raise
        (self.output/'model.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),device=device,seed=seed,max_steps=max_steps,**self.metadata),indent=2))
    def _read(self):
        for line in self.process.stdout:
            if line.startswith('RING_RESULT '):
                value=json.loads(line[len('RING_RESULT '):])
                if not value['ok']:raise RuntimeError(value['error'])
                return value
            self.log.write(line);self.log.flush()
        raise RuntimeError(f'OneRING worker exited; see {self.output}/worker.log')
    def _request(self,value):
        self.process.stdin.write(json.dumps(value)+'\n');self.process.stdin.flush();return self._read()
    def reset(self):return self._request(dict(reset=True))
    def predict(self,packet,goal):
        path=self.output/'current_rgb.npz'
        np.savez(path,navigation=packet.cameras['navigation'].rgb,wrist=packet.cameras['wrist'].rgb)
        result=self._request(dict(frame=str(path),goal=goal));self.count+=1
        with open(self.output/'predictions.jsonl','a') as f:f.write(json.dumps(dict(timestamp=packet.timestamp,goal=goal,**result))+'\n')
        return result['action']
    def close(self):
        if hasattr(self,'process'):
            if self.process.poll() is None:
                try:self.process.stdin.write('{"close":true}\n');self.process.stdin.flush();self.process.wait(timeout=10)
                except (BrokenPipeError,subprocess.TimeoutExpired):self.process.terminate();self.process.wait(timeout=10)
            self.process.stdin.close();self.process.stdout.close()
        if hasattr(self,'log'):self.log.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

# Upstream SPOCV1: 20 cm translation, 30/6 degree turns. Simulator yaw has
# positive left turns, unlike upstream clockwise-positive theta.
BASE_ACTIONS={'m':(.2,0.),'b':(-.2,0.),'l':(0.,math.pi/6),'r':(0.,-math.pi/6),'ls':(0.,math.pi/30),'rs':(0.,-math.pi/30)}
ALIASES={'move_ahead':'m','move_back':'b','rotate_left':'l','rotate_right':'r','rotate_left_small':'ls','rotate_right_small':'rs','done':'end'}

class IncrementalMotionGuard(HouseExplorer):
    """Check newly swept space, without requiring an explored surrounding disk.

    Assumes the configured stowed circular envelope is clear at the initial
    parked pose. This is a local occupied-envelope prior, not map evidence.
    Observed obstacles always override it. New swept area needs observed floor;
    reverse motion receives exactly the same check. Live depth remains active.
    """
    def _command_clear(self,xy,yaw,command):
        grid=self.grid;cfg=self.config;res=grid.config.resolution
        if not hasattr(self,"initial_envelope_center"):self.initial_envelope_center=xy.copy()
        radius=cfg.footprint_radius+.04
        horizon=cfg.sensor_timeout+abs(command[0])/cfg.acceleration
        p=xy.copy();heading=yaw;local=np.zeros(2);local_yaw=0.
        for t in np.arange(0.,horizon+.02,.02):
            v=np.sign(command[0])*max(0.,abs(command[0])-cfg.acceleration*max(0.,t-cfg.sensor_timeout))
            p+=v*.02*np.array([math.cos(heading),math.sin(heading)])
            heading+=command[1]*.02
            local+=v*.02*np.array([math.cos(local_yaw),math.sin(local_yaw)])
            local_yaw+=command[1]*.02
            row,col=self.cell(p);n=int(np.ceil(radius/res))+1
            rr,cc=np.mgrid[row-n:row+n+1,col-n:col+n+1]
            centers=grid.origin+(np.stack((cc,rr),-1)+.5)*res
            swept=np.linalg.norm(centers-p,axis=-1)<=radius+res*.7072
            valid=(rr>=0)&(rr<grid.floor_views.shape[0])&(cc>=0)&(cc<grid.floor_views.shape[1])
            if np.any(swept&~valid):self.stop_detail={'reason':'unmapped_swept_boundary'};return False
            r,c=rr[swept],cc[swept]
            if np.any(self.obstacles[r,c]):self.stop_detail={'reason':'observed_swept_obstacle'};return False
            initial=np.linalg.norm(centers[swept]-self.initial_envelope_center,axis=-1)<=radius+res*.7072
            observed=grid.floor_views[r,c]>=grid.config.minimum_floor_views
            if np.any(~initial&~observed):
                self.stop_detail={'reason':'unobserved_swept_space','position':p.tolist()};return False
            # Sub-cell leading-edge samples prevent small commands creeping
            # through unknown cells before their centers cross a grid boundary.
            angles=np.linspace(0.,2*np.pi,128,endpoint=False)
            edge=p+radius*np.column_stack((np.cos(angles),np.sin(angles)))
            new=edge[np.linalg.norm(edge-self.initial_envelope_center,axis=1)>radius+.02]
            if len(new):
                ec,er=grid.world_to_cell(new).T
                valid=(er>=0)&(er<grid.floor_views.shape[0])&(ec>=0)&(ec<grid.floor_views.shape[1])
                if not valid.all() or np.any(grid.floor_views[er,ec]<grid.config.minimum_floor_views):
                    self.stop_detail={'reason':'unobserved_leading_edge'};return False
            if len(self.near_points) and np.any(np.linalg.norm(self.near_points[:,:2]-local,axis=1)<radius):
                self.stop_detail={'reason':'depth_braking_arc'};return False
        return True

class OneRingLocalNavigator:
    """Incremental execution at 50 Hz, preserving native policy history.

    `local_done` is a policy suggestion, never a manipulation-success check.
    Unknown or non-navigation actions stop; no manipulation command is executed.
    """
    def __init__(self,grid,policy,goal,config=HouseExplorationConfig()):
        self.policy=policy;self.goal=goal;self.guard=IncrementalMotionGuard(grid,config)
        self.active=None;self.start=None;self.started=None;self.last_time=None;self.events=[];self.last_prediction_stamp=None
        self.shape_checker_factory=None
    def reset_motion(self):
        self.active=None;self.start=None;self.started=None;self.guard.velocity=np.zeros(2);self.last_time=None;self.last_prediction_stamp=None
    def step(self,packet,tracking,now):
        stop=self.guard._stop
        if not tracking.accepted:return stop('tracking_lost')
        if not 0<=now-packet.timestamp<=self.guard.config.sensor_timeout:return stop('stale_observation')
        if not np.allclose(packet.arm.position,self.guard.config.arm_posture,atol=.05):return stop('arm_not_stowed')
        if self.shape_checker_factory is None and not np.allclose(packet.arm.position,ScanConfig().stowed_arm,atol=.05):return stop('carrying_requires_shape_checker')
        T=tracking.T_map_base;xy=T[:2,3];yaw=math.atan2(T[1,0],T[0,0])
        self.guard.layers(xy);self.guard._depth(packet.cameras['navigation'])
        dt=.02 if self.last_time is None else min(.02,max(0.,now-self.last_time));self.last_time=now
        if self.active is None:
            if self.last_prediction_stamp==packet.timestamp:return stop('await_fresh_ring_observation')
            self.last_prediction_stamp=packet.timestamp
            action=self.policy.predict(packet,self.goal);action=ALIASES.get(action,action)
            self.events.append(dict(time=now,action=action))
            if action in {'end','sub_done'}:return stop('local_done')
            if action not in BASE_ACTIONS:return stop('unsupported_ring_action')
            self.active=action;self.start=np.r_[xy,yaw];self.started=now
        travel,turn=BASE_ACTIONS[self.active]
        projected=float(np.dot(xy-self.start[:2],[math.cos(self.start[2]),math.sin(self.start[2])]))
        turned=math.atan2(math.sin(yaw-self.start[2]),math.cos(yaw-self.start[2]))
        remaining=travel-projected if travel else turn-turned
        if abs(remaining)<(.015 if travel else .025):self.active=None;return stop('ring_action_complete')
        if now-self.started>5.:self.active=None;return stop('ring_action_timeout')
        cfg=self.guard.config
        desired=np.array([np.clip(1.5*remaining,-cfg.speed,cfg.speed),0.]) if travel else np.array([0.,np.clip(1.8*remaining,-cfg.yaw_speed,cfg.yaw_speed)])
        command=self.guard.velocity+np.clip(desired-self.guard.velocity,-np.array([cfg.acceleration,cfg.yaw_acceleration])*dt,np.array([cfg.acceleration,cfg.yaw_acceleration])*dt)
        clear=self.guard._command_clear(xy,yaw,command) if self.shape_checker_factory is None else self._shape_command_clear(packet,xy,yaw,command)
        if not clear:
            self.active=None
            self.events.append(dict(time=now,event='blocked',detail=self.guard.stop_detail))
            result=stop('ring_collision_stop');result['detail']=self.guard.stop_detail
            return result
        self.guard.velocity=command
        return dict(status='ring_'+self.active,forward=float(command[0]),yaw=float(command[1]))

    def face_point(self,packet,tracking,now,point):
        """Checked, acceleration-limited orientation toward an observed hint."""
        stop=self.guard._stop
        if not tracking.accepted:return stop('tracking_lost')
        cfg=self.guard.config
        if not 0<=now-packet.timestamp<=cfg.sensor_timeout:return stop('stale_observation')
        if not np.allclose(packet.arm.position,cfg.arm_posture,atol=.05):return stop('arm_not_stowed')
        T=tracking.T_map_base;xy=T[:2,3];yaw=math.atan2(T[1,0],T[0,0])
        delta=np.asarray(point)[:2]-xy
        error=math.atan2(delta[1],delta[0])-yaw
        error=math.atan2(math.sin(error),math.cos(error))
        if abs(error)<.04:return stop('viewpoint_facing_ready')
        self.guard.layers(xy);self.guard._depth(packet.cameras['navigation'])
        dt=.02 if self.last_time is None else min(.02,max(0.,now-self.last_time));self.last_time=now
        desired=np.array([0.,np.clip(1.8*error,-cfg.yaw_speed,cfg.yaw_speed)])
        limits=np.array([cfg.acceleration,cfg.yaw_acceleration])*dt
        command=self.guard.velocity+np.clip(desired-self.guard.velocity,-limits,limits)
        clear=(self.guard._command_clear(xy,yaw,command) if self.shape_checker_factory is None
               else self._shape_command_clear(packet,xy,yaw,command))
        if not clear:return stop('viewpoint_turn_blocked')
        self.guard.velocity=command
        return dict(status='face_observed_search_hint',forward=float(command[0]),yaw=float(command[1]))

    def _shape_command_clear(self,packet,xy,yaw,command):
        """Use the same robot envelope for search and docking, including mast.

        Height-aware checks avoid treating a countertop as a floor-level wall.
        Both translation and rotation include sensor latency and braking travel.
        """
        checker=self.shape_checker_factory(np.r_[xy,yaw],packet.arm.position)
        cfg=self.guard.config;p=np.r_[xy,yaw]
        horizon=cfg.sensor_timeout+max(abs(command[0])/cfg.acceleration,abs(command[1])/cfg.yaw_acceleration)
        previous=p.copy()
        for t in np.arange(0.,horizon+.02,.02):
            elapsed=max(0.,t-cfg.sensor_timeout)
            v=np.sign(command[0])*max(0.,abs(command[0])-cfg.acceleration*elapsed)
            w=np.sign(command[1])*max(0.,abs(command[1])-cfg.yaw_acceleration*elapsed)
            p[:2]+=v*.02*np.array([math.cos(p[2]),math.sin(p[2])]);p[2]+=w*.02
            if not checker.swept(previous,p):self.guard.stop_detail=checker.detail;return False
            previous=p.copy()
        return True
