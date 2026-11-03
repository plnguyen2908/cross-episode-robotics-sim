"""Robot-only collision calibration and sensor-map SE(2) close docking.

No live simulator data enters the planner. Static geometry comes only from the
robot subtree. Pose and arm angles come from SLAM and encoder observations.
"""
from dataclasses import dataclass
import heapq,math
import numpy as np
from scipy.spatial.transform import Rotation
from .local_scan import ScanConfig


def wrap(a):return math.atan2(math.sin(a),math.cos(a))
def turn_rate(error, gain=2.5, cap=.6):
    # Leave room for sensor latency and braking before the requested heading.
    braking=max(0.,math.sqrt((.8*.25)**2+2*.8*abs(error))-.8*.25)
    return math.copysign(min(cap,gain*abs(error),braking),error)
def rotation(a):return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
def transform(p,q):
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();T[:3,3]=p;return T

@dataclass
class RobotBox:
    name:str
    center:np.ndarray
    axes:np.ndarray
    half:np.ndarray

class RobotGeometry:
    """Copies only known robot kinematics and collision shapes, never scene poses."""
    def __init__(self,model,gripper_joint_positions=None):
        import mujoco
        root=model.body('robot_0/base').id;self.root=root;self.bodies={};self.geoms={}
        self.joint_names=tuple(f'robot_0/fr3_joint{i}' for i in range(1,8))
        self.limits=np.array([model.jnt_range[model.joint(n).id] for n in self.joint_names])
        for b in range(model.nbody):
            n=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or ''
            if not n.startswith('robot_0/'):continue
            # Fingers have passive couplings: use a fixed full-stroke envelope.
            if '/gripper/' in n and n!='robot_0/gripper/base' and gripper_joint_positions is None:continue
            joints=[]
            local=transform(model.body_pos[b],model.body_quat[b])
            if b!=root:
                for j in range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b]):
                    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
                    if name not in self.joint_names:
                        if gripper_joint_positions is None or name not in gripper_joint_positions:raise ValueError('Unknown robot joint '+str(name))
                        angle=float(gripper_joint_positions[name])-float(model.qpos0[model.jnt_qposadr[j]])
                        A=np.eye(4)
                        if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:
                            A[:3,:3]=Rotation.from_rotvec(model.jnt_axis[j]*angle).as_matrix()
                            A[:3,3]=model.jnt_pos[j]-A[:3,:3]@model.jnt_pos[j]
                        elif model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:A[:3,3]=model.jnt_axis[j]*angle
                        else:raise ValueError('Unsupported finger joint type')
                        local=local@A
                        continue
                    joints.append((self.joint_names.index(name),model.jnt_axis[j].copy(),model.jnt_pos[j].copy(),float(model.qpos0[model.jnt_qposadr[j]])))
            self.bodies[b]=(int(model.body_parentid[b]),local,joints)
            self.geoms[b]=[]
            for g in range(model.body_geomadr[b],model.body_geomadr[b]+model.body_geomnum[b]):
                if not(model.geom_contype[g] or model.geom_conaffinity[g]):continue
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or n+'/collision'
                self.geoms[b].append((name,transform(model.geom_pos[g],model.geom_quat[g]),model.geom_aabb[g,:3].copy(),model.geom_aabb[g,3:].copy()))
            if n=='robot_0/gripper/base' and gripper_joint_positions is None:
                self.geoms[b].append(('full_stroke_gripper',np.eye(4),np.array([0.,0.,.09]),np.array([.10,.10,.12])))
        from cross_episode_sim.robot.profile import MOBILE_FRANKA
        site=model.site('robot_0/'+MOBILE_FRANKA.tcp_site).id
        self.tcp_body=int(model.site_bodyid[site]);self.tcp_local=transform(model.site_pos[site],model.site_quat[site])
    def fk(self,q):
        poses={self.root:np.eye(4)}
        for b,(parent,local,joints) in self.bodies.items():
            if b==self.root:continue
            T=poses[parent]@local
            for i,axis,pivot,zero in joints:
                A=np.eye(4);A[:3,:3]=Rotation.from_rotvec(axis*(q[i]-zero)).as_matrix();A[:3,3]=pivot-A[:3,:3]@pivot;T=T@A
            poses[b]=T
        return poses
    def boxes(self,q):
        poses=self.fk(q);out=[]
        for b,geoms in self.geoms.items():
            for name,local,center,half in geoms:
                T=poses[b]@local;out.append(RobotBox(name,T[:3,:3]@center+T[:3,3],T[:3,:3],half))
        return out
    def reach(self,target_base,q):
        from scipy.optimize import least_squares
        lo,hi=self.limits[:,0]+1e-5,self.limits[:,1]-1e-5
        for tilt in (0.,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3):
            orientation=Rotation.from_euler('y',tilt).as_matrix()@np.diag([1.,-1.,-1.])
            def residual(v):
                T=self.fk(v)[self.tcp_body]@self.tcp_local
                return np.r_[T[:3,3]-target_base,.2*Rotation.from_matrix(orientation@T[:3,:3].T).as_rotvec()]
            r=least_squares(residual,np.clip(q,lo,hi),bounds=(lo,hi),max_nfev=100)
            e=residual(r.x)
            if np.linalg.norm(e[:3])<.01 and np.linalg.norm(e[3:])<.03:return r.x
        return None

class ObservedShapeChecker:
    def __init__(self,grid,boxes,initial_pose,margin=.035,support_poses=None):
        self.grid=grid;self.boxes=boxes;self.initial=np.asarray(initial_pose).copy();self.margin=margin
        # Floor support belongs to the physical base, not an enlarged collision
        # clearance square or the mast/arm projection. Keep full collision pads.
        base=min(boxes,key=lambda b:b.center[2]-np.abs(b.axes[2])@b.half)
        u=np.linspace(-1.,1.,13);xx,yy=np.meshgrid(u,u)
        local=np.column_stack((xx.ravel()*base.half[0],yy.ravel()*base.half[1],np.full(xx.size,-base.half[2])))
        self.support_local=(local@base.axes.T+base.center)[:,:2]
        self.support_poses=[self.initial] if support_poses is None else [np.asarray(p).copy() for p in support_poses]
        self.detail=None;self.refresh()
    def refresh(self):
        g=self.grid;ij=np.argwhere(g.evidence>=2)
        self.points=np.column_stack((g.origin[0]+(ij[:,1]+.5)*g.config.resolution,g.origin[1]+(ij[:,0]+.5)*g.config.resolution,g.config.min_obstacle_height+(ij[:,2]+.5)*g.config.height_resolution))
        from scipy.spatial import cKDTree
        self.point_index=cKDTree(self.points[:,:2])
        self.spatial_index=cKDTree(self.points)
        self.support_mask=np.zeros(g.floor_views.shape,bool)
        for pose in self.support_poses:self.remember_support(pose)
    def remember_support(self,pose):
        """Cache only the footprint already occupied at an accepted SLAM pose."""
        col,row=self.grid.world_to_cell(self.support_local@rotation(pose[2]).T+pose[:2]).T
        valid=(row>=0)&(row<self.support_mask.shape[0])&(col>=0)&(col<self.support_mask.shape[1])
        self.support_mask[row[valid],col[valid]]=True
    def clear(self,pose):
        xy=np.asarray(pose[:2]);R=rotation(pose[2]);g=self.grid
        pad=self.margin+np.sqrt(2*g.config.resolution**2+g.config.height_resolution**2)/2
        for b in self.boxes:
            center=b.center.copy();center[:2]=b.center[:2]@R.T+xy
            # Bounding sphere contains the entire padded oriented box. Keep
            # the same exact box test and local range as the former scan.
            ids=self.spatial_index.query_ball_point(center,np.linalg.norm(b.half+pad)+1e-12)
            points=self.points[ids].copy()
            points=points[np.linalg.norm(points[:,:2]-xy,axis=1)<1.2]
            points[:,:2]=(points[:,:2]-xy)@R
            inside=np.all(np.abs((points-b.center)@b.axes)<=b.half+pad,axis=1)
            if inside.any():
                self.detail={'reason':'observed_3d_collision','part':b.name,'points_base':points[inside][:8].tolist()};return False
        # Actual base support, not the mast/arm's projected radius.
        samples=self.support_local@R.T+xy;col,row=g.world_to_cell(samples).T
        valid=(row>=0)&(row<g.floor_views.shape[0])&(col>=0)&(col<g.floor_views.shape[1])
        if not valid.all():self.detail={'reason':'unknown_support_boundary'};return False
        # Floor evidence is cell-based. Rasterize the known initial support in
        # that same grid, rather than comparing noisy SLAM coordinates to an
        # exact floating-point rectangle inside those cells. Occupancy is still
        # checked above, including inside the initial support cells.
        initial=self.support_mask[row,col]
        if np.any((g.floor_views[row,col]<g.config.minimum_floor_views)&~initial):
            missing=(g.floor_views[row,col]<g.config.minimum_floor_views)&~initial
            self.detail={'reason':'unobserved_base_support','points_map':samples[missing][:8].tolist()};return False
        return True
    def swept(self,a,b):
        d=wrap(b[2]-a[2]);n=max(1,int(np.ceil(np.linalg.norm(np.asarray(b[:2])-a[:2])/.025)),int(np.ceil(abs(d)/.04)))
        return all(self.clear(np.r_[np.asarray(a[:2])+(np.asarray(b[:2])-a[:2])*t,a[2]+d*t]) for t in np.linspace(0,1,n+1))

class ShapedDocking:
    """A* over base position AND heading; all execution is checked online."""
    def __init__(self,grid,geometry,max_nodes=6000):
        self.grid=grid;self.geometry=geometry;self.max_nodes=max_nodes
        self.support_poses=[]
        self.target=None;self.terminal=None;self.path=None;self.events=[];self.velocity=np.zeros(2);self.started=None;self.last_frame=None;self.reach_joints=None
    def make_checker(self,pose,q):
        # Retain only footprints already occupied by the robot, estimated by
        # SLAM. Replanning must not forget the starting/previous support cells.
        # This never clears occupancy and does not extrapolate unseen floor.
        self.record_support(pose)
        return ObservedShapeChecker(self.grid,self.geometry.boxes(q),pose,support_poses=list(getattr(self.grid,'inherited_support_poses',[]))+self.support_poses+[pose])
    def record_support(self,pose):
        """Remember actual estimated footprints, including room-search motion."""
        pose=np.asarray(pose)
        if not self.support_poses or np.linalg.norm(pose[:2]-self.support_poses[-1][:2])>.05 or abs(wrap(pose[2]-self.support_poses[-1][2]))>.1:
            self.support_poses.append(pose.copy())
    def reset(self,target):
        self.target=target;self.terminal=None;self.path=None;self.started=None;self.velocity[:]=0;self.approaching=False;self.curve=None;self.clearance_retries=0;self.reobserve=None;self.retreat_attempts=0;self.surface_outward=None;self.alignment_refinements=0
    def plan(self,pose,q):
        self.checker=self.make_checker(pose,q)
        if not self.checker.clear(pose):
            self.events.append(dict(event='shape_start_blocked',detail=self.checker.detail));return False
        self.initial=pose.copy();R=rotation(pose[2]);resolution=.1
        def world(s):return np.r_[pose[:2]+R@(np.array(s[:2])*resolution),pose[2]+s[2]*math.pi/4]
        target=self.target.position
        def h(s):return max(0.,np.linalg.norm(world(s)[:2]-target[:2])-.70)
        start=(0,0,0);queue=[(h(start),0.,start)];cost={start:0.};parent={};cache={}
        self.reach_joints=None;visited=0;best=None;candidate_cache={};candidates=0
        while queue and visited<self.max_nodes:
            priority,dist,s=heapq.heappop(queue)
            if best is not None and priority>best[0]:break
            if dist>cost[s]+1e-9:continue
            visited+=1;p=world(s);offset=target[:2]-p[:2];gap=np.linalg.norm(offset)
            if .50<=gap<=.70 and abs(wrap(math.atan2(offset[1],offset[0])-p[2]))<.4:
                # Same candidate-and-reach principle as oracle docking, but all
                # target and obstacle positions come from the observed map.
                # The lattice is for routing only; final facing is continuous.
                dock=p.copy();dock[2]=math.atan2(offset[1],offset[0])
                key=s[:2]
                if key not in candidate_cache:
                    pregrasp=target.copy();pregrasp[2]+=.08
                    pregrasp[:2]=rotation(dock[2]).T@(pregrasp[:2]-dock[:2])
                    candidate_cache[key]=self.geometry.reach(pregrasp,q) if self.checker.clear(dock) else None
                solution=candidate_cache[key]
                if solution is not None and self.checker.swept(p,dock):
                    candidates+=1
                    score=dist+.12*abs(wrap(dock[2]-p[2]))/(math.pi/4)+.5*abs(gap-.60)
                    if best is None or score<best[0]:best=(score,s,dock,solution)
            k=s[2];dx,dy=int(round(math.cos(k*math.pi/4))),int(round(math.sin(k*math.pi/4)))
            neighbors=[((s[0],s[1],(k+1)%8),.12),((s[0],s[1],(k-1)%8),.12),((s[0]+dx,s[1]+dy,k),math.hypot(dx,dy)*resolution),((s[0]-dx,s[1]-dy,k),math.hypot(dx,dy)*resolution+.04)]
            for nxt,weight in neighbors:
                if abs(nxt[0])>25 or abs(nxt[1])>25:continue
                ng=dist+weight
                if ng>=cost.get(nxt,float('inf')):continue
                key=(s,nxt)
                if key not in cache:cache[key]=self.checker.swept(p,world(nxt))
                if not cache[key]:continue
                cost[nxt]=ng;parent[nxt]=s;heapq.heappush(queue,(ng+h(nxt),ng,nxt))
        if best is not None:
            score,s,dock,solution=best;states=[s]
            while s in parent:s=parent[s];states.append(s)
            self.path=[world(k) for k in states[::-1]][1:]+[dock]
            self.goal=dock;self.reach_joints=solution
            self.events.append(dict(event='shape_astar',visited=visited,candidates=candidates,
                goal=dock.tolist(),route=[v.tolist() for v in self.path],pregrasp_ik=True,
                selection='route cost + final turn + preferred reach distance',continuous_facing=True))
            return True
        self.events.append(dict(event='shape_plan_failed',visited=visited,detail=self.checker.detail));return False
    def smooth_docking(self,pose):
        """Forward cubic Bezier shortcut; retain original route if uncertified."""
        distance=np.linalg.norm(self.goal[:2]-pose[:2]);self.curve=None
        if distance<.15:return
        for scale in (.4,.65,.25):
            points=np.array([pose[:2],pose[:2]+scale*distance*np.array([math.cos(pose[2]),math.sin(pose[2])]),
                self.goal[:2]-scale*distance*np.array([math.cos(self.goal[2]),math.sin(self.goal[2])]),self.goal[:2]])
            t=np.linspace(0,1,max(81,int(distance/.005)))[:,None];u=1-t
            xy=u**3*points[0]+3*u*u*t*points[1]+3*u*t*t*points[2]+t**3*points[3]
            derivative=3*u*u*(points[1]-points[0])+6*u*t*(points[2]-points[1])+3*t*t*(points[3]-points[2])
            if np.min(np.linalg.norm(derivative,axis=1))<.005:continue
            yaw=np.unwrap(np.arctan2(derivative[:,1],derivative[:,0]));curve=np.c_[xy,yaw]
            length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
            if length[-1]>2*distance:continue
            if not all(self.checker.swept(a,b) for a,b in zip(curve[:-1],curve[1:])):continue
            self.curve=curve;self.curve_length=length;self.curve_index=0
            self.events.append(dict(event='smooth_docking',original_waypoints=len(self.path),length_m=float(length[-1]),method='forward_cubic_bezier',scale=scale));return
        self.events.append(dict(event='smoothing_rejected',reason='No swept-clear forward curve; retained checked route'))

    def curve_command(self,pose):
        remaining_points=self.curve[self.curve_index:,:2]
        self.curve_index+=int(np.argmin(np.linalg.norm(remaining_points-pose[:2],axis=1)))
        remaining=self.curve_length[-1]-self.curve_length[self.curve_index]
        gap=np.linalg.norm(self.goal[:2]-pose[:2])
        if gap<.025:
            # Inside the existing 3 cm docking tolerance, turn to the object
            # rather than pursuing a tiny lateral residual with a large turn.
            final_yaw=self.goal[2]
            if not getattr(self,'approaching',False) and self.target is not None and getattr(self,'surface_outward',None) is None:
                delta=self.target.position[:2]-pose[:2]
                final_yaw=math.atan2(delta[1],delta[0])
            error=wrap(final_yaw-pose[2])
            if abs(error)<.015:self.path=[];return np.zeros(2)
            return np.array([0.,turn_rate(error)])
        ahead=min(len(self.curve)-1,int(np.searchsorted(self.curve_length,self.curve_length[self.curve_index]+.09)))
        delta=self.curve[ahead,:2]-pose[:2];angle=wrap(math.atan2(delta[1],delta[0])-pose[2])
        curvature=2*math.sin(angle)/max(np.linalg.norm(delta),.03)
        # Brake before the end, including camera latency and the safety horizon.
        # Otherwise a short curved dock can be geometrically clear but its
        # stopping envelope overshoots into an unobserved neighboring cell.
        end_budget=max(0.,gap-.022)
        braking=max(0.,math.sqrt((.4*.35)**2+2*.4*end_budget)-.4*.35)
        speed=min(.30,braking,math.sqrt(.8*max(remaining,.001)),2.0*gap,.5/max(abs(curvature),.01))
        if abs(angle)>.6:speed=0.
        weight=float(np.clip((gap-.02)/.08,0.,1.))
        final_error=wrap(self.goal[2]-pose[2])
        steering=final_error+weight*wrap(angle-final_error)
        return np.array([speed,turn_rate(steering,gain=1.+speed*2/max(np.linalg.norm(delta),.03))])

    def plan_candidates(self,pose,q,candidates,source="shared_oracle_candidate"):
        """Try the shared oracle's ordered candidates using observed swept clearance."""
        self.checker=self.make_checker(pose,q)
        self.initial=pose.copy();self.path=None
        if not self.checker.clear(pose):
            self.events.append(dict(event='start_blocked',detail=self.checker.detail));return False
        R=rotation(pose[2])
        def world(s):return np.r_[pose[:2]+R@(np.array(s[:2])*.1),pose[2]+s[2]*math.pi/4]
        rejected=[]
        for index,goal in enumerate(candidates):
            lattice_goal=R.T@(goal[:2]-pose[:2])
            if np.max(np.abs(lattice_goal))>2.15:
                rejected.append(dict(index=index,detail={'reason':'outside_local_search_window'}));continue
            if not self.checker.clear(goal):
                rejected.append(dict(index=index,goal=goal.tolist(),detail=self.checker.detail));continue
            def heuristic(s):return max(0.,np.linalg.norm(world(s)[:2]-goal[:2])-.15)
            start=(0,0,0);queue=[(heuristic(start),0.,start)];cost={start:0.};parent={};visited=0
            while queue and visited<self.max_nodes:
                _,dist,s=heapq.heappop(queue)
                if dist>cost[s]+1e-9:continue
                visited+=1;p=world(s);delta=goal[:2]-p[:2]
                if np.linalg.norm(delta)<.15:
                    heading=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>.005 else p[2]
                    connector=[np.r_[p[:2],heading],np.r_[goal[:2],heading],goal]
                    if all(self.checker.swept(a,b) for a,b in zip([p]+connector,connector)):
                        states=[s]
                        while s in parent:s=parent[s];states.append(s)
                        self.path=[world(k) for k in states[::-1]][1:]+connector;self.goal=goal
                        self.events.append(dict(event=source,index=index,goal=goal.tolist(),visited=visited));self.smooth_docking(pose);return True
                k=s[2];dx,dy=int(round(math.cos(k*math.pi/4))),int(round(math.sin(k*math.pi/4)))
                for nxt,weight in [((s[0],s[1],(k+1)%8),.12),((s[0],s[1],(k-1)%8),.12),((s[0]+dx,s[1]+dy,k),math.hypot(dx,dy)*.1),((s[0]-dx,s[1]-dy,k),math.hypot(dx,dy)*.1+.04)]:
                    ng=dist+weight
                    if abs(nxt[0])>20 or abs(nxt[1])>20 or ng>=cost.get(nxt,float('inf')):continue
                    if not self.checker.swept(p,world(nxt)):continue
                    cost[nxt]=ng;parent[nxt]=s;heapq.heappush(queue,(ng+heuristic(nxt),ng,nxt))
        self.events.append(dict(event='dock_candidates_unverified',rejected=rejected))
        return False

    def plan_advance(self,pose,q):
        """Advance only through observed space, then reconsider the full dock."""
        self.checker=self.make_checker(pose,q)
        delta=self.target.position[:2]-pose[:2];gap=np.linalg.norm(delta)
        if gap<=.63:return False
        heading=math.atan2(delta[1],delta[0])
        distances=(.08,.15,.25) if gap<1.3 and abs(wrap(heading-pose[2]))>.1 else (.25,.15,.08)
        if gap>1.6:
            # Longer observed corridors need not be split into 25 cm stop/start
            # motions. Leave the close-range band to the precise dock planner.
            budget=gap-1.3
            longer=sorted({min(1.,budget),.75,.50},reverse=True)
            distances=tuple(x for x in longer if .25<x<=budget)+distances
        for distance in distances:
            for offset in (0.,wrap(pose[2]-heading),-.25,.25,-.5,.5):
                yaw=heading+offset;turn=np.r_[pose[:2],yaw]
                goal=np.r_[pose[:2]+min(distance,gap-.60)*np.array([math.cos(yaw),math.sin(yaw)]),yaw]
                if np.linalg.norm(goal[:2]-self.target.position[:2])>=gap-.02:continue
                if not self.checker.swept(pose,turn) or not self.checker.swept(turn,goal):continue
                self.path=[turn,goal];self.goal=goal;self.curve=None;self.approaching=True
                self.events.append(dict(event='rolling_advance',distance_m=float(np.linalg.norm(goal[:2]-pose[:2])),goal=goal.tolist()))
                self.smooth_docking(pose)
                return True
        self.events.append(dict(event='rolling_advance_blocked',detail=self.checker.detail))
        return False

    def plan_observe(self,pose,q):
        """Turn incrementally toward the target to reveal its approach floor."""
        delta=self.target.position[:2]-pose[:2]
        if np.linalg.norm(delta)>1.6:return False
        error=wrap(math.atan2(delta[1],delta[0])-pose[2])
        if abs(error)<.08:return False
        self.checker=self.make_checker(pose,q)
        for angle in (.30,.20,.10,.05):
            goal=pose.copy();goal[2]+=math.copysign(min(angle,abs(error)),error)
            if not self.checker.swept(pose,goal):continue
            self.path=[goal];self.goal=goal;self.curve=None;self.approaching=True
            self.events.append(dict(event='observe_approach_floor',angle_rad=float(goal[2]-pose[2])))
            return True
        return False

    def plan_retreat(self,pose,q):
        """Short checked undocking to gain maneuvering space; never blind reverse."""
        if self.retreat_attempts>=2 or np.linalg.norm(self.target.position[:2]-pose[:2])>1.3:return False
        self.checker=self.make_checker(pose,q)
        for distance in (.18,.12,.06):
            goal=pose.copy();goal[:2]-=distance*np.array([math.cos(pose[2]),math.sin(pose[2])])
            if not self.checker.swept(pose,goal):continue
            self.retreat_attempts+=1;self.path=[goal];self.goal=goal;self.curve=None;self.approaching=True
            self.events.append(dict(event='observed_undock',distance_m=distance,goal=goal.tolist()))
            return True
        return False

    def step(self,packet,tracking,now):
        def stop(status):self.velocity[:]=0;return dict(status=status,forward=0.,yaw=0.)
        if self.terminal:return stop(self.terminal)
        if not tracking.accepted:return stop('tracking_lost')
        if not 0<=now-packet.timestamp<=.25:return stop('stale_observation')
        if not np.allclose(packet.arm.position,ScanConfig().stowed_arm,atol=.05):return stop('arm_not_stowed')
        T=tracking.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
        if self.started is None:self.started=now
        if now-self.started>45:self.terminal='shape_dock_timeout';return stop(self.terminal)
        if getattr(self,'reobserve',None) is not None:
            stamp,previous,stable=self.reobserve
            if packet.timestamp<=stamp:return stop('clearance_reobserve')
            settled=np.linalg.norm(pose[:2]-previous[:2])<.005 and abs(wrap(pose[2]-previous[2]))<.01
            stable=stable+1 if settled else 0
            self.reobserve=(packet.timestamp,pose.copy(),stable)
            if stable<2:return stop('clearance_reobserve')
            self.reobserve=None;self.path=None;self.curve=None
        if self.path is None and np.linalg.norm(self.target.position[:2]-pose[:2])>1.6:
            self.plan_advance(pose,packet.arm.position)
        if self.path is None:
            from .docking_candidates import observed_surface,frontal_surface_docks
            try:
                lo,hi,outward=observed_surface(packet,T,self.target.position)
                self.checker=self.make_checker(pose,packet.arm.position)
                candidates=[];solutions={};candidate_rejections=[]
                for candidate in frontal_surface_docks(self.target.position,lo,hi,outward):
                    if not self.checker.clear(candidate):
                        candidate_rejections.append(dict(goal=candidate.tolist(),detail=self.checker.detail));continue
                    pregrasp=self.target.position.copy();pregrasp[2]+=.08
                    pregrasp[:2]=rotation(candidate[2]).T@(pregrasp[:2]-candidate[:2])
                    solution=self.geometry.reach(pregrasp,packet.arm.position)
                    if solution is not None:
                        candidates.append(candidate);solutions[tuple(candidate)]=solution
                if not candidates:self.events.append(dict(event='pickup_candidates_rejected',rejected=candidate_rejections))
                accepted=self.plan_candidates(pose,packet.arm.position,candidates,source="frontal_pickup_candidate")
                if accepted:
                    self.reach_joints=solutions[tuple(self.goal)]
                    self.surface_outward=outward.copy();tangent=np.array([outward[1],-outward[0]])
                    self.events.append(dict(event='pickup_alignment',lateral_offset_m=float((self.goal[:2]-self.target.position[:2])@tangent),outward=outward.tolist()))
            except ValueError as exc:
                self.events.append(dict(event='surface_unverified',reason=str(exc)));accepted=False
            self.approaching=False
            if not accepted:accepted=self.plan_observe(pose,packet.arm.position)
            if not accepted:accepted=self.plan_advance(pose,packet.arm.position)
            if not accepted:accepted=self.plan_retreat(pose,packet.arm.position)
            if not accepted:self.terminal='shape_route_unverified';return stop(self.terminal)
        if packet.timestamp!=self.last_frame:
            self.checker.refresh();self.checker.remember_support(pose);self.last_frame=packet.timestamp
        while self.path and np.linalg.norm(self.path[0][:2]-pose[:2])<.025 and abs(wrap(self.path[0][2]-pose[2]))<(.015 if len(self.path)==1 else .045):self.path.pop(0)
        if not self.path and getattr(self,'approaching',False):
            self.path=None;self.curve=None;self.approaching=False
            return stop('rolling_update_map')
        if not self.path:
            if getattr(self,'surface_outward',None) is not None:
                try:
                    from .docking_candidates import observed_surface
                    _,_,normal=observed_surface(packet,T,self.target.position)
                except ValueError:normal=self.surface_outward
                tangent=np.array([normal[1],-normal[0]])
                angle=abs(wrap(math.atan2(-normal[1],-normal[0])-pose[2]))
                lateral=abs(float((pose[:2]-self.target.position[:2])@tangent))
                if angle>.05 or lateral>.04:
                    self.events.append(dict(event='observed_alignment_replan',normal_error_rad=angle,lateral_error_m=lateral))
                    if self.alignment_refinements>=2:self.terminal='shape_alignment_unverified';return stop(self.terminal)
                    self.alignment_refinements+=1;self.path=None;self.curve=None
                    return stop('refine_observed_alignment')
            gap=np.linalg.norm(pose[:2]-self.target.position[:2])
            alignment=abs(wrap(math.atan2(self.target.position[1]-pose[1],self.target.position[0]-pose[0])-pose[2]))
            self.events.append(dict(event='final_alignment',distance_m=float(gap),heading_error_rad=float(alignment)))
            self.terminal='shape_docked' if np.linalg.norm(pose[:2]-self.goal[:2])<.03 and alignment<.05 else 'shape_goal_tracking_error'
            return stop(self.terminal)
        goal=self.path[0];delta=goal[:2]-pose[:2];body=rotation(pose[2]).T@delta
        yaw_error=wrap(goal[2]-pose[2])
        if np.linalg.norm(delta)<.025:desired=np.array([0.,turn_rate(yaw_error,gain=1.8)])
        else:
            direction=1 if body[0]>=0 else -1
            angle=wrap(math.atan2(direction*delta[1],direction*delta[0])-pose[2])
            desired=np.array([direction*min(.25,1.5*np.linalg.norm(delta))*max(0.,math.cos(angle)) if abs(angle)<.25 else 0.,turn_rate(angle,gain=1.8)])
        if getattr(self,'curve',None) is not None:desired=self.curve_command(pose)
        limit=np.array([.4,.8])*.02
        nominal=self.velocity+np.clip(desired-self.velocity,-limit,limit)
        # Select a slower reachable command if the nominal braking envelope
        # reaches newly observed geometry. Never weaken the clearance check.
        for scale in (1.,.75,.5,.25,0.):
            cmd=self.velocity+np.clip(nominal*scale-self.velocity,-limit,limit)
            p=pose.copy();safe=True
            for t in np.arange(0,.25+max(abs(cmd[0])/.4,abs(cmd[1])/.8)+.02,.02):
                v=np.sign(cmd[0])*max(0,abs(cmd[0])-.4*max(0,t-.25));w=np.sign(cmd[1])*max(0,abs(cmd[1])-.8*max(0,t-.25))
                p[:2]+=v*.02*np.array([math.cos(p[2]),math.sin(p[2])]);p[2]+=w*.02
                if not self.checker.clear(p):safe=False;break
            if safe and scale==0. and np.linalg.norm(cmd)<1e-6 and np.linalg.norm(nominal)>1e-3:
                # A zero command is collision-free but makes no progress. Do
                # not report it as motion forever; use the existing measured
                # settle/re-observe/replan recovery below.
                safe=False
            if safe:
                self.velocity=cmd
                return dict(status='rolling_approach' if getattr(self,'approaching',False) else 'shape_docking',forward=float(cmd[0]),yaw=float(cmd[1]))
        self.events.append(dict(event='shape_clearance_stop',detail=self.checker.detail,pose=pose.tolist(),predicted_pose=p.tolist(),command=cmd.tolist()))
        if self.clearance_retries<3:
            self.clearance_retries+=1;self.reobserve=(packet.timestamp,pose.copy(),0)
            self.events.append(dict(event='clearance_replan',attempt=self.clearance_retries))
            return stop('clearance_reobserve')
        self.terminal='shape_clearance_stop';return stop(self.terminal)
