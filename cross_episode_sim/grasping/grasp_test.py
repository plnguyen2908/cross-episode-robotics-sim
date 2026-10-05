"""One-object physical grasp qualification, before navigation integration.

Candidate poses come from native collision-mesh bounds. Local Jacobian IK is
used for this parked-base component test; this is not the cuRobo integration.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,traceback
from pathlib import Path
import numpy as np
import mujoco,robosuite,robocasa
from scipy.spatial.transform import Rotation
from cross_episode_sim.grasping.robosuite_tidybot import TidyBotFranka,controller_config,TUCK
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.utils.env_utils import set_robot_to_position

class TidyBotGraspTest(Kitchen):
    test_asset = 'lightwheel/marshmallow/Marshmallow001'
    trial_seed = None
    grasp_height = None
    def _load_model(self, attempt_num=0):
        # Deterministic one-object setup: don't rebuild the same invalid scene 50 times.
        if attempt_num >= 3:
            raise RuntimeError('Object placement setup failed after 3 attempts')
        return super()._load_model(attempt_num=attempt_num)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.table=self.get_fixture('dining_table_dining_room')
    def _create_objects(self):
        super()._create_objects()
        # Room for rotated/asymmetric bounding boxes, still bounded by the tabletop.
        size = np.asarray(self.objects['test_object'].size)
        extent = float(np.linalg.norm(size[:2]) + .08)
        for cfg in self.object_cfgs:
            if cfg['name'] == 'test_object':
                cfg['placement']['size'] = (extent, extent)

    def _get_obj_cfgs(self):
        path=Path(robocasa.models.assets_root)/'objects'/self.test_asset/'model.xml'
        rng = np.random.default_rng(self.trial_seed)
        xpos = 0. if self.trial_seed is None else float(rng.uniform(-.35,.35))
        yaw = 0. if self.trial_seed is None else float(rng.uniform(-np.pi,np.pi))
        return [dict(name='test_object',obj_groups=str(path),placement=dict(
            fixture=self.table,size=('obj','obj'),pos=(xpos,1),offset=(0,-.08),rotation=yaw))]
    def _check_success(self):return False

class Trial:
    def __init__(self,env,out):
        self.env=env;self.out=out;self.m=env.sim.model._model;self.d=env.sim.data._data
        self.robot=env.robots[0];self.states=[];self.labels=[];self.events=[]
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.qids=[self.m.joint(n).qposadr[0] for n in self.robot.robot_model.arm_joints]
        self.vids=[self.m.joint(n).dofadr[0] for n in self.robot.robot_model.arm_joints]
        self.limits=np.array([self.m.joint(n).range for n in self.robot.robot_model.arm_joints])
        self.site=self.m.site('gripper0_right_grip_site').id
        self.obj=env.obj_body_id['test_object']
        self.obj_geoms={i for i in range(self.m.ngeom) if self.m.geom(i).name and self.m.geom(i).name.startswith('test_object_') and self.m.geom_contype[i]}
        self.pads={side:{self.m.geom(f'gripper0_right_{side}_pad{i}').id for i in (1,2)} for side in ('left','right')}
        mesh=self.m.mesh('gripper0_right_base').id
        a=self.m.mesh_vertadr[mesh];n=self.m.mesh_vertnum[mesh]
        assert .06 < float(np.max(np.ptp(self.m.mesh_vert[a:a+n],axis=0))) < .15, 'Incorrect gripper mesh scale'
        self.plan=mujoco.MjData(self.m);self.grip=-1.;self.target=TUCK.copy();self.regulate=False
    def event(self,stage,**kw):
        entry=dict(stage=stage,time=float(self.d.time),**kw);self.events.append(entry);print(json.dumps(entry),flush=True)
    def contact(self):
        forces=dict(left=0.,right=0.);table_force=0.;penetration=0.
        for i,c in enumerate(self.d.contact):
            pair={int(c.geom1),int(c.geom2)}
            if not pair&self.obj_geoms:continue
            force=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,i,force)
            for side,pads in self.pads.items():
                if pair&pads:forces[side]+=abs(float(force[0]));penetration=max(penetration,float(-c.dist))
            if any((self.m.geom(g).name or '').startswith('dining_table_') for g in pair):table_force+=abs(float(force[0]))
        return forces,table_force,penetration
    def tick(self,label,target=None,steps=1):
        if target is not None:self.target=np.asarray(target)
        for _ in range(steps):
            if self.regulate:
                force,_,_=self.contact()
                if max(force.values())>4.:self.grip=max(-1.,self.grip-.015)
                elif min(force.values())<.5:self.grip=min(1.,self.grip+.01)
            self.env.step(self.robot.create_action_vector(dict(right=self.target,right_gripper=[self.grip],base=[0,0,0],base_mode=1)))
            state=np.empty(mujoco.mj_stateSize(self.m,self.spec));mujoco.mj_getState(self.m,self.d,state,self.spec)
            self.states.append(state);self.labels.append(label)
            if not np.all(np.isfinite(self.d.qpos)):raise RuntimeError('Nonfinite simulation state')
            for c in self.d.contact:
                pair={int(c.geom1),int(c.geom2)}
                names=[self.m.geom(g).name or '' for g in pair]
                robot=[n.startswith(('robot0_','gripper0_','mobilebase0_')) for n in names]
                if c.dist<-.003 and any(robot) and not pair&self.obj_geoms:
                    raise RuntimeError(f'Unexpected robot collision: {names}, depth={-c.dist:.4f}')
    def mesh_points(self):
        points=[]
        for g in self.obj_geoms:
            mesh=self.m.geom_dataid[g];a=self.m.mesh_vertadr[mesh];n=self.m.mesh_vertnum[mesh]
            points.append(self.m.mesh_vert[a:a+n]@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g])
        return np.concatenate(points)
    def bounds(self):
        points=self.mesh_points()
        return points.min(0),points.max(0)
    def ik(self,pos,rot,seed=None):
        self.plan.qpos[:]=self.d.qpos;self.plan.qvel[:]=0
        q=self.d.qpos[self.qids].copy() if seed is None else seed.copy()
        jp=np.zeros((3,self.m.nv));jr=jp.copy()
        for _ in range(220):
            self.plan.qpos[self.qids]=q;mujoco.mj_kinematics(self.m,self.plan);mujoco.mj_comPos(self.m,self.plan)
            ep=pos-self.plan.site_xpos[self.site]
            er=Rotation.from_matrix(rot@self.plan.site_xmat[self.site].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(ep)<.0015 and np.linalg.norm(er)<.02:return q
            mujoco.mj_jacSite(self.m,self.plan,jp,jr,self.site)
            j=np.vstack((jp[:,self.vids],.3*jr[:,self.vids]));e=np.r_[ep,.3*er]
            dq=j.T@np.linalg.solve(j@j.T+np.eye(6)*.0005,e)
            q=np.clip(q+np.clip(dq,-.12,.12),self.limits[:,0]+.015,self.limits[:,1]-.015)
        if seed is None:
            from scipy.optimize import least_squares
            def residual(x):
                self.plan.qpos[self.qids]=x;mujoco.mj_kinematics(self.m,self.plan)
                return np.r_[pos-self.plan.site_xpos[self.site],.3*Rotation.from_matrix(rot@self.plan.site_xmat[self.site].reshape(3,3).T).as_rotvec()]
            rng=np.random.default_rng(7)
            seeds=[np.array([0,-.3,0,-1.8,0,2.,0])]+[rng.uniform(self.limits[:,0]+.05,self.limits[:,1]-.05) for _ in range(7)]
            for guess in seeds:
                result=least_squares(residual,guess,bounds=(self.limits[:,0]+.015,self.limits[:,1]-.015),max_nfev=200)
                e=residual(result.x)
                if np.linalg.norm(e[:3])<.0015 and np.linalg.norm(e[3:])<.006:return result.x
        raise RuntimeError(f'IK failed: {np.linalg.norm(ep):.3f}m {np.linalg.norm(er):.3f}rad')
    def move(self,label,pos,rot,duration=2.):
        start=self.d.site_xpos[self.site].copy();q=self.d.qpos[self.qids].copy()
        # Solve Cartesian waypoints first; validate actual physics while tracking.
        waypoints=[]
        for t in np.linspace(0,1,int(duration*20)):
            q=self.ik(start+(pos-start)*t,rot,q);waypoints.append(q.copy())
        for q in waypoints:
            self.tick(label,q)
            if label=='lower onto table':
                _,support,_=self.contact()
                if support>.025:
                    self.target=self.d.qpos[self.qids].copy()
                    self.event(label,stop_reason='object physically supported by table',table_force_n=support)
                    return
        self.tick(label,steps=12)
        error=float(np.linalg.norm(self.d.site_xpos[self.site]-pos))
        self.event(label,tcp_error_m=error)
        if error>.012:raise RuntimeError(f'{label}: tracking error {error:.3f}m')
    def run(self):
        self.tick('settle upright',steps=40)
        low,high=self.bounds();center=(low+high)/2
        self.event('mesh grasp candidates',object_bounds=[low.tolist(),high.tolist()],object_mass_kg=float(self.m.body_mass[self.obj]))
        # Gripper TCP y is its finger closing axis; approach points downwards.
        options=[]
        rng = np.random.default_rng(self.env.trial_seed)
        object_rotation=self.d.xmat[self.obj].reshape(3,3)
        object_yaw=np.arctan2(object_rotation[1,0],object_rotation[0,0])
        yaw_offset=0. if self.env.trial_seed is None else object_yaw+rng.uniform(-.15,.15)
        points=self.mesh_points()
        for yaw in yaw_offset+np.array([0.,np.pi/2,np.pi,-np.pi/2]):
            rot=Rotation.from_euler('z',yaw).as_matrix()@np.diag([-1.,1.,-1.])
            width=float(np.ptp(points@rot[:,1]))
            if width<.075:options.append((width,rot))
        options.sort(key=lambda item:item[0]);rejections=[]
        rng = np.random.default_rng(self.env.trial_seed)
        if self.env.trial_seed is not None:
            rng.shuffle(options)
        for width,rot in options:
            grasp=center.copy();grasp[2]-=.019
            if self.env.trial_seed is not None:
                grasp[:2] += rng.uniform(-.12,.12,2)*(high-low)[:2]
                grasp[2] += rng.uniform(-.006,.006)
            if self.env.grasp_height is not None:
                # Desired pad contact band; the TCP lies 19 mm below pad center.
                band_z=low[2]+self.env.grasp_height*(high[2]-low[2])
                band=points[np.abs(points[:,2]-band_z)<.012]
                if len(band)<4:continue
                grasp[:2]=(band[:,:2].min(0)+band[:,:2].max(0))/2
                grasp[2]=band_z-.019
                width=float(np.ptp(band@rot[:,1]))
                if width>=.075:continue
            above=grasp+np.array([0,0,.08])
            try:
                q=self.ik(above,rot)
                self.ik(grasp,rot,q)
            except RuntimeError as exc:rejections.append(str(exc));continue
            object_rotation=self.d.xmat[self.obj].reshape(3,3)
            self.grasp_record=dict(asset=self.env.test_asset,
                gripper='Robotiq2f85_v4',scope='single asset, scale and tested pose',
                object_scale=np.asarray(self.env.objects['test_object']._scale).tolist(),
                tcp_position_in_object=(object_rotation.T@(grasp-self.d.xpos[self.obj])).tolist(),
                tcp_rotation_in_object=(object_rotation.T@rot).tolist(),
                collision_mesh_extents_m=(high-low).tolist())
            self.event('selected mesh grasp',grasp_height_fraction=self.env.grasp_height,width_m=width,tcp=grasp.tolist(),candidate_rejections=rejections)
            break
        else:raise RuntimeError(f'No feasible mesh grasp candidate: {rejections}')
        self.execute_grasp(grasp,rot,above)

    def execute_grasp(self,grasp,rot,above):
        # Rotate/fold in free space before the straight approach.
        start=self.d.qpos[self.qids].copy()
        route=getattr(self,'approach_path',None)
        if route is None:
            q=self.ik(above,rot)
            route=[start+(q-start)*(3*t*t-2*t*t*t) for t in np.linspace(0,1,100)]
        for q in route:self.tick('approach above object',q)
        self.tick('approach above object',steps=20)
        self.move('lower to grasp',grasp,rot)
        self.regulate=True;self.tick('close fingers',steps=140)
        force,_,penetration=self.contact();self.event('grasp contact',forces_n=force,penetration_m=penetration)
        if min(force.values())<.15:raise RuntimeError('No bilateral physical finger force')
        # Record the achieved physical grasp, not just the requested IK target.
        object_rotation=self.d.xmat[self.obj].reshape(3,3)
        self.grasp_record.update(
            trial_seed=self.env.trial_seed,
            tcp_position_in_object=(object_rotation.T@(self.d.site_xpos[self.site]-self.d.xpos[self.obj])).tolist(),
            tcp_rotation_in_object=(object_rotation.T@self.d.site_xmat[self.site].reshape(3,3)).tolist(),
            arm_qpos=self.d.qpos[self.qids].tolist(),
            gripper_command=float(self.grip),
            gripper_joint_positions={self.m.joint(j).name:float(self.d.qpos[self.m.joint(j).qposadr[0]])
                for j in range(self.m.njnt) if (self.m.joint(j).name or '').startswith('gripper0_')},
            pad_separation_m=float(np.linalg.norm(
                self.d.geom_xpos[next(iter(self.pads['left']))]-self.d.geom_xpos[next(iter(self.pads['right']))])),
            object_pose_world=dict(position=self.d.xpos[self.obj].tolist(),rotation=object_rotation.tolist()))
        reference_relative = self.d.site_xmat[self.site].reshape(3,3).T@(self.d.xpos[self.obj]-self.d.site_xpos[self.site])
        original=self.d.xpos[self.obj].copy()
        self.move('lift object',grasp+[0,0,.12],rot,3.)
        hold=[];slips=[];gap=0.;max_gap=0.
        for _ in range(40):
            self.tick('hold object');force,_,pen=self.contact();hold.append((min(force.values()),self.d.xpos[self.obj,2]-original[2],pen))
            relative=self.d.site_xmat[self.site].reshape(3,3).T@(self.d.xpos[self.obj]-self.d.site_xpos[self.site])
            slips.append(float(np.linalg.norm(relative-reference_relative)))
            gap = gap+.05 if min(force.values())<.1 else 0.
            max_gap=max(max_gap,gap)
        hold=np.asarray(hold)
        self.event('lift and hold',min_force_n=float(hold[:,0].min()),min_lift_m=float(hold[:,1].min()),max_penetration_m=float(hold[:,2].max()))
        qualification=dict(min_lift_m=float(hold[:,1].min()), min_force_n=float(hold[:,0].min()),
                           max_slip_m=max(slips), max_bilateral_gap_s=max_gap,
                           hold_duration_s=2., max_penetration_m=float(hold[:,2].max()))
        self.event('grasp qualification',**qualification)
        if hold[:,1].min()<.09 or max_gap>=.25 or max(slips)>.02:
            raise RuntimeError('Physical lift/hold failed')
        self.grasp_record['qualification']=qualification
        self.out.joinpath('qualified_grasp.json').write_text(json.dumps(self.grasp_record,indent=2))
        destination=grasp+np.array([.14,0,0])
        self.move('carry above placement',destination+[0,0,.12],rot,3.)
        self.move('lower onto table',destination+[0,0,.004],rot,3.)
        self.regulate=False;self.grip=-1.;self.tick('release object',steps=30)
        self.move('withdraw fingers',destination+[0,0,.10],rot,2.)
        self.tick('settle placement',steps=40)
        force,support,pen=self.contact();final=self.d.xpos[self.obj].copy()
        self.event('placement check',object_xyz=final.tolist(),table_force_n=support,gripper_forces_n=force)
        if support<=0 or np.linalg.norm(final[:2]-(original[:2]+[.14,0]))>.05:raise RuntimeError('Object not supported at destination')
        self.out.joinpath('validated_grasp.json').write_text(json.dumps(self.grasp_record,indent=2))
    def save(self,error):
        self.out.joinpath('scene.xml').write_text(self.env.model.get_xml())
        np.savez_compressed(self.out/'states.npz',states=self.states,labels=self.labels)
        self.out.joinpath('report.json').write_text(json.dumps(dict(asset=self.env.test_asset,success=error is None,error=error,events=self.events,planner='local Jacobian IK, parked base; not cuRobo',physical_grasp=True),indent=2))
        if not self.states:return
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        self.m.vis.global_.offwidth=960;self.m.vis.global_.offheight=720
        cam=mujoco.MjvCamera();cam.lookat[:]=self.d.xpos[self.obj]+[0,0,.10];cam.distance=2.4;cam.azimuth=115;cam.elevation=-30
        opt=mujoco.MjvOption();opt.geomgroup[0]=0
        with mujoco.Renderer(self.m,height=720,width=960) as renderer,imageio.get_writer(self.out/'grasp_test.mp4',fps=20,codec='libx264') as writer:
            for i,state in enumerate(self.states):
                mujoco.mj_setState(self.m,self.d,state,self.spec);mujoco.mj_forward(self.m,self.d)
                renderer.update_scene(self.d,camera=cam,scene_option=opt)
                im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((20,20),self.labels[i],fill='white');writer.append_data(np.asarray(im))
                if i==len(self.states)-1:im.save(self.out/'result.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--trial-seed',type=int);p.add_argument('--grasp-height',type=float,choices=None,help='Contact height fraction of object (0 to 1)');p.add_argument('--asset',default=TidyBotGraspTest.test_asset,help='Native asset path relative to objects/, without model.xml');args=p.parse_args();TidyBotGraspTest.test_asset=args.asset;TidyBotGraspTest.trial_seed=args.trial_seed;TidyBotGraspTest.grasp_height=args.grasp_height;args.output.mkdir(parents=True,exist_ok=True)
    env=robosuite.make('TidyBotGraspTest',robots='TidyBotFranka',controller_configs=controller_config(),layout_ids=[1101],style_ids=[1],seed=7,initialization_noise=None,use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,randomize_cameras=False)
    try:
        env.reset();env.sim.forward();trial=Trial(env,args.output)
        obj=trial.d.xpos[trial.obj].copy()
        rng=np.random.default_rng(args.trial_seed)
        offset=np.array([0.,.34,-obj[2]])
        if args.trial_seed is not None:
            offset[:2]+=rng.uniform([-.03,-.02],[.03,.02])
        set_robot_to_position(env,obj+offset)
        yaw_joint=trial.m.joint('mobilebase0_joint_mobile_yaw').qposadr[0]
        base=trial.m.body('mobilebase0_base').id
        yaw=np.arctan2(trial.d.xmat[base,3],trial.d.xmat[base,0]);trial.d.qpos[yaw_joint]+=(-np.pi/2-yaw)
        env.sim.forward();trial.robot.composite_controller.reset()
        error=None
        try:trial.run()
        except Exception as exc:error=str(exc);traceback.print_exc()
        print(json.dumps(dict(physics_finished=True,error=error)),flush=True);trial.save(error)
    finally:env.close()

if __name__=='__main__':main()
