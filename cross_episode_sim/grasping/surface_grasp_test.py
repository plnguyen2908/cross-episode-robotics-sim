"""Bounded multi-region, multi-direction grasp search with isolated physical trials."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
import argparse
from collections import Counter
import json
from pathlib import Path
import traceback
import numpy as np
import mujoco
from cross_episode_sim.grasping.grasp_test import Trial,TidyBotGraspTest,robosuite,controller_config,TUCK,set_robot_to_position
from cross_episode_sim.manipulation.surface_grasps import sample_collision_surface,candidates


class SurfaceTrial(Trial):
    def collision(self,q,allow_pads=False):
        self.plan.qpos[:]=self.d.qpos
        self.plan.qpos[self.qids]=q
        mujoco.mj_fwdPosition(self.m,self.plan)
        pads=set.union(*self.pads.values())
        for c in self.plan.contact:
            pair={int(c.geom1),int(c.geom2)}
            names=[self.m.geom(g).name or '' for g in pair]
            if not any(n.startswith(('robot0_','gripper0_','mobilebase0_')) for n in names):continue
            if c.dist>=-.001:continue
            if allow_pads and pair&self.obj_geoms and pair&pads:continue
            return dict(geoms=names,depth=float(-c.dist))
        return None

    def joint_route(self,start,goal):
        def clear(a,b):
            for t in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.035)+1)):
                if self.collision(a+(b-a)*t):return False
            return True
        if clear(start,goal):return [start,goal]
        # Bidirectional RRT: a bad straight interpolation does not reject a valid grasp.
        rng=np.random.default_rng(19)
        trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
        def extend(tree,target):
            nodes,parents=tree
            nearest=int(np.argmin([np.linalg.norm(q-target) for q in nodes]))
            q=nodes[nearest]
            new=q+(target-q)*min(1.,.22/max(np.linalg.norm(target-q),1e-9))
            if not clear(q,new):return None
            nodes.append(new);parents.append(nearest)
            return len(nodes)-1
        def branch(tree,index):
            nodes,parents=tree;path=[]
            while index>=0:path.append(nodes[index]);index=parents[index]
            return path[::-1]
        for iteration in range(240):
            first=iteration%2;second=1-first
            sample=trees[second][0][-1] if iteration%5==0 else rng.uniform(self.limits[:,0]+.03,self.limits[:,1]-.03)
            ia=extend(trees[first],sample)
            if ia is None:continue
            target=trees[first][0][ia]
            for _ in range(30):
                ib=extend(trees[second],target)
                if ib is None:break
                if np.linalg.norm(trees[second][0][ib]-target)<1e-6:
                    a=branch(trees[first],ia);b=branch(trees[second],ib)
                    route=a+b[-2::-1]
                    if first:route=route[::-1]
                    # Greedy shortcut still checks every interpolated joint segment.
                    simplified=[route[0]];i=0
                    while i<len(route)-1:
                        j=len(route)-1
                        while j>i+1 and not clear(route[i],route[j]):j-=1
                        simplified.append(route[j]);i=j
                    return simplified
        raise RuntimeError('No collision-free joint approach within RRT budget')

    def preflight(self,candidate):
        rot=np.array(candidate['rotation']);grasp=np.array(candidate['tcp'])
        above=grasp-rot[:,2]*.08
        q=self.ik(above,rot)
        if self.collision(q):
            # Try alternate redundant-arm IK solutions before rejecting the stance.
            rng=np.random.default_rng(11)
            for _ in range(5):
                try:q=self.ik(above,rot,rng.uniform(self.limits[:,0]+.03,self.limits[:,1]-.03))
                except RuntimeError:continue
                if not self.collision(q):break
            else:raise RuntimeError('All pre-grasp IK solutions collide')
        route=self.joint_route(self.d.qpos[self.qids].copy(),q.copy())
        self.approach_path=[]
        for a,b in zip(route,route[1:]):
            n=max(12,int(np.max(np.abs(b-a))/.02)+1)
            self.approach_path.extend(a+(b-a)*(3*t*t-2*t*t*t) for t in np.linspace(0,1,n))
        for t in np.linspace(0,1,15):
            q=self.ik(above+(grasp-above)*t,rot,q)
            hit=self.collision(q,allow_pads=t>.9)
            if hit:raise RuntimeError(f'Contact approach collision: {hit}')
        for t in np.linspace(0,1,8):
            q=self.ik(grasp+[0,0,.12*t],rot,q)
            hit=self.collision(q,allow_pads=True)
            if hit:raise RuntimeError(f'Lift collision: {hit}')
        return grasp,rot,above

    def execute(self,candidate,prepared):
        grasp,rot,above=prepared
        object_rotation=self.d.xmat[self.obj].reshape(3,3)
        self.grasp_record=dict(asset=self.env.test_asset,gripper='Robotiq2f85_v4',
            scope='sampled opposing surface contacts, physically qualified at tested pose/scale',
            object_scale=np.asarray(self.env.objects['test_object']._scale).tolist(),
            candidate=candidate,
            base_position_world=self.d.xpos[self.m.body('mobilebase0_base').id].tolist(),
            base_rotation_world=self.d.xmat[self.m.body('mobilebase0_base').id].reshape(3,3).tolist(),
            collision_mesh_extents_m=(self.bounds()[1]-self.bounds()[0]).tolist(),
            contacts_in_object=[(object_rotation.T@(np.array(p)-self.d.xpos[self.obj])).tolist() for p in candidate['contacts']])
        self.event('selected surface grasp',**candidate)
        self.execute_grasp(grasp,rot,above)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asset',default='lightwheel/salt_and_pepper_shaker/SaltShaker001')
    p.add_argument('--molmo',help='Imported MolmoSpaces asset ID')
    p.add_argument('--stop-after-success',action='store_true')
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--physics-trials',type=int,default=9)
    p.add_argument('--candidate-offset',type=int,default=0)
    p.add_argument('--planning-budget',type=int,default=60)
    args=p.parse_args();out=args.output
    if args.physics_trials<1 or args.planning_budget<1:
        p.error('Trial and planning budgets must be positive')
    if out.exists() and any(out.glob('trial_*/report.json')):
        p.error('Output already contains trials; use a fresh directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'grasp_library.json').write_text('[]')
    if args.molmo:
        from cross_episode_sim.manipulation.molmo_objects import register,OUTPUT
        register()
        args.asset=str((OUTPUT/args.molmo/'model.xml').resolve().parent)
    TidyBotGraspTest.test_asset=args.asset;TidyBotGraspTest.trial_seed=args.seed
    env=robosuite.make('TidyBotGraspTest',robots='TidyBotFranka',controller_configs=controller_config(),
        layout_ids=[1101],style_ids=[1],seed=7,initialization_noise=None,use_camera_obs=False,
        has_renderer=False,has_offscreen_renderer=False,randomize_cameras=False)
    records=[];rejections=[];annotations=[]
    try:
        env.reset();env.sim.forward()
        trial=SurfaceTrial(env,out)
        obj=trial.d.xpos[trial.obj].copy()
        set_robot_to_position(env,obj+[0,.34,-obj[2]])
        base=trial.m.body('mobilebase0_base').id
        yaw_joint=trial.m.joint('mobilebase0_joint_mobile_yaw').qposadr[0]
        yaw=np.arctan2(trial.d.xmat[base,3],trial.d.xmat[base,0])
        trial.d.qpos[yaw_joint]+=(-np.pi/2-yaw)
        env.sim.forward();trial.robot.composite_controller.reset()
        trial.tick('settle upright',steps=40)
        state=np.empty(mujoco.mj_stateSize(trial.m,trial.spec));mujoco.mj_getState(trial.m,trial.d,state,trial.spec)
        points,normals=sample_collision_surface(trial.m,trial.d,trial.obj_geoms,seed=args.seed)
        options=candidates(points,normals,seed=args.seed)
        if args.molmo:
            from cross_episode_sim.manipulation.molmo_objects import annotation_candidates
            imported=annotation_candidates(trial.m,trial.d,trial.obj,args.molmo,points)
            # Interleave source annotations with new surface hypotheses.
            mixed=[]
            for i in range(max(len(imported),len(options))):
                if i<len(imported):mixed.append(imported[i])
                if i<len(options):mixed.append(options[i])
            options=mixed
        if options:
            offset=args.candidate_offset%len(options)
            options=options[offset:]+options[:offset]
        (out/'candidates.json').write_text(json.dumps(options,indent=2))
        print(json.dumps(dict(candidates=len(options),families=Counter(c['family'] for c in options))),flush=True)
        # Base positions are alternative episode initializations, not teleporting during a grasp.
        base_offsets=[(0.,0.),(-.08,.04),(.08,.04)]
        physical_families=Counter()
        planned=0
        for candidate in options:
            if len(records)>=args.physics_trials or planned>=args.planning_budget:break
            if physical_families[candidate['family']]>=max(1,args.physics_trials//3):continue
            prepared=None
            for dx,dy in base_offsets:
                if planned>=args.planning_budget:break
                planned+=1
                mujoco.mj_setState(trial.m,trial.d,state,trial.spec);mujoco.mj_forward(trial.m,trial.d)
                set_robot_to_position(env,obj+[dx,.34+dy,-obj[2]])
                # Keep forward-facing yaw after setting translation.
                current=np.arctan2(trial.d.xmat[base,3],trial.d.xmat[base,0])
                trial.d.qpos[yaw_joint]+=(-np.pi/2-current)
                env.sim.forward();trial.robot.composite_controller.reset()
                trial=SurfaceTrial(env,out/f"trial_{len(records):03d}")
                trial.out.mkdir(exist_ok=True)
                try:
                    prepared=trial.preflight(candidate)
                    break
                except RuntimeError as exc:
                    rejections.append(dict(candidate=candidate['id'],family=candidate['family'],base_offset=[dx,dy],reason=str(exc)))
            if prepared is None:continue
            physical_families[candidate['family']]+=1
            error=None
            try:trial.execute(candidate,prepared)
            except Exception as exc:error=str(exc);traceback.print_exc()
            print(json.dumps(dict(physics_finished=True,candidate=candidate['id'],family=candidate['family'],error=error)),flush=True)
            try:
                trial.save(error)
            except Exception as render_error:
                # Physics report/states are already saved by Trial.save before rendering.
                traceback.print_exc()
                (trial.out/'render_error.txt').write_text(str(render_error))
            annotation=trial.out/'qualified_grasp.json'
            if annotation.exists():
                annotations.append(json.loads(annotation.read_text()))
            records.append(dict(candidate=candidate['id'],family=candidate['family'],height=candidate['height_fraction'],
                base_offset=[dx,dy],success=error is None,error=error,qualified=annotation.exists(),video=str(trial.out/'grasp_test.mp4')))
            (out/'grasp_library.json').write_text(json.dumps(annotations,indent=2))
            (out/'summary.json').write_text(json.dumps(dict(status='running',trials=records,rejections=rejections),indent=2))
            if args.stop_after_success and error is None:break
        (out/'summary.json').write_text(json.dumps(dict(status='complete',planned=planned,
            candidates=len(options),trials=records,rejections=rejections),indent=2))
    finally:env.close()

if __name__=='__main__':main()
