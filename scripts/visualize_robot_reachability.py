"""Measure cuRobo IK workspace using robot-only calibration and plot it.

The saved model/state initialize a private diagnostic harness; only calibrated
robot geometry and joint readings enter IK. No scene objects enter the solver.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,mujoco,h5py
from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.reachability import ReachabilityEvaluator,interior_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=float,default=.08)
    p.add_argument('--reuse-volume',action='store_true',help='Reuse this diagnostic output volume; recompute missing slices only')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.reuse_volume);start=time.monotonic()
    model=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'));data=mujoco.MjData(model)
    with h5py.File(a.recording/'states.h5') as f:mujoco.mj_setState(model,data,f['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    kin=robot_calibration(model,data,a.output/'robot')
    signature_data={k:v for k,v in kin.items() if k not in ('urdf_path','asset_root_path')}
    signature_data['urdf']=Path(kin['urdf_path']).read_text()
    signature=hashlib.sha256(json.dumps(signature_data,sort_keys=True).encode()).hexdigest()
    signature_path=a.output/'robot_signature.txt'
    if a.reuse_volume and (not signature_path.exists() or signature_path.read_text().strip()!=signature):
        raise ValueError('Cannot reuse a volume without matching robot calibration; use a fresh output')
    signature_path.write_text(signature+'\n')
    evaluator=ReachabilityEvaluator(kin)
    target=np.load(a.input)['target'];res=a.resolution
    axes=[np.arange(.1,1.02,res),np.arange(-.64,.65,res),np.arange(.4,1.51,res)]
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);points=grid.reshape(-1,3)
    slice_axes=[np.arange(.1,1.001,.04),np.arange(-.64,.641,.04)]
    xy=np.stack(np.meshgrid(*slice_axes,indexing='ij'),axis=-1)
    sliced=np.column_stack((xy.reshape(-1,2),np.full(xy.size//2,target[2])))
    orientations={'top_down':np.diag([1.,-1.,-1.]),'side':np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])}
    reports={};volumes={};slices={}
    for name,R in orientations.items():
        print(json.dumps(dict(stage='volume',orientation=name,targets=len(points))),flush=True)
        volume_path=a.output/(name+'.npz')
        if a.reuse_volume and volume_path.exists():
            saved=np.load(volume_path)
            if not np.array_equal(saved['points'],points) or not np.array_equal(saved['orientation'],R):
                raise ValueError('Cached volume grid or orientation differs')
            volume={k:saved[k] for k in ('grasp','approach','lift','all_endpoints','interior_distance_m')}
        else:
            volume,joints=evaluator.evaluate(points,R);volume={k:v.reshape(grid.shape[:-1]) for k,v in volume.items()}
            volume['interior_distance_m']=interior_distance(volume['all_endpoints'],res)
            np.savez_compressed(volume_path,points=points,orientation=R,**volume,**{k+'_joints':v for k,v in joints.items()})
        print(json.dumps(dict(stage='height_slice',orientation=name,targets=len(sliced))),flush=True)
        slab,_=evaluator.evaluate(sliced,R);slab={k:v.reshape(xy.shape[:-1]) for k,v in slab.items()}
        np.savez_compressed(a.output/(name+'_slice.npz'),points=sliced,**slab)
        at_target,_=evaluator.evaluate(target[None],R)
        volumes[name]=volume;slices[name]=slab
        reports[name]=dict(target={k:bool(v[0]) for k,v in at_target.items()},grasp_samples=int(volume['grasp'].sum()),all_endpoint_samples=int(volume['all_endpoints'].sum()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Rectangle,Patch
    colors=['#e6e9ee','#f3bb53','#62bd9b','#157554'];labels=['No IK solution found','Grasp endpoint only','Approach + grasp + lift endpoints','Interior (>= 8 cm in this height slice)']
    fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,(name,slab) in zip(axs, slices.items()):
        level=slab['grasp'].astype(int);level[slab['all_endpoints']]=2
        # Interior on this exact-height slice, not a guarantee in full 3D.
        dist=interior_distance(slab['all_endpoints'],.04);level[(level==2)&(dist>=.08)]=3
        ax.pcolormesh(slice_axes[0],slice_axes[1],level.T,cmap=ListedColormap(colors),vmin=0,vmax=3,shading='nearest')
        ax.scatter(*target[:2],marker='*',s=210,c='#cc2450',edgecolor='white',zorder=5,label='Observed cup')
        ax.annotate('Observed cup',target[:2],xytext=(target[0]+.12,target[1]+.14),arrowprops={'arrowstyle':'->','color':'#cc2450'})
        ax.scatter(0,0,c='black',s=35);ax.annotate('Base origin',(0,0),xytext=(-.05,-.22),arrowprops={'arrowstyle':'->'})
        ax.set(xlim=(-.08,1.04),ylim=(-.7,.7),xlabel='Forward from base (m)',ylabel='Left from base (m)',title=name.replace('_',' ').title(),aspect='equal')
    fig.suptitle(f'cuRobo reachability at cup height: {target[2]:.2f} m\nRobot joint limits + self-collision; furniture/path clearance checked separately',fontsize=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],loc='outside lower center',ncols=2,fontsize=9)
    fig.savefig(a.output/'reachability_top_view.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(13,6),layout='constrained')
    from cross_episode_sim.navigation.shaped_docking import RobotGeometry
    geometry=RobotGeometry(model);q=np.array([data.joint(n).qpos[0] for n in geometry.joint_names]);fk=geometry.fk(q)
    arm=np.array([fk[model.body(f'robot_0/fr3_link{i}').id][:3,3] for i in range(1,8)]+[geometry.tcp_pose(q)[:3,3]])
    for i,(name,v) in enumerate(volumes.items()):
        ax=fig.add_subplot(1,2,i+1,projection='3d');keep=v['all_endpoints'].ravel();xyz=points[keep];interior=v['interior_distance_m'].ravel()[keep]
        ax.scatter(*xyz.T,c=interior,cmap='YlGn',vmin=0,vmax=.2,s=6,alpha=.55)
        ax.scatter(*target,c='#cc2450',marker='*',s=150,depthshade=False)
        ax.plot(*arm.T,color='#343b47',linewidth=4,marker='o',markersize=3,label='Arm at recorded start')
        ax.scatter(0,0,0,c='black',s=35);ax.quiver(0,0,0,.35,0,0,color='black')
        ax.set(xlabel='Forward (m)',ylabel='Left (m)',zlabel='Height (m)',title=name.replace('_',' ').title(),xlim=(-.1,1.05),ylim=(-.7,.7),zlim=(0,1.6));ax.view_init(elev=22,azim=-58)
    fig.suptitle('Forward workspace for approach, grasp and lift endpoints\nDark line = robot arm; green = IK samples; star = observed cup',fontsize=14)
    fig.savefig(a.output/'reachability_3d.png',dpi=180);plt.close(fig)
    report=dict(wall_seconds=time.monotonic()-start,reused_volume=a.reuse_volume,robot_signature=signature,target_base=target.tolist(),resolution_m=res,volume_samples=len(points),orientations=reports,
        provenance='Robot-only calibration; cup marker from saved onboard perception',limitations='Two fixed grasp orientations; finite-seed IK failures are not proof of impossibility. No furniture, payload or continuous path checks. Interior estimates are grid-based, not certified clearance. Not installed in docking policy.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
