"""LeRobot v3 arm-only skill episodes from successful fine-tuning recordings.

This is offline data conversion, not policy execution or regenerated physics.
Scene truth is retained only in supplementary provenance, never model features.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def sample_segment(times,start,end,fps):
    """Sample pre-action records on a fixed clock; never cross a skill boundary."""
    desired=start+np.arange(int(np.floor((end-start)*fps)))/fps
    indices=np.searchsorted(times,desired,side='left')
    good=(indices<len(times))
    indices=indices[good];desired=desired[good]
    good=times[indices]<end-1e-8
    return indices[good],desired[good]


def stationary(base,command,xy_limit=.02,yaw_limit=.03):
    def span(a):
        return float(np.max(np.linalg.norm(a[:,:2]-a[0,:2],axis=1))),float(np.ptp(np.unwrap(a[:,2])))
    xy,yaw=span(base);cxy,cyaw=span(command)
    return xy<=xy_limit and yaw<=yaw_limit and cxy<=.002 and cyaw<=.005,dict(measured_xy_span_m=xy,measured_yaw_span_rad=yaw,command_xy_span_m=cxy,command_yaw_span_rad=cyaw)


def plan(source,output,seed=7):
    meta=json.loads((source/'meta/cross_sim.json').read_text())
    if meta['split']!='fine_tuning' or meta['task']!='set_table':raise ValueError('First qualification is set-table fine-tuning only')
    runs=[Path(e['source']) for e in meta['episodes']]
    # Randomized whole-source split; all skill segments from a scene stay together.
    ordered=sorted(runs,key=lambda p:hashlib.sha256(f'{seed}/{p.name}'.encode()).hexdigest())
    valid=set(ordered[:max(1,round(len(runs)*.12))])
    result=dict(version=1,source_dataset=str(source),seed=seed,source_task='set_table',splits={s:[str(p) for p in runs if (p in valid)==(s=='validation')] for s in ('fine_tuning','validation')})
    output.mkdir(parents=True,exist_ok=False);(output/'split_plan.json').write_text(json.dumps(result,indent=2));return result


def export(root,split,repo_id,fps=15,size=224):
    import mujoco
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/openpi/packages/openpi-client/src'))
    from openpi_client.image_tools import resize_with_pad
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm
    split_plan=json.loads((root/'split_plan.json').read_text());dest=root/split
    names=[f'joint_{i}' for i in range(1,8)]+['gripper_closed_fraction']
    features={k:dict(dtype='float32',shape=(8,),names=names) for k in ('observation.state','action')}
    for c in ('navigation','wrist'):features[f'observation.images.{c}_cam']=dict(dtype='video',shape=(size,size,3),names=['height','width','channels'])
    ds=LeRobotDataset.create(repo_id=repo_id,root=dest,fps=fps,robot_type='mobile_franka_stationary_arm',features=features,vcodec='h264',image_writer_threads=4)
    audit=dict(status='incomplete',split=split,fps=fps,action_space='absolute_joint_position',action_dim=8,
        state_gripper='measured closure from calibrated robot finger distance',action_gripper='recorded control / 255',
        sampling='fixed clock, first raw pre-action state at or after requested time; max timing error audited',
        rendering_schema='onboard_visual_geometry_v1',physics_replay_validated=False,episodes=[],rejected=[])
    # Match OnboardSensorInterface: hide render-only collision proxies/sites.
    option=mujoco.MjvOption();option.geomgroup[0]=0;option.sitegroup[:]=0
    for run_text in tqdm(split_plan['splits'][split],desc=f'{split}: source episodes'):
        run=Path(run_text)
        if not json.loads((run/'report.json').read_text())['success']:raise ValueError('Unsuccessful source')
        cfg=json.loads((run/'config.json').read_text())
        if cfg.get('dataset_split') not in ('train','fine_tuning'):raise ValueError('Test episode prohibited')
        model=mujoco.MjModel.from_binary_path(str(run/'action_recording/model.mjb'));data=mujoco.MjData(model)
        with np.load(run/'action_recording/initial_state.npz') as z:mujoco.mj_setState(model,data,z['state'],int(z['state_type']))
        with np.load(run/'action_recording/actions.npz') as z:raw={k:z[k] for k in ('time','ctrl','qpos')}
        qids=[int(model.joint(f'robot_0/fr3_joint{i}').qposadr[0]) for i in range(1,8)]
        acts=[model.actuator(f'robot_0/fr3_joint{i}').id for i in range(1,8)]
        bq=[int(model.joint('robot_0/base_'+n).qposadr[0]) for n in ('x','y','theta')]
        ba=[model.actuator('robot_0/base_'+n+'_act').id for n in ('x','y','theta')]
        ga=model.actuator('robot_0/gripper/fingers_actuator').id
        pads=[model.geom('robot_0/gripper/'+n+'_pad2').id for n in ('left','right')]
        bindings={r['role']:r for r in json.loads((run/'task_manifest.json').read_text())['bindings']}
        held=None
        model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),640)
        model.vis.global_.offheight=max(int(model.vis.global_.offheight),480)
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            for call_index,call in enumerate(json.loads((run/'skill_calls.json').read_text())):
                skill=call['skill'];args=call['arguments']
                if skill=='pick':held=bindings.get(args.get('target'))
                if skill not in ('pick','place') or not call['success']:continue
                if held is None:raise ValueError('Missing held-object semantic category')
                category=held['category'];asset=held['asset']
                if category not in ('bowl','cup'):raise ValueError('Unqualified category')
                prompt=f'Pick up a {category} from the counter and hold it.' if skill=='pick' else f'Place the held {category} on the dining table and release it.'
                start,end=call['start_time'],call['end_time'];ids,desired=sample_segment(raw['time'],start,end,fps)
                region=np.flatnonzero((raw['time']>=start)&(raw['time']<end))
                valid,metrics=stationary(raw['qpos'][region][:,bq],raw['ctrl'][region][:,ba])
                entry=dict(source=str(run),source_episode=run.name,call_index=call_index,skill=skill,category=category,asset=asset,prompt=prompt,start_time=start,end_time=end,**metrics)
                if not valid or len(ids)<2:
                    audit['rejected'].append(dict(**entry,reason='base_motion' if not valid else 'too_short'));continue
                ep=len(audit['episodes']);side=dest/'supplementary'/f'episode_{ep:06d}';side.mkdir(parents=True)
                # Keep layout/object lists separately; never place hidden metadata in features.
                for name in ('house_overview.png','asset_selection.json','base_layout.json'):
                    if (run/name).exists():shutil.copy2(run/name,side/name)
                for i in tqdm(ids,desc=f'{run.name} call {call_index}: {skill} {category}',leave=False):
                    data.qpos[:]=raw['qpos'][i];data.qvel[:]=0;data.ctrl[:]=raw['ctrl'][i];data.time=float(raw['time'][i]);mujoco.mj_forward(model,data)
                    distance=max(0.,mujoco.mj_geomDistance(model,data,*pads,.1,None));closure=np.clip(1-distance/.087,0,1)
                    state=np.r_[data.qpos[qids],closure].astype('float32');action=np.r_[raw['ctrl'][i,acts],np.clip(raw['ctrl'][i,ga]/255.,0,1)].astype('float32')
                    frame={'observation.state':state,'action':action,'task':prompt}
                    for c in ('navigation','wrist'):
                        renderer.update_scene(data,camera=f'robot_0/{c}_cam',scene_option=option);frame[f'observation.images.{c}_cam']=resize_with_pad(renderer.render().copy(),size,size)
                    ds.add_frame(frame)
                ds.save_episode()
                entry.update(episode_index=ep,frames=len(ids),max_timestamp_error_s=float(np.max(raw['time'][ids]-desired)),source_sample_indices=ids.tolist())
                (side/'provenance.json').write_text(json.dumps(entry,indent=2));audit['episodes'].append(entry)
                (dest/'meta/arm_skills.json').write_text(json.dumps(audit,indent=2))
                if skill=='place':held=None
    ds.finalize()
    check=LeRobotDataset(repo_id,root=dest,video_backend='pyav',tolerance_s=1e-3);offset=0
    for ep in tqdm(audit['episodes'],desc='Validate every segment boundary'):
        for i in (offset,offset+ep['frames']-1):
            row=check[i]
            for k in ('observation.state','action'):assert row[k].shape==(8,) and np.isfinite(row[k].numpy()).all()
            for c in ('navigation','wrist'):assert row[f'observation.images.{c}_cam'].shape==(3,size,size)
            assert row['task']==ep['prompt']
        offset+=ep['frames']
    assert offset==len(check)
    audit.update(status='complete',total_frames=offset,all_segment_endpoints_decoded=True)
    (dest/'meta/arm_skills.json').write_text(json.dumps(audit,indent=2));print(json.dumps(dict(split=split,episodes=len(audit['episodes']),frames=offset,rejected=len(audit['rejected']))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['fine_tuning','validation']);p.add_argument('--repo-id',default='plnguyen2908/cross-sim-data');p.add_argument('--fps',type=int,default=15)
    a=p.parse_args()
    if a.split:export(a.output,a.split,a.repo_id,a.fps)
    else:plan(a.source,a.output)
