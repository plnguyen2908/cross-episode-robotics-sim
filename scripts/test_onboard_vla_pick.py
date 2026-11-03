"""Stationary-base VLA smoke test from a private docking checkpoint.

Only RGB, joint readings and measured robot gripper closure enter the policy.
Private grasp diagnostics do not terminate the rollout or feed back to inference.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import imageio.v2 as imageio
from tqdm import trange
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.control.onboard_vla import OpenPIArmPolicy
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.robot.observations import depth_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--gpu',default='0')
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--prompt',default='Pick up a bowl from the counter and hold it in the air.')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_binary_path(str(a.dock/'private_checkpoint/model.mjb'));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(a.dock/'private_checkpoint/state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(model,data)
    controller=MolmoSpacesController(model,data);base=controller.robot_view.get_move_group('base').joint_pos.copy()
    grip=controller.robot_view.get_move_group('gripper')
    source=Path(json.loads((a.dock/'configuration.json').read_text())['source'])
    manifest=json.loads((source/'task_manifest.json').read_text())
    bowls={r['role']:model.body(r['body']).id for r in manifest['bindings'] if r['category']=='bowl'}
    def descendants(root):
        found={root}
        for i in range(root+1,model.nbody):
            if int(model.body_parentid[i]) in found:found.add(i)
        return found
    bowl_bodies={k:descendants(v) for k,v in bowls.items()}
    fingers={i for i in range(model.nbody) if (model.body(i).name or '').startswith('robot_0/gripper/')}
    z0={k:float(data.xpos[v,2]) for k,v in bowls.items()};initial_base=base.copy()
    records=[];actions=[];measured=[];start=float(data.time);wall=time.monotonic();latencies=[];max_base_drift=0.
    config=dict(checkpoint=str(a.checkpoint),config='pi05_droid_jointpos_polaris',dock=str(a.dock),prompt=a.prompt,
        seconds=a.seconds,policy_hz=15,controller_hz=50,action_space='absolute_joint_position',base='held by separate native controller',
        input_views=['navigation RGB','wrist RGB'],depth_input=False,scope='stationary-base pick smoke test; not a composite evaluation',
        checkpoint_kind='project_fine_tuned' if (a.checkpoint/'assets/set_table_arm/norm_stats.json').exists() else 'upstream_zero_shot')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    with OpenPIArmPolicy(a.checkpoint,a.output/'policy',a.gpu) as policy,OnboardSensorInterface(model,data) as sensors,imageio.get_writer(str(a.output/'rollout.mp4'),fps=10,codec='libx264',pixelformat='yuv420p',output_params=['-movflags','+faststart']) as video,open(a.output/'diagnostics.jsonl','w',buffering=1) as log:
        chunk=None
        for tick in trange(round(a.seconds*50),desc='VLA bowl pick'):
            if chunk is None or tick>=chunk_end:
                packet=sensors.capture();lo,hi=grip.inter_finger_dist_range
                closure=float(np.clip(1-(grip.inter_finger_dist-lo)/(hi-lo),0,1))
                chunk,latency=policy.predict(packet,a.prompt,closure);latencies.append(latency)
                chunk_start=tick;chunk_end=tick+round(len(chunk)*50/15)
                print(f'chunk {len(latencies)} | sim {tick/50:.1f}s | inference {latency:.2f}s',flush=True)
            index=min(len(chunk)-1,int((tick-chunk_start)*15/50));action=chunk[index]
            # Upstream DROID gripper convention: 0=open, 1=closed.
            controller.step(dict(base=base,arm=action[:7],gripper=np.array([255. if action[7]>.5 else 0.])))
            for _ in range(round(.02/model.opt.timestep)):mujoco.mj_step(model,data)
            actions.append(action.copy())
            measured.append(controller.robot_view.get_move_group('arm').joint_pos.copy())
            # Private diagnostics: no observation or policy feedback path.
            touching=set()
            for contact in data.contact:
                b1,b2=model.geom_bodyid[[contact.geom1,contact.geom2]]
                for key,ids in bowl_bodies.items():
                    if (b1 in ids and b2 in fingers) or (b2 in ids and b1 in fingers):touching.add(key)
            lifted={k:float(data.xpos[v,2]-z0[k]) for k,v in bowls.items()}
            row=dict(time=float(data.time-start),lift_m=lifted,gripper_contact=sorted(touching),held_lift=any(lifted[k]>.06 and k in touching for k in bowls))
            records.append(row);log.write(json.dumps(row)+'\n')
            max_base_drift=max(max_base_drift,float(np.linalg.norm(controller.robot_view.get_move_group('base').joint_pos[:2]-initial_base[:2])))
            if tick%5==0:
                packet=sensors.capture();video.append_data(np.concatenate([np.concatenate([c.rgb,depth_rgb(c.depth_m)],axis=1) for c in packet.cameras.values()],axis=0))
    np.save(a.output/'executed_actions.npy',np.asarray(actions))
    np.save(a.output/'measured_arm.npy',np.asarray(measured))
    # Require the SAME bowl held above its initial height throughout the last second.
    held=[k for k in bowls if len(records)>=50 and all(r['lift_m'][k]>.06 and k in r['gripper_contact'] for r in records[-50:])]
    result=dict(success=bool(held),held_bowls=held,max_lift_m={k:max(r['lift_m'][k] for r in records) for k in bowls},
        simulated_seconds=float(data.time-start),wall_seconds=time.monotonic()-wall,calls=len(latencies),inference_seconds=latencies,
        max_base_drift_m=max_base_drift,video=str(a.output/'rollout.mp4'),scope=config['scope'])
    (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
