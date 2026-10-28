#!/usr/bin/env python3
"""Paired Astra evaluation using pinned RoboICL/GPT-Policy-Eval clients."""
import argparse
import json
import os
from pathlib import Path
import time
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework',choices=['roboicl','gpt-policy-eval'],required=True)
    p.add_argument('--memory',choices=['none','all'],required=True)
    p.add_argument('--source',type=Path,help='Single saved evaluation episode')
    p.add_argument('--dataset',type=Path,default=ROOT/'runs/train_test_v1')
    p.add_argument('--split',choices=['test_two_rooms','test_multi_room','both'],default='both')
    p.add_argument('--task',default='set_table',choices=['set_table','clear_and_store','prepare_and_serve_meal','unpack_and_sort_groceries','warm_two_pots','all'])
    p.add_argument('--history-root',type=Path,default=ROOT/'runs/train_test_v1/train')
    p.add_argument('--history-manifest',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--effort',default='high')
    p.add_argument('--max-calls',type=int,default=100)
    p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--request-timeout',type=float,default=240)
    p.add_argument('--limit',type=int,help='Maximum episodes; omitted evaluates all matching episodes')
    p.add_argument('--mock',action='store_true',help='Offline physical wiring test; not a policy evaluation')
    p.add_argument('--preflight',action='store_true',help='Validate inputs without creating simulator/model')
    p.add_argument('--controller-config',type=Path,default=ROOT/'configs/controllers/molmospaces_mobile_franka.json')
    a=p.parse_args()
    if min(a.max_calls,a.seconds,a.request_timeout)<=0 or (a.limit is not None and a.limit<=0):p.error('Limits must be positive')
    from cross_episode_sim.agent_policy.history import catalog,public_catalog,frames
    from cross_episode_sim.agent_policy.interface import validate
    if a.source:sources=[a.source.resolve()]
    else:
        splits=['test_two_rooms','test_multi_room'] if a.split=='both' else [a.split]
        sources=[]
        for split in splits:
            for file in sorted((a.dataset/split).glob('batches/*/*/physics_result.json')):
                source=file.parent
                if not json.loads(file.read_text()).get('physics_success'):continue
                task=json.loads((source/'config.json').read_text())['task']
                if a.task=='all' or a.task==task:sources.append(source.resolve())
    if a.limit:sources=sources[:a.limit]
    if not sources:raise ValueError('No evaluation scenes matched')
    for source in sources:
        for name in ['action_recording/model.mjb','action_recording/initial_state.npz','initial_snapshot.json','config.json','task_manifest.json','instruction.txt']:
            if not (source/name).is_file():raise FileNotFoundError(source/name)
    histories=[]
    if a.memory=='all':
        histories=catalog(a.history_root)
        if a.history_manifest:
            supplied=json.loads(a.history_manifest.read_text())
            if supplied!=histories:raise ValueError('History manifest differs from allowed fine-tuning pool')
        if {str(s) for s in sources}&{h['source'] for h in histories}:raise ValueError('Evaluation/history overlap')
    print(json.dumps({'episodes':len(sources),'history_videos':len(histories),'framework':a.framework,'memory':a.memory,'model':a.model}),flush=True)
    if a.preflight:return
    if a.output:a.output.mkdir(parents=True,exist_ok=False);output=a.output.resolve()
    else:
        (ROOT/'runs').mkdir(exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix=f'astra_{a.framework}_{a.memory}_',dir=ROOT/'runs'))
    print('OUTPUT='+str(output),flush=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(sources=[str(s) for s in sources],method='cross_sim_framework_client_adaptation',current_episode_history=True,
                  controller='native MolmoSpaces joint position; optional native IK',control_hz=50)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'history_manifest.json').write_text(json.dumps(histories,indent=2))
    from tqdm import tqdm
    results=[]
    for index,source in enumerate(tqdm(sources,desc='Astra episodes',unit='episode')):
        result=run_episode(a,source,output/f'{index:04d}_{source.name}',histories)
        results.append(result)
        summary={'episodes':len(results),'successes':sum(x['success'] for x in results),
                 'success_rate':sum(x['success'] for x in results)/len(results),
                 'mock':a.mock,'results':results}
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


def run_episode(a,source,output,histories):
    from cross_episode_sim.agent_policy.environment import Environment
    from cross_episode_sim.agent_policy.backends import Backend
    from cross_episode_sim.agent_policy.interface import validate
    from cross_episode_sim.agent_policy.history import frames,public_catalog
    output.mkdir();env=None;backend=None;failure=None;calls=0;termination='call_limit';start=time.monotonic()
    feedback={'ok':True,'message':'Initial observation'};extra=[];goals={};simtime=0.;success=False
    try:
        env=Environment(source,output,a.controller_config)
        if not a.mock:backend=Backend(a.framework,a.model,a.effort,a.memory,a.request_timeout,output)
        for turn in range(a.max_calls):
            state,images=env.observe(turn)
            sent_images=images+extra
            obs={'task':env.prompt,'robot':state,'last_tool_result':feedback}
            if turn==0:obs['history_catalog']=public_catalog(histories)
            if a.mock:
                decision=({'name':'wait','arguments':{'seconds':.2}} if turn==0 else {'name':'finish','arguments':{}})
            else:decision=backend.decide(obs,sent_images)
            calls+=1;extra=[]
            record={'turn':turn,'observation':obs,'images':[str(x) for x in sent_images],
                    'decision':decision,'usage':getattr(backend,'last_usage',None)}
            try:
                validate(decision,a.memory)
                name,args=decision['name'],decision['arguments']
                if name=='finish':termination='agent_finished';feedback={'ok':True}
                elif name=='history_frame':
                    extra=frames(histories,args['video_id'],args['seconds'],output)
                    feedback={'ok':True,'history_video_id':args['video_id'],'timestamps':args['seconds'],'historical_images':[x.name for x in extra]}
                else:
                    remaining=a.seconds-(float(env.data.time)-env.start)
                    if args.get('seconds',0)>remaining:
                        termination='simulation_time_limit';break
                    before=float(env.data.time)
                    try:feedback=env.act(name,args)
                    except (ValueError,RuntimeError) as exc:
                        feedback={'ok':False,'error':str(exc),'executed_seconds':float(env.data.time)-before}
                        if 'Physics diverged' in str(exc):raise
                record['feedback']=feedback
            except (ValueError,KeyError,TypeError) as exc:
                feedback={'ok':False,'error':str(exc),'executed_seconds':0};record['feedback']=feedback
            with (output/'exchanges.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(f'{source.name}: [{calls}/{a.max_calls}] {decision.get("name")} | sim={float(env.data.time)-env.start:.2f}s | ok={feedback["ok"]}',flush=True)
            goals=env.goals();success=env.stable>=2
            (output/'progress.json').write_text(json.dumps({'calls':calls,'goal_evidence':goals,'simulated_seconds':float(env.data.time)-env.start},indent=2))
            if success:termination='success';break
            if termination=='agent_finished':break
            if float(env.data.time)-env.start>=a.seconds:termination='simulation_time_limit';break
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}';termination='error'
    finally:
        if backend:backend.close()
        if env:
            simtime=float(env.data.time)-env.start;goals=env.goals();env.close()
    result={'source':str(source),'success':success and not failure,'mock':a.mock,'failure':failure,
            'termination':termination,'goal_evidence':goals,'calls':calls,'simulated_seconds':simtime,
            'wall_seconds':time.monotonic()-start,'video':str(output/'rollout.mp4')}
    (output/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
