#!/usr/bin/env python3
"""Paired agent-as-policy evaluation: pinned RoboICL/GPT-Policy-Eval Codex clients, or the OpenAI API directly."""
import argparse
import json
import os
from pathlib import Path
import time
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework',choices=['roboicl','gpt-policy-eval','openai-api'],required=True,
        help='openai-api calls the Responses API with OPENAI_API_KEY; the others go through Codex')
    p.add_argument('--memory',choices=['none','all'],default='none',
        help='Default: none (no historical videos or history-frame tool); all explicitly enables history')
    p.add_argument('--source',type=Path,help='Single saved evaluation episode')
    p.add_argument('--episode-manifest',type=Path,help='JSON with an episodes list of source paths and optional case labels')
    p.add_argument('--dataset',type=Path,default=ROOT/'runs/train_test_v1')
    p.add_argument('--split',choices=['test_two_rooms','test_multi_room','both','test_same_map','test_new_map','dev_same_map','dev_new_map','dev_both'],default='both')
    p.add_argument('--task',default='set_table',choices=['set_table','clear_and_store','prepare_and_serve_meal','unpack_and_sort_groceries','warm_two_pots','all'])
    p.add_argument('--history-root',type=Path,default=ROOT/'runs/train_test_v1/train')
    p.add_argument('--history-manifest',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--effort',default='high')
    p.add_argument('--max-calls',type=int,default=100)
    p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--control-steps',type=int,help='Simulated budget as 50 Hz control steps; overrides --seconds')
    p.add_argument('--wall-seconds',type=float,help='Per-episode wall-clock limit from episode start; omitted means none')
    p.add_argument('--request-timeout',type=float,default=240)
    p.add_argument('--api-retries',type=int,default=6,help='openai-api: retries of rate-limit/connection/server errors per request')
    p.add_argument('--stateless',action='store_true',help='openai-api: resend the thread each call (store=False) instead of chaining stored responses')
    p.add_argument('--resume',action='store_true',help='Continue in an existing --output; several workers may share it')
    p.add_argument('--check-model',action='store_true',help='openai-api: verify the key and model ID, then exit')
    p.add_argument('--log-interval',type=float,default=15,help='Heartbeat interval during scene/model/tool work, seconds')
    p.add_argument('--limit',type=int,help='Maximum episodes; omitted evaluates all matching episodes')
    p.add_argument('--mock',action='store_true',help='Offline physical wiring test; not a policy evaluation')
    p.add_argument('--preflight',action='store_true',help='Validate inputs without creating simulator/model')
    p.add_argument('--controller-config',type=Path,default=ROOT/'configs/controllers/molmospaces_mobile_franka.json')
    a=p.parse_args()
    if a.source and a.episode_manifest:p.error('Use --source or --episode-manifest, not both')
    if a.control_steps is not None:a.seconds=a.control_steps*.02
    if min(a.max_calls,a.seconds,a.request_timeout,a.log_interval)<=0 or (a.limit is not None and a.limit<=0):p.error('Limits must be positive')
    if a.wall_seconds is not None and a.wall_seconds<=0:p.error('Limits must be positive')
    if a.resume and not a.output:p.error('--resume needs --output')
    from cross_episode_sim.agent_policy.history import catalog,public_catalog,frames
    from cross_episode_sim.agent_policy.interface import validate
    selection=None;rooms={};source_splits={}
    if a.episode_manifest:
        selection=json.loads(a.episode_manifest.read_text())
        sources=[]
        for entry in selection['episodes']:
            source=Path(entry['source'])
            if not source.is_absolute():source=ROOT/source
            source=source.resolve()
            if not json.loads((source/'physics_result.json').read_text()).get('physics_success'):
                raise ValueError(f'Selected source lacks successful physics: {source}')
            task=json.loads((source/'config.json').read_text())['task']
            if a.task!='all' and task!=a.task:raise ValueError(f'Selected task differs from --task: {source}')
            sources.append(source)
        if len(set(sources))!=len(sources):raise ValueError('Duplicate episodes in manifest')
    elif a.source:sources=[a.source.resolve()]
    else:
        from cross_episode_sim.evaluation.test_sources import test_sources
        splits=(['test_two_rooms','test_multi_room'] if a.split=='both' else
                ['dev_same_map','dev_new_map'] if a.split=='dev_both' else [a.split])
        sources=test_sources(a.dataset,splits,a.task)
        # By manifest, not by resolved location: v4 test_same_map episodes are
        # links to the v3 episodes they keep, so they resolve outside the split.
        for split in splits:
            source_splits.update({s:split for s in test_sources(a.dataset,[split],a.task)})
        from cross_episode_sim.agent_policy.batch import room_counts
        rooms=room_counts(a.dataset,splits)
    if a.limit:sources=sources[:a.limit]
    if not sources:raise ValueError('No evaluation scenes matched')
    for source in sources:
        # scene.xml: frozen fixture joints are restored from it at load (articulated_scene).
        for name in ['action_recording/model.mjb','action_recording/initial_state.npz','scene.xml','initial_snapshot.json','config.json','task_manifest.json','instruction.txt']:
            if not (source/name).is_file():raise FileNotFoundError(source/name)
    from cross_episode_sim.agent_policy.progress import Progress
    batch_log=Progress('Astra',interval=a.log_interval)
    batch_log.log(f'Selected {len(sources)} episodes | task={a.task} split={a.split} framework={a.framework} memory={a.memory}')
    histories=[]
    if a.memory=='all':
        with batch_log.stage(f'Index history videos from {a.history_root}'):
            histories=catalog(a.history_root)
        selected_tasks={json.loads((source/'config.json').read_text())['task'] for source in sources}
        histories=[h for h in histories if h['task'] in selected_tasks]
        if not histories:raise ValueError('No fine-tuning history videos match the selected tasks')
        batch_log.log(f'History ready: {len(histories)} videos')
        if a.history_manifest:
            supplied=json.loads(a.history_manifest.read_text())
            if supplied!=histories:raise ValueError('History manifest differs from allowed fine-tuning pool')
        if {str(s) for s in sources}&{h['source'] for h in histories}:raise ValueError('Evaluation/history overlap')
    from collections import Counter
    print(json.dumps({'episodes':len(sources),'history_videos':len(histories),'framework':a.framework,'memory':a.memory,'model':a.model,
        'by_task':dict(Counter(json.loads((s/'config.json').read_text())['task'] for s in sources)),
        'by_split':dict(Counter(source_splits[s] for s in sources if s in source_splits)),
        'limits':dict(calls=a.max_calls,control_steps=round(a.seconds*50),simulated_seconds=a.seconds,
                      wall_seconds=a.wall_seconds,request_timeout=a.request_timeout)}),flush=True)
    if a.check_model or (a.framework=='openai-api' and not a.mock and not a.preflight):
        from cross_episode_sim.agent_policy.openai_api import check_model
        found=check_model(a.model)
        batch_log.log(f'Model {a.model}: available' if found else f'Model {a.model}: this key may not read the model list; the first request will tell')
    if a.preflight or a.check_model:return
    from cross_episode_sim.agent_policy.batch import INVALID,claim,summarize,write_json
    if a.output:a.output.mkdir(parents=True,exist_ok=a.resume);output=a.output.resolve()
    else:
        (ROOT/'runs').mkdir(exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix=f'astra_{a.framework}_{a.memory}_',dir=ROOT/'runs'))
    print('OUTPUT='+str(output),flush=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(sources=[str(s) for s in sources],current_episode_history=True,
                  method='cross_sim_direct_responses_api_adaptation' if a.framework=='openai-api' else 'cross_sim_framework_client_adaptation',
                  controller='native MolmoSpaces joint position; optional native IK',control_hz=50,
                  evaluation_metrics_version='semantic_progress_v1')
    if a.resume and (output/'config.json').exists():
        saved=json.loads((output/'config.json').read_text())
        fixed=('framework','memory','model','effort','max_calls','seconds','wall_seconds','request_timeout','stateless','mock','controller_config','sources','evaluation_metrics_version')
        changed=[k for k in fixed if saved.get(k)!=config[k]]
        if changed or json.loads((output/'history_manifest.json').read_text())!=histories:
            raise ValueError(f'--resume: {output} was started with different settings: {changed or ["history videos"]}')
    else:
        write_json(output/'config.json',config)
        write_json(output/'history_manifest.json',histories)
        if selection is not None:write_json(output/'episode_manifest.json',selection)
    from tqdm import tqdm
    for index,source in enumerate(tqdm(sources,desc='Astra episodes',unit='episode',disable=None)):  # no bar in a log file
        folder=output/f'{index:04d}_{source.name}'
        if not claim(folder,a.resume):
            batch_log.log(f'Episode {index+1}/{len(sources)} | finished or held by another worker: {folder.name}')
            continue
        batch_log.log(f'Episode {index+1}/{len(sources)} | source={source}')
        episode_task=json.loads((source/'config.json').read_text())['task']
        episode_histories=[h for h in histories if h['task']==episode_task]
        result=run_episode(a,source,folder,episode_histories,rooms.get(source),source_splits.get(source))
        summary=summarize(output,len(sources),a.mock)
        batch_log.log(f'Completed {summary["episodes"]}/{len(sources)} | successes={summary["successes"]} | service_errors={summary["service_errors"]}')
        if result['termination']==INVALID:
            batch_log.log(f'Stopping this worker, the model service failed: {result["failure"]} | fix it and rerun with --resume')
            raise SystemExit(2)
    summary=summarize(output,len(sources),a.mock)
    print(json.dumps(summary),flush=True)


def run_episode(a,source,output,histories,room_count=None,split=None):
    from cross_episode_sim.agent_policy.environment import Environment
    from cross_episode_sim.agent_policy.backends import Backend
    from cross_episode_sim.agent_policy.interface import validate,problem
    from cross_episode_sim.agent_policy.history import frames,public_catalog
    from cross_episode_sim.agent_policy.openai_api import ModelServiceError,RequestTimeout,ResponsesBackend,WallTimeLimit
    from cross_episode_sim.agent_policy.batch import INVALID,write_json
    import jsonschema
    output.mkdir(exist_ok=True);env=None;backend=None;failure=None;calls=0;termination='call_limit';start=time.monotonic()
    from cross_episode_sim.agent_policy.progress import Progress
    log=Progress(output.name,output/'run.log',a.log_interval)
    log.log(f'framework={a.framework} model={a.model} effort={a.effort} memory={a.memory} history_videos={len(histories)}')
    log.log(f'Limits: {a.max_calls} calls, {a.seconds:g}s simulated ({round(a.seconds*50)} control steps), '
            f'{"no" if a.wall_seconds is None else format(a.wall_seconds,"g")+"s"} wall limit, {a.request_timeout:g}s per request | output={output}')
    deadline=None if a.wall_seconds is None else start+a.wall_seconds;usage_total={};steps=0
    task=json.loads((source/'config.json').read_text())['task']
    feedback={'ok':True,'message':'Initial observation'};extra=[];goals={};simtime=0.;success=False
    task_progress=None;first_success=None
    def record_progress(phase):
        # Private evaluator only: never include these fields in obs or feedback.
        nonlocal task_progress,first_success
        task_progress=env.scorer.progress(env.initial,env.data)
        snapshot=dict(phase=phase,calls=calls,control_steps=env.steps,
            simulated_seconds=float(env.data.time)-env.start,wall_seconds=time.monotonic()-start,
            task_progress=task_progress,success=env.stable>=2,goal_evidence=env.goals())
        if snapshot['success'] and first_success is None:
            first_success={k:snapshot[k] for k in ('calls','control_steps','simulated_seconds','wall_seconds')}
        with (output/'progress_trace.jsonl').open('a') as stream:stream.write(json.dumps(snapshot,allow_nan=False)+'\n')
        write_json(output/'progress.json',snapshot)
    try:
        with log.stage('Scene setup and initial rendering'):
            env=Environment(source,output,a.controller_config)
        record_progress('initial')
        log.log(f'Task: {env.prompt}')
        if not a.mock:
            with log.stage('Start model client'):
                if a.framework=='openai-api':backend=ResponsesBackend(a.model,a.effort,a.memory,a.request_timeout,a.stateless,a.api_retries)
                else:backend=Backend(a.framework,a.model,a.effort,a.memory,a.request_timeout,output)
        for turn in range(a.max_calls):
            call_label=f'Call {turn+1}/{a.max_calls}'
            with log.stage(f'{call_label}: capture current views'):
                state,images=env.observe(turn)
            sent_images=images+extra
            obs={'task':env.prompt,'robot':state,'last_tool_result':feedback}
            if turn==0 and a.memory=='all':obs['history_catalog']=public_catalog(histories)
            log.log(f'{call_label}: input={len(images)} current images + {len(extra)} historical images | sim={state["simulation_seconds"]:.2f}s')
            request_start=time.monotonic()
            if a.mock:
                decision=({'name':'wait','arguments':{'seconds':.2}} if turn==0 else {'name':'finish','arguments':{}})
            else:
                with log.stage(f'{call_label}: waiting for {a.model} (timeout {a.request_timeout:g}s)'):
                    decision=backend.decide(obs,sent_images,deadline)
            response_seconds=time.monotonic()-request_start
            log.log(f'{call_label}: decision after {response_seconds:.1f}s: {json.dumps(decision,ensure_ascii=False)}')
            usage=getattr(backend,'last_usage',None)
            if usage:
                log.log(f'{call_label}: token usage {json.dumps(usage)}')
                for key,value in usage.items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool):usage_total[key]=usage_total.get(key,0)+value
            calls+=1;extra=[]
            record={'turn':turn,'observation':obs,'images':[str(x) for x in sent_images],
                    'decision':decision,'usage':getattr(backend,'last_usage',None),
                    'response':getattr(backend,'last_response',None),'model_response_seconds':response_seconds}
            try:
                validate(decision,a.memory)
                name,args=decision['name'],decision['arguments']
                if name=='finish':termination='agent_finished';feedback={'ok':True}
                elif name=='history_frame':
                    with log.stage(f'{call_label}: inspect history {args["video_id"]} at {args["seconds"]}s'):
                        extra=frames(histories,args['video_id'],args['seconds'],output)
                    feedback={'ok':True,'history_video_id':args['video_id'],'timestamps':args['seconds'],'historical_images':[x.name for x in extra]}
                else:
                    remaining=a.seconds-(float(env.data.time)-env.start)
                    if args.get('seconds',0)>remaining:
                        log.log(f'{call_label}: requested duration exceeds remaining simulation budget ({remaining:.2f}s)')
                        termination='simulation_time_limit';break
                    before=float(env.data.time)
                    try:
                        with log.stage(f'{call_label}: execute {name}'):
                            feedback=env.act(name,args,deadline)
                    except (ValueError,RuntimeError) as exc:
                        feedback={'ok':False,'error':str(exc),'executed_seconds':float(env.data.time)-before}
                        if 'Physics diverged' in str(exc):raise
                record['feedback']=feedback
            except jsonschema.ValidationError:
                feedback={'ok':False,'error':'Invalid tool call: '+problem(decision,a.memory),'executed_seconds':0};record['feedback']=feedback
            except (ValueError,KeyError,TypeError) as exc:
                feedback={'ok':False,'error':str(exc),'executed_seconds':0};record['feedback']=feedback
            with (output/'exchanges.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            log.log(f'{call_label}: feedback {json.dumps(feedback,ensure_ascii=False)} | sim={float(env.data.time)-env.start:.2f}s')
            goals=env.goals();success=env.stable>=2
            log.log(f'{call_label}: goals {sum(bool(v) for v in goals.values())}/{len(goals)} | {json.dumps(goals)} | stable_checks={env.stable}/2')
            record_progress('decision_end')
            log.log(f'{call_label}: task progress {task_progress["done"]}/{task_progress["total"]} | rate={task_progress["rate"]}')
            if success:termination='success';break
            if termination=='agent_finished':break
            if float(env.data.time)-env.start>=a.seconds:termination='simulation_time_limit';break
            if deadline is not None and time.monotonic()>=deadline:termination='wall_time_limit';break
    except WallTimeLimit as exc:
        termination='wall_time_limit';log.log(str(exc))
    except RequestTimeout as exc:
        failure=str(exc);termination='request_timeout';log.log(f'ERROR: {failure}')
    except ModelServiceError as exc:
        failure=str(exc);termination=INVALID;log.log(f'MODEL SERVICE ERROR (episode not scored): {failure}')
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}';termination='error'
        log.log(f'ERROR: {failure}')
    finally:
        if backend:backend.close()
        if env:
            simtime=float(env.data.time)-env.start;steps=env.steps;goals=env.goals()
            try:record_progress('final')
            finally:env.close()
    result={'source':str(source),'task':task,'room_count':room_count,'split':split,'success':success and not failure,'mock':a.mock,'failure':failure,
            'termination':termination,'goal_evidence':goals,'calls':calls,'simulated_seconds':simtime,'control_steps':steps,
            'task_progress':task_progress,'first_success':first_success,
            'wall_seconds':time.monotonic()-start,'usage':usage_total,'retry_wait_seconds':getattr(backend,'retry_wait',0.),
            'video':str(output/'rollout.mp4')}
    write_json(output/'result.json',result)
    log.log(f'FINISHED: success={result["success"]} reason={termination} calls={calls} steps={steps} sim={simtime:.2f}s wall={result["wall_seconds"]:.1f}s')
    log.log(f'Video: {result["video"]} | Result: {output / "result.json"}')
    return result

if __name__=='__main__':main()
