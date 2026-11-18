"""Observe a semantic basket anchor, approach, then acquire placement geometry.

All geometry comes from onboard RGB-D and the episode's estimated map. The
anchor is not release geometry; it is never a simulator basket mesh.
Basket redocking uses Qwen to rank observed facing poses, then the shared
checked-route planner. This is independent of cross-room navigation and the
qualified set-table placement path.
"""
import json
import sys
import numpy as np
from . import artifacts


def refine_basket_dock(pipeline,root,source,receptacle,description,station,number,retry=True):
    """Observe after arrival, then physically recenter before arm planning.

    A redock that completes its route but ends just outside the heading
    tolerance is observed again and recentred once from where it stopped.
    """
    from ..perception.receptacle_front import interior_dock_anchor,observed_front,frontal_heading_error
    perception=root/'basket_close_perception'
    pipeline.stage(f'{number}_observe_interior',[sys.executable,'scripts/test_basket_place.py',
        '--dock',str(source),'--receptacle',description,'--semantic-receptacle',receptacle,
        '--observe-only','--output',str(perception)],perception/'report.json','observation_success')
    evidence=json.loads(artifacts.read_text(perception/'report.json'))
    if not evidence.get('observation_success'):return source
    data=artifacts.load_array(perception/'input.npz')
    anchor=interior_dock_anchor(data['table_surface'])
    front=observed_front(data['points'],anchor)
    aligned=abs(anchor[1])<.05 and .50<=anchor[0]<=.65
    if front is not None:aligned=aligned and frontal_heading_error([0,0,0],front['normal_base'])<np.deg2rad(5)
    evidence.update(interior_refinement=True,station=station,interior_anchor_base=anchor.tolist(),
                    refinement_needed=not aligned,pre_dock_arm_ik=False)
    artifacts.write_text(perception/'report.json',json.dumps(evidence))
    print('BASKET_INTERIOR_REDOCK',json.dumps(evidence),flush=True)
    if aligned:
        if not retry:
            # The stop pose of the missed redock, now measured as aligned.
            arrival=json.loads(artifacts.read_text(source/'report.json'))
            artifacts.write_text(source/'report.json',json.dumps(dict(arrival,docking_success=True,
                docking_confirmed_by='fresh interior observation after the redock stopped')))
        return source
    dock=root/'basket_refined_dock'
    pipeline.stage(f'{number}_refine_dock',[sys.executable,'scripts/test_basket_place_redock.py',
        '--start',str(source),'--perception',str(perception),'--continuation-map',str(source),
        '--output',str(dock)],dock/'report.json','docking_success')
    result=json.loads(artifacts.read_text(dock/'report.json'))
    if not result.get('docking_success') and retry and result.get('error') is None and result.get('fixed_map_unchanged'):
        # The route was completed with the object held; only the final heading missed.
        print('BASKET_REDOCK_RETRY',json.dumps(dict(heading_error_rad=result.get('docking_heading_error_rad'))),flush=True)
        again=root/'basket_redock_retry';again.mkdir(exist_ok=True)
        stopped=pipeline.checkpoint(again/'handoff',dock)
        artifacts.save_array(stopped/'observed_payload_tool.npy',artifacts.load_array(dock/'observed_payload_tool.npy'))
        artifacts.write_text(stopped/'report.json',json.dumps(dict(result,initialization_valid=True)))
        return refine_basket_dock(pipeline,again,stopped,receptacle,description,station,f'{number}_retry',retry=False)
    if not result.get('docking_success'):
        raise RuntimeError('Basket interior redock failed: '+str(result.get('error')))
    # Placement must observe again after moving; the old cloud is not reused.
    return dock


def place_in_basket(pipeline, root, carry, receptacle, description, station, number, progress):
    from ..navigation.receptacle_approach import approach_receptacle
    from ..control.raised_payload_posture import raise_payload_clearance,tuck_payload_before_place

    runtime = pipeline.runtime
    payload = artifacts.load_array(carry/'observed_payload_tool.npy')
    progress('basket_interior_view', 'running')
    # Carry already reaches the kitchen. Inspect this view before doing any
    # additional base motion; Qwen selects the actual placement dock below.
    view = pipeline.checkpoint(root/'basket_view_handoff', carry)
    artifacts.save_array(view/'observed_payload_tool.npy', payload)
    artifacts.write_text(view/'report.json', json.dumps(dict(
        initialization_valid=True, scope='current carry view; retained grasp')))

    perception = root/'receptacle_perception'
    script = 'scripts/test_basket_place.py'
    observe = 'scripts/observe_receptacle_approach.py'
    pipeline.stage(f'{number}_observe_receptacle', [sys.executable, observe,
        '--dock', str(view), '--receptacle', receptacle, '--output', str(perception)],
        perception/'report.json', 'observation_success')
    evidence = json.loads(artifacts.read_text(perception/'report.json'))
    # Approach needs a sensed target, not a certified release footprint.
    usable=lambda value: value.get('observation_success')
    if not usable(evidence):
        # Obtain a closer view before raising the object into the camera axis.
        # This approach only obtains observation; Qwen still selects the
        # manipulation dock using arm reach and payload feasibility below.
        progress('basket_view_recovery', 'running')
        approach_receptacle(runtime,receptacle,root/'basket_approach',payload)
        view=pipeline.checkpoint(root/'basket_approach_handoff',view)
        artifacts.save_array(view/'observed_payload_tool.npy',payload)
        artifacts.write_text(view/'report.json',json.dumps(dict(initialization_valid=True)))
        perception=root/'receptacle_perception_retry'
        pipeline.stage(f'{number}_observe_receptacle_retry',[sys.executable,observe,
            '--dock',str(view),'--receptacle',receptacle,'--output',str(perception)],
            perception/'report.json','observation_success')
        evidence=json.loads(artifacts.read_text(perception/'report.json'))
        if not usable(evidence):
            raise_payload_clearance(runtime,payload,root/'basket_view_recovery',target_height=1.3)
            view=pipeline.checkpoint(root/'basket_recovery_handoff',view)
            artifacts.save_array(view/'observed_payload_tool.npy',payload)
            artifacts.write_text(view/'report.json',json.dumps(dict(initialization_valid=True)))
            perception=root/'receptacle_perception_clearance'
            pipeline.stage(f'{number}_observe_receptacle_clearance',[sys.executable,observe,
                '--dock',str(view),'--receptacle',receptacle,'--output',str(perception)],
                perception/'report.json','observation_success')
            evidence=json.loads(artifacts.read_text(perception/'report.json'))
            if not usable(evidence):
                raise RuntimeError('Basket target still unobserved after physical view recovery; no approach target available')
        progress('basket_view_recovery','completed')
    # Preserve failed observation attempts, but identify their successful
    # same-call recovery so orchestration does not mislabel the completed skill.
    observation_stages=[stage for stage in getattr(pipeline,'stages',[])
                        if stage['name'].startswith(f'{number}_observe_receptacle')]
    if observation_stages and observation_stages[-1]['success']:
        for stage in observation_stages[:-1]:
            if not stage['success']:stage['recovered_by']=observation_stages[-1]['name']
    evidence['station'] = station
    artifacts.write_text(perception/'report.json', json.dumps(evidence))

    # Retain the sensed navigation anchor during a checked clearance motion.
    # Fresh support reconstruction happens only after docking.
    raise_payload_clearance(runtime, payload, root/'basket_clearance', target_height=1.3)
    raised = pipeline.checkpoint(root/'basket_clearance_handoff', view)
    artifacts.save_array(raised/'observed_payload_tool.npy', payload)
    # A checkpoint stores state/map continuity, not stage-result metadata.
    # Redocking also consumes the semantic arrival record after its motion.
    artifacts.write_text(raised/'report.json', artifacts.read_text(view/'report.json'))
    progress('basket_interior_view', 'completed')
    dock = root/'place_dock'
    pipeline.stage(f'{number}_place_dock', [sys.executable,
        'scripts/test_basket_place_redock.py', '--start', str(raised),
        '--perception', str(perception), '--continuation-map', str(raised),
        '--interior-view-approach', '--output', str(dock)],
        dock/'report.json', 'docking_success')
    placed = root/'place'
    docking=json.loads(artifacts.read_text(dock/'report.json'))
    if not docking.get('docking_success'):
        # Keep the real current state; do not feed a failed dock into a stage
        # which requires a successful approach, obscuring its original reason.
        placed.mkdir(parents=True,exist_ok=True)
        artifacts.write_text(placed/'report.json',json.dumps(dict(execution_success=False,
            error=docking.get('error','Local docking failed'),phase='docking',
            manipulation_attempted=False)))
        return placed,dock
    placement_source=dock
    if getattr(runtime,'sensors',None) is not None and docking.get('placement_target_base') is not None:
        from ..perception.target_visibility import robot_occlusion_fraction
        packet=runtime.sensors.capture()
        blocked=robot_occlusion_fraction(packet.cameras['navigation'],docking['placement_target_base'])
        print('BASKET_NAV_OCCLUSION',blocked,flush=True)
        if blocked>.15:
            for index,lateral in enumerate((-.25,.25)):
                try:
                    raise_payload_clearance(runtime,payload,root/f'basket_initial_reveal_{index}',
                                            target_height=1.3,lateral_offset=lateral)
                    placement_source=pipeline.checkpoint(root/'basket_visible_handoff',dock)
                    artifacts.save_array(placement_source/'observed_payload_tool.npy',payload)
                    artifacts.write_text(placement_source/'report.json',json.dumps(docking))
                    break
                except RuntimeError as exc:
                    print('BASKET_INITIAL_REVEAL_UNAVAILABLE',str(exc),flush=True)
    dock=placement_source
    docking=json.loads(artifacts.read_text(dock/'report.json'))
    progress('loaded_tuck_before_place','running')
    tuck_payload_before_place(runtime,payload,root/'loaded_tuck')
    placement_source=pipeline.checkpoint(root/'loaded_tuck_handoff',dock)
    artifacts.save_array(placement_source/'observed_payload_tool.npy',payload)
    artifacts.write_text(placement_source/'report.json',json.dumps(docking))
    progress('loaded_tuck_before_place','completed')
    # The loaded tuck changes visibility. Refine from this final view, not
    # from the preceding raised arm that may occlude the basket floor.
    placement_source=refine_basket_dock(pipeline,root,placement_source,receptacle,description,station,number)
    dock=placement_source
    docking=json.loads(artifacts.read_text(dock/'report.json'))
    pipeline.stage(f'{number}_place', [sys.executable, script, '--dock', str(placement_source),
        '--receptacle', description, '--semantic-receptacle', receptacle,
        '--output', str(placed)], placed/'report.json', 'execution_success')
    result=json.loads(artifacts.read_text(placed/'report.json'))
    # The first dock is unconditional on manipulation feasibility. Only now,
    # after actual close-range feedback, uncover a hidden basket with the arm.
    # phase=observe means no insertion/release commands have been executed.
    for index,lateral in enumerate((-.25,.25)):
        if result.get('execution_success') or result.get('phase')!='observe':break
        progress('basket_view_clearance','running')
        try:
            raise_payload_clearance(runtime,payload,root/f'basket_reveal_{index}',
                                    target_height=1.3,lateral_offset=lateral)
        except RuntimeError as exc:
            print('BASKET_REVEAL_UNAVAILABLE',str(exc),flush=True)
            continue
        source=pipeline.checkpoint(root/f'basket_reveal_handoff_{index}',dock)
        artifacts.save_array(source/'observed_payload_tool.npy',payload)
        artifacts.write_text(source/'report.json',json.dumps(docking))
        # Arm-only reobservation can reveal a far interior. Move the base to
        # that newly measured floor before asking IK to reach it again.
        recovery=root/f'basket_recovery_{index}';recovery.mkdir(exist_ok=True)
        source=refine_basket_dock(pipeline,recovery,source,receptacle,description,station,f'{number}_recovery_{index}')
        dock=source;docking=json.loads(artifacts.read_text(dock/'report.json'))
        # A long held item can itself hide the basket from the wrist camera once
        # the arm is tucked again. Try the place from the raised arm, which just
        # showed the interior, before tucking.
        revealed=root/f'place_revealed_{index}';name=f'{number}_place_revealed_{index}'
        pipeline.stage(name,[sys.executable,script,'--dock',str(dock),
            '--receptacle',description,'--semantic-receptacle',receptacle,
            '--output',str(revealed)],revealed/'report.json','execution_success')
        result=json.loads(artifacts.read_text(revealed/'report.json'))
        if result.get('execution_success') or result.get('phase')!='observe':
            placed=revealed
            if result.get('execution_success'):
                for stage in getattr(pipeline,'stages',[]):
                    if stage['name'].startswith(f'{number}_place') and not stage['success']:
                        stage['recovered_by']=name
            progress('basket_view_clearance','completed')
            continue
        tuck_payload_before_place(runtime,payload,recovery/'loaded_tuck')
        source=pipeline.checkpoint(recovery/'placement_handoff',dock)
        artifacts.save_array(source/'observed_payload_tool.npy',payload)
        artifacts.write_text(source/'report.json',json.dumps(docking))
        retry=root/f'place_reobserve_{index}'
        name=f'{number}_place_reobserve_{index}'
        pipeline.stage(name,[sys.executable,script,'--dock',str(source),
            '--receptacle',description,'--semantic-receptacle',receptacle,
            '--output',str(retry)],retry/'report.json','execution_success')
        placed=retry;result=json.loads(artifacts.read_text(placed/'report.json'))
        if result.get('execution_success'):
            for stage in getattr(pipeline,'stages',[]):
                if stage['name'].startswith(f'{number}_place') and not stage['success']:
                    stage['recovered_by']=name
        progress('basket_view_clearance','completed')
    from ..mapping.continuity import load_memory,save_memory
    save_memory(placed,load_memory(carry))
    return placed, dock
