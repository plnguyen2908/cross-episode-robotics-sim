"""Observe basket support before lifting the retained payload out of the view.

All geometry comes from onboard RGB-D and the episode's estimated map. The
cached floor is reanchored by docking; it is never a simulator basket mesh.
"""
import json
import sys
from . import artifacts


def place_in_basket(pipeline, root, carry, receptacle, description, station, number, progress):
    from ..navigation.receptacle_approach import approach_receptacle
    from ..control.raised_payload_posture import raise_payload_clearance

    runtime = pipeline.runtime
    payload = artifacts.load_array(carry/'observed_payload_tool.npy')
    progress('basket_interior_view', 'running')
    approach_receptacle(runtime, receptacle, root/'basket_approach', payload)
    view = pipeline.checkpoint(root/'basket_view_handoff', carry)
    artifacts.save_array(view/'observed_payload_tool.npy', payload)
    artifacts.write_text(view/'report.json', json.dumps(dict(
        initialization_valid=True, scope='observed basket approach; retained grasp')))

    perception = root/'receptacle_perception'
    script = 'scripts/test_basket_place.py'
    pipeline.stage(f'{number}_observe_receptacle', [sys.executable, script,
        '--dock', str(view), '--observe-only', '--receptacle', description,
        '--semantic-receptacle', receptacle, '--output', str(perception)],
        perception/'report.json', 'observation_success')
    evidence = json.loads(artifacts.read_text(perception/'report.json'))
    evidence['station'] = station
    artifacts.write_text(perception/'report.json', json.dumps(evidence))

    # A raised arm may occlude the interior. Preserve the already measured
    # floor instead of asking the VLM to hallucinate a now-hidden surface.
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
    pipeline.stage(f'{number}_place', [sys.executable, script, '--dock', str(dock),
        '--receptacle', description, '--semantic-receptacle', receptacle,
        '--output', str(placed)], placed/'report.json', 'execution_success')
    return placed, dock
