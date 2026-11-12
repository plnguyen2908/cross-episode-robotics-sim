"""Close an observed appliance panel through the shared panel-push backend."""
import json
import sys
import numpy as np
from .closing_redock import closing_face
from ..control.sensor_appliance_backend import camera_points
from ..skills.m2t2_interface import SkillOutcome


def push_close(controls,target,root,progress):
    T=np.linalg.inv(controls.estimated_pose())
    memory=controls.hinges[target]
    point,normal=closing_face(memory)
    point=T[:3,:3]@point+T[:3,3];normal=T[:3,:3]@normal
    packet=controls.packet()
    points=np.concatenate([camera_points(cam)[0] for cam in packet.cameras.values()])
    source=root/'observed_panel';source.mkdir(parents=True,exist_ok=False)
    # Only a prior RGB-D estimate and current measured geometry enter the
    # contact backend. No learned grasp or simulator fixture pose is invented.
    local=dict(open_handle=point.tolist(),open_normal=normal.tolist(),
        push_edge_margins=(.12,.18,.04) if target=="microwave" else (.18,.25,.32),
        hinge_xy=(T[:3,:3]@memory['hinge']+T[:3,3])[:2].tolist(),
        closed_normal=(T[:3,:3]@memory['closed_normal']).tolist())
    reference=source/'reference.json';reference.write_text(json.dumps(local))
    (source/'report.json').write_text(json.dumps(dict(surface_info=dict(panel_normal=normal.tolist()),
        scope='observed panel seed; no handle grasp performed')))
    patch=points[np.linalg.norm(points-point,axis=1)<.18]
    np.savez(source/'handle_surface.npz',points=patch)
    # Preserve already observed floor coverage for coordinated base contact.
    # The contact backend otherwise sees only the small current camera footprint.
    grid=getattr(controls.runtime,'observed_navigation_map',None)
    if grid is not None:
        from shapely.geometry import box
        from shapely.ops import unary_union
        from ..control.contact_retreat import transform_support_region
        rr,cc=np.where(grid.floor_views>=grid.config.minimum_floor_views)
        resolution=grid.config.resolution
        xy=grid.origin+np.c_[cc,rr]*resolution
        floor=unary_union([box(x,y,x+resolution,y+resolution) for x,y in xy])
        controls.runtime.contact_support_region=transform_support_region(floor,T)
    progress('panel_push_close','running')
    command=[sys.executable,'scripts/test_sensor_hinged_cycle.py','--grasp-run',str(source),
        '--output',str(root/'motion'),'--panel-close-reference',str(reference),
        '--open-only','--handle-then-push','--push-after','.001','--pull-push-after','25',
        '--contact-ik','finite','--contact-motion-profile','fast','--base-push',
        '--planar-push','--base-yaw-follow','--lateral-base-fraction','1','--exact-closed-fingers',
        '--calibrated-contact-boxes','--local-panel-reach','--orientation-gain','.5']
    try:
        controls.run_script(command,root)
    finally:
        # The reused backend records opening-oriented continuation metadata.
        # Never offer Qwen an open-door continuation after a closing attempt.
        controls.runtime.door_push_memory=None
    report=json.loads((root/'motion/report.json').read_text())
    if report.get('error'):raise RuntimeError(report['error'])
    progress('panel_push_close','completed')
    controls.navigation_posture(root,progress)
    return SkillOutcome(True,'observed_panel_push_completed; final goal scored separately')
