"""Approach a visible receptacle before requiring a view of its interior.

This stage proposes no release and certifies no support surface. The ordinary
placement stage still observes the interior, plans insertion and verifies it.
"""
import json,math
import numpy as np
from PIL import Image,ImageDraw
from .shaped_docking import RobotGeometry,ShapedDocking
from .placement_docks import placement_docks,plan_to_any_dock
from .dock_follower import DockFollower
from ..control.observed_payload import ObservedPayload
from ..control.onboard_base import OnboardBaseActuator
from ..control.sensor_appliance_backend import pixel_point,camera_points
from ..perception.molmo_pointing import MolmoPointer
from ..perception.daaam_grounding import LocalVLM,parse_label
from ..recording.deferred import StateVideoRecorder


def approach_receptacle(runtime,label,output,tool_points,seconds=40.):
    output.mkdir(parents=True,exist_ok=False)
    geometry=RobotGeometry(runtime.model);payload=ObservedPayload(geometry,tool_points)
    grid=runtime.observed_navigation_map;tracker=runtime.mapper.view()
    raw=runtime.sensors.capture(include_robot_mask=False)
    estimate=tracker.update(raw.cameras['navigation'],raw.timestamp)
    if not estimate.accepted:raise RuntimeError('Receptacle approach needs localization')
    pointer=MolmoPointer();verifier=LocalVLM(device='cuda:0');target=None;observations=[]
    for camera in raw.cameras.values():
        image=Image.fromarray(camera.rgb);answer=pointer.point(image,label)
        _,_,valid=camera_points(camera)
        for pixel in answer.get('points',[]):
            marked=image.copy();draw=ImageDraw.Draw(marked);x,y=pixel
            draw.ellipse((x-6,y-6,x+6,y+6),outline='magenta',width=3)
            reply=verifier.label_with_prompt([image,marked],
                'Does the magenta point belong to the requested '+label+'? '
                'Accept its rim or outer body; the interior does not need to be visible yet. '
                'Return JSON category "'+label+'", "other" or "unknown", and description.',max_new_tokens=120)
            try:accepted=parse_label(reply,(label,))['category']==label
            except (ValueError,KeyError):accepted=False
            observations.append(dict(camera=camera.frame_id,pixel=pixel,response=reply,accepted=accepted))
            if not accepted:continue
            try:local=pixel_point(camera,valid,pixel)
            except ValueError:continue
            T=estimate.T_map_base;target=T[:3,:3]@local+T[:3,3];break
        if target is not None:break
    (output/'observations.json').write_text(json.dumps(observations,indent=2))
    if target is None:raise RuntimeError('Requested receptacle not visually confirmed: '+label)
    base=OnboardBaseActuator(runtime.model,runtime.data,motion_guard=lambda forward,yaw,dt:follower is not None);checker=None;follower=None
    class LoadedDock(ShapedDocking):
        def make_checker(self,pose,q):return checker
    docking=LoadedDock(grid,geometry)
    start=float(runtime.data.time);budget=seconds;tick=0
    with StateVideoRecorder(runtime.model,runtime.data,output/'state_recording') as recorder:
        while float(runtime.data.time)-start<budget:
            tick+=1
            if tick==1 or tick%5==0:
                packet=payload.filter_packet(runtime.sensors.capture(include_robot_mask=False))
                estimate=tracker.update(packet.cameras['navigation'],packet.timestamp)
                recorder.record_map(grid,estimate,'receptacle_view_approach')
            if not estimate.accepted:base.step(0.,0.);continue
            T=estimate.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
            if checker is None:
                checker=payload.checker(grid,packet.arm.position,pose,list(getattr(grid,'inherited_support_poses',[]))+[pose])
            checker.remember_support(pose)
            if follower is None:
                # Facing a receptacle does not require seeing its bottom first.
                candidates=[]
                for proposal in placement_docks(target[None],pose,distances=(.65,.75,.55,.85)):
                    # The carried hand lies on the camera's forward axis. Look
                    # slightly beside the basket so the hand does not hide it.
                    for offset in (.35,-.35):
                        candidate=proposal.pose.copy();candidate[2]+=offset;candidates.append(candidate)
                if not plan_to_any_dock(docking,pose,packet.arm.position,candidates):
                    raise RuntimeError('No observed loaded approach to '+label+': '+str(docking.events[-1:]))
                from .observed_undock import departure_route
                original=list(docking.path)
                docking.path=departure_route(pose,original,checker,keep_final_rotation=True)
                route=[pose]+docking.path
                distance=sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip(route,route[1:]))
                turns=sum(abs((b[2]-a[2]+np.pi)%(2*np.pi)-np.pi) for a,b in zip(route,route[1:]))
                # Keep checked turns/contact posture; omit redundant lattice
                # stops and derive the allowance from the actual route.
                allowance=max(15.,distance/.15+turns/.5+5.*len(docking.path))
                budget=max(budget,float(runtime.data.time)-start+allowance)
                event=dict(event='receptacle_approach_route',route=[p.tolist() for p in docking.path],
                    removed_waypoints=len(original)-len(docking.path),distance_m=distance,
                    turn_radians=turns,budget_seconds=budget)
                docking.events.append(event);print(json.dumps(event),flush=True)
                (output/'route.json').write_text(json.dumps(event,indent=2))
                follower=DockFollower(docking.path,max_speed=.15,max_yaw_speed=.5)
            try:speed,yaw=follower.command(pose)
            except RuntimeError as exc:
                if str(exc)!='Dock turn drift requires replanning':raise
                follower=None;base.step(0.,0.);continue
            base.step(speed,yaw)
            if follower.complete and base.stationary():
                report=dict(success=True,semantic_target=label,target=target.tolist(),
                    simulated_seconds=float(runtime.data.time)-start,route_events=docking.events,
                    scope='Observed receptacle approach only; no support or placement success certified')
                (output/'report.json').write_text(json.dumps(report,indent=2));return report
    raise RuntimeError('Receptacle approach budget exhausted')
