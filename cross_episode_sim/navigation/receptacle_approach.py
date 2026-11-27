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


# The navigation camera spans about 91 degrees horizontally (75 vertical at
# 4:3), so views one LOOK_STEP apart overlap by half a step.
LOOK_STEP=math.pi/3


def verify_prompt(label):
    """The point check for a receptacle approach. 2026-11-25 (x12_grocery call 13): the old wording ('the interior does
    not need to be visible yet' next to the category list) got category "other" with the description 'The magenta point is
    located on the orange basket, specifically on its rim or outer body, as requested', and the basket place failed."""
    return ('Is the magenta point on the requested '+label+'? Any visible part counts: its rim, side, outer body or handle; '
            'its interior does not need to be visible. Return JSON with "category" and "description". Set "category" to "'+label+
            '" when the point is on the '+label+'; to "other" only when the point is on a different object or surface; '
            'to "unknown" when you cannot tell.')


def look_headings(yaw,step=LOOK_STEP):
    """Headings that finish one turn in place after the view at `yaw`."""
    return [yaw+k*step for k in range(1,int(math.ceil(2*math.pi/step-1e-9)))]


def approach_receptacle(runtime,label,output,tool_points,seconds=40.):
    output.mkdir(parents=True,exist_ok=False)
    geometry=RobotGeometry(runtime.model);payload=ObservedPayload(geometry,tool_points)
    grid=runtime.observed_navigation_map;tracker=runtime.mapper.view()
    pointer=MolmoPointer();verifier=LocalVLM(device='cuda:0');observations=[];looks=[]
    def confirm():
        # The wrist camera looks at the carried object. Its depth inside the payload
        # volume is invalid here, so a point on it can never become the receptacle.
        raw=payload.filter_packet(runtime.sensors.capture(include_robot_mask=False))
        estimate=tracker.update(raw.cameras['navigation'],raw.timestamp)
        if not estimate.accepted:raise RuntimeError('Receptacle approach needs localization')
        for camera in raw.cameras.values():
            image=Image.fromarray(camera.rgb);answer=pointer.point(image,label)
            _,_,valid=camera_points(camera)
            for pixel in answer.get('points',[]):
                marked=image.copy();draw=ImageDraw.Draw(marked);x,y=pixel
                draw.ellipse((x-6,y-6,x+6,y+6),outline='magenta',width=3)
                reply=verifier.label_with_prompt([image,marked],verify_prompt(label),max_new_tokens=120)
                try:accepted=parse_label(reply,(label,))['category']==label
                except (ValueError,KeyError):accepted=False
                observations.append(dict(view=len(looks),camera=camera.frame_id,pixel=pixel,response=reply,accepted=accepted))
                if not accepted:continue
                try:local=pixel_point(camera,valid,pixel)
                except ValueError:continue
                T=estimate.T_map_base;return T[:3,:3]@local+T[:3,3],estimate
        return None,estimate
    base=OnboardBaseActuator(runtime.model,runtime.data,motion_guard=lambda forward,yaw,dt:follower is not None);checker=None;follower=None
    class LoadedDock(ShapedDocking):
        def make_checker(self,pose,q):return checker
    docking=LoadedDock(grid,geometry)
    start=float(runtime.data.time);tick=0;packet=None;estimate=None
    with StateVideoRecorder(runtime.model,runtime.data,output/'state_recording') as recorder:
        def drive(plan,name,budget):
            """Follow one swept-checked route from `plan(pose,packet)`; None means it found no route."""
            nonlocal checker,follower,tick,packet,estimate
            began=float(runtime.data.time);follower=None
            while float(runtime.data.time)-began<budget:
                tick+=1
                if packet is None or tick%5==0:
                    packet=payload.filter_packet(runtime.sensors.capture(include_robot_mask=False))
                    estimate=tracker.update(packet.cameras['navigation'],packet.timestamp)
                    recorder.record_map(grid,estimate,'receptacle_view_approach')
                if not estimate.accepted:base.step(0.,0.);continue
                T=estimate.T_map_base;pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
                if checker is None:
                    checker=payload.checker(grid,packet.arm.position,pose,list(getattr(grid,'inherited_support_poses',[]))+[pose])
                checker.remember_support(pose)
                if follower is None:
                    planned=plan(pose,packet)
                    if planned is None:base.step(0.,0.);return None
                    path,detail=planned;route=[pose]+path
                    distance=sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip(route,route[1:]))
                    turns=sum(abs((b[2]-a[2]+np.pi)%(2*np.pi)-np.pi) for a,b in zip(route,route[1:]))
                    # Keep checked turns/contact posture; omit redundant lattice
                    # stops and derive the allowance from the actual route.
                    allowance=max(15.,distance/.15+turns/.5+5.*len(path))
                    budget=max(budget,float(runtime.data.time)-began+allowance)
                    event=dict(event=name,route=[p.tolist() for p in path],distance_m=distance,
                        turn_radians=turns,budget_seconds=budget,**detail)
                    docking.events.append(event);print(json.dumps(event),flush=True)
                    if name=='receptacle_approach_route':(output/'route.json').write_text(json.dumps(event,indent=2))
                    follower=DockFollower(path,max_speed=.15,max_yaw_speed=.5)
                try:speed,yaw=follower.command(pose)
                except RuntimeError as exc:
                    if str(exc)!='Dock turn drift requires replanning':raise
                    follower=None;base.step(0.,0.);continue
                base.step(speed,yaw)
                if follower.complete and base.stationary():return pose
            raise RuntimeError('Receptacle approach budget exhausted')
        target,estimate=confirm()
        if target is None:
            # A carry may stop facing the furniture beside the receptacle, or
            # with the carried object in front of it. Look around in place
            # before giving up; every turn is swept-checked with the payload.
            T=estimate.T_map_base;yaw0=math.atan2(T[1,0],T[0,0]);pending=look_headings(yaw0);reversed_once=False
            # then a second turn half a step off the first (2026-11-25 x16_grocery call 13: the orange basket only ever
            # showed at the right edge of the fifth view, next to the purple one, and the point landed on the purple)
            second=[yaw0+LOOK_STEP/2+k*LOOK_STEP for k in range(int(math.ceil(2*math.pi/LOOK_STEP-1e-9)))];second_used=False
            while (pending or not second_used) and target is None:
                if not pending:pending=list(second);second_used=True;reversed_once=False
                heading=pending[0]
                def turn(pose,packet):
                    if not plan_to_any_dock(docking,pose,packet.arm.position,[np.r_[pose[:2],heading]]):return None
                    return list(docking.path),dict(view=len(looks)+1)
                if drive(turn,'receptacle_look_turn',seconds/2) is None:
                    # This side sweeps the payload into something observed.
                    # Visit the remaining headings from the other side once.
                    if reversed_once:break
                    reversed_once=True;pending.reverse();continue
                pending.pop(0);looks.append(heading);target,estimate=confirm()
        (output/'observations.json').write_text(json.dumps(observations,indent=2))
        if target is None:raise RuntimeError('Requested receptacle not visually confirmed: '+label)
        def approach(pose,packet):
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
            return docking.path,dict(removed_waypoints=len(original)-len(docking.path))
        drive(approach,'receptacle_approach_route',seconds)
        report=dict(success=True,semantic_target=label,target=target.tolist(),look_headings=looks,
            simulated_seconds=float(runtime.data.time)-start,route_events=docking.events,
            scope='Observed receptacle approach only; no support or placement success certified')
        (output/'report.json').write_text(json.dumps(report,indent=2));return report
