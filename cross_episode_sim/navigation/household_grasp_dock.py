"""Fresh wrist-depth correction for cookware pickup after map-based docking."""
import json,math
import numpy as np
import mujoco
from PIL import Image
from .shaped_docking import RobotGeometry,ShapedDocking
from .dock_follower import DockFollower
from ..control.onboard_base import OnboardBaseActuator
from ..control.sensor_appliance_backend import camera_points,pixel_point
from ..perception.molmo_pointing import MolmoPointer
from ..recording.deferred import StateVideoRecorder
from ..runtime import artifacts


# Fresh wrist-depth distance (m) accepted for a cookware pickup, and the number of looks.
# Base-to-cookware-CENTRE distance (m) accepted for a cookware pickup, and the stand distance aimed at.
# 2026-11-24 (warm_two_pots_000033, Pot_22, 16 cm deep): the old band (.59,.67) was measured to Molmo's pointed
# spot, which lands on the front wall of a deep pot, so the base stood 0.73 m from its centre and ~95% of rim grasps
# had no IK. The v4 oracle stands 0.40-0.46 m from the pot centre for counter picks (Pot_19 and Pot_22).
COOKWARE_PICKUP_BAND=(.42,.52)
COOKWARE_STAND_M=.46
# Farther stands tried, nearest first, only when no stand at COOKWARE_STAND_M is clear (2026-11-24 warm_two_pots_000033
# call 21: a pot on a burner sits behind the stove front; every 0.46 m stand hit it. The oracle stood 0.49-0.59 m there).
COOKWARE_FARTHER_STANDS_M=(.50,.54,.58,.62)
COOKWARE_MAX_EXTENT_M=.35  # a FastSAM mask wider than this in x or y is the counter, not the cookware
COOKWARE_REFINE_LOOKS=4


def cookware_centre(camera,valid,pixel,segmenter):
    """Cookware centre (base frame) from the largest FastSAM mask through the pointed pixel whose observed depth
    extent is cookware sized; the xy middle of its 2-98 percentile extent. None when no mask qualifies."""
    u,v=np.round(pixel).astype(int)
    _,masks=segmenter.segment(camera.rgb[:,:,::-1].copy())
    points,_,_=camera_points(camera);index=np.full(valid.shape,-1);index[valid]=np.arange(int(valid.sum()))
    best=None
    for mask in masks:
        mask=np.asarray(mask,bool)
        if mask.shape!=valid.shape or not mask[v,u]:continue
        P=points[index[mask&valid]]
        if len(P)<30:continue
        lo,hi=np.percentile(P,2,axis=0),np.percentile(P,98,axis=0)
        if (hi[:2]-lo[:2]).max()>COOKWARE_MAX_EXTENT_M:continue
        if best is None or len(P)>best[0]:best=(len(P),np.r_[(lo[:2]+hi[:2])/2,P[:,2].max()],(hi-lo).tolist())
    return None if best is None else (best[1],best[2])


def refine_cookware_view(runtime,source,output):
    """Correct the local distance using RGB-D, not an old map target centroid."""
    output.mkdir(parents=True,exist_ok=False)
    tracker=runtime.mapper.view();geometry=RobotGeometry(runtime.model)
    from .household_map_handoff import bind_observed_map
    docking=ShapedDocking(bind_observed_map(runtime,source),geometry)
    base=OnboardBaseActuator(runtime.model,runtime.data);pointer=MolmoPointer()
    from ..perception.rpc import configured_client,RemoteSegmenter
    segmenter=RemoteSegmenter(configured_client('fastsam'))
    label=json.loads(artifacts.read_text(source/'report.json'))['selected_object']['label']
    start=float(runtime.data.time);events=[];stand_used=None
    with StateVideoRecorder(runtime.model,runtime.data,output/'state_recording') as rec:
        # Four looks, three moves: the last move is measured too (fix184 call 21:
        # 0.666 m, then 0.660 m after the third move was never measured, and
        # the pick ended 'outside the pickup band').
        for attempt in range(COOKWARE_REFINE_LOOKS):
            packet=runtime.sensors.capture()
            estimate=tracker.update(packet.cameras['navigation'],packet.timestamp)
            if not estimate.accepted:raise RuntimeError('Cookware alignment needs localization')
            camera=packet.cameras['wrist'];from ..perception.molmo_pointing import point_cookware_on_burner;answer=point_cookware_on_burner(pointer,Image.fromarray(camera.rgb),label)
            if not answer['points']:
                # 2026-11-25 p5b_pots21 (5 N): after the stand move 'cookware on kitchen counter' found nothing in the
                # wrist view. Plain 'cookware' first; else the centre this look sequence already measured (the pot
                # has not moved, the robot did), placed in this view by the robot's own pose.
                plain=pointer.point(Image.fromarray(camera.rgb),'cookware')
                if plain.get('points'):answer=dict(plain,fallback='cookware')
            T=estimate.T_map_base;_,_,valid=camera_points(camera)
            if answer['points']:
                local=pixel_point(camera,valid,answer['points'][0])
                found=cookware_centre(camera,valid,answer['points'][0],segmenter)
            elif events and events[-1].get('centre_map') is not None:
                Ti=np.linalg.inv(T);local=Ti[:3,:3]@np.asarray(events[-1]['centre_map'],float)+Ti[:3,3];found=None
                print('COOKWARE_KEPT_PREVIOUS_CENTRE',json.dumps(dict(attempt=attempt,local=np.round(local,3).tolist())),flush=True)
            else:raise RuntimeError('Cookware not visible in wrist view')
            target=T[:3,:3]@local+T[:3,3]
            near=local if found is None else found[0]  # no cookware-sized mask: the pointed spot, as before
            centre=T[:3,:3]@near+T[:3,3]
            distance=float(np.linalg.norm(near[:2]));heading=float(np.arctan2(near[1],near[0]))
            events.append(dict(attempt=attempt,local_target=local.tolist(),local_centre=near.tolist(),centre_map=centre.tolist(),
                centre_source='pointed spot' if found is None else 'FastSAM mask depth extent',
                extent=None if found is None else found[1],distance=distance,heading=heading))
            upper=COOKWARE_PICKUP_BAND[1] if stand_used is None else max(COOKWARE_PICKUP_BAND[1],stand_used+.04)
            if COOKWARE_PICKUP_BAND[0]<=distance<=upper and abs(heading)<.06:break
            # last look only: a stand close to the band is handed to the pick, whose grasp checks decide reachability
            # (2026-11-25 f15b_pots33 pot 2: two looks went to counter escapes, the staged stand arrived 9.5 deg off,
            # no move left -> 'outside the pickup band' and no grasp was even tried)
            if attempt==COOKWARE_REFINE_LOOKS-1 and COOKWARE_PICKUP_BAND[0]-.03<=distance<=upper+.05 and abs(heading)<.20:
                events[-1].update(accepted_near_band=True)
                print('COOKWARE_NEAR_BAND',json.dumps(dict(distance=round(distance,3),heading=round(heading,3))),flush=True);break
            if attempt==COOKWARE_REFINE_LOOKS-1:continue
            pose=np.r_[T[:2,3],math.atan2(T[1,0],T[0,0])]
            forward=centre[:2]-pose[:2];forward/=np.linalg.norm(forward)
            first_stand=.54 if 'stove burner' in str(label).lower() else COOKWARE_STAND_M
            goal=np.r_[centre[:2]-first_stand*forward,math.atan2(forward[1],forward[0])]
            checker=docking.make_checker(pose,packet.arm.position)
            turn=(goal[2]-pose[2]+np.pi)%(2*np.pi)-np.pi
            # The pick redock stops about 0.55 m from the target, inside this
            # band, so the base often backs off a few cm next to the counter.
            # fix167: the base stood 7 cm from a counter corner (private), inside
            # the padded footprint, so every plan was 'start_blocked'. Back
            # straight out while the overlap never grows, then measure again.
            from .observed_undock import straight_escape
            escape=None
            if not checker.clear(pose):
                # Straight to the pickup pose itself when it lies straight
                # ahead/behind (fix167 replay: three short escapes to the edge
                # of the margin used every attempt and ended 'outside band').
                if abs(turn)<.08 and checker.clear(goal):
                    count=checker.contact_count(pose);steps=max(1,int(np.ceil(np.linalg.norm(goal[:2]-pose[:2])/.01)))
                    for k in range(1,steps+1):
                        p=np.r_[pose[:2]+(goal[:2]-pose[:2])*k/steps,pose[2]+turn*k/steps];now=checker.contact_count(p)
                        if now>count:break
                        count=now
                    else:escape=goal.copy()
                if escape is None:escape=straight_escape(checker,pose)
            print('COOKWARE_REFINE',attempt,events[-1]['centre_source'],'distance',round(distance,3),'heading',round(heading,3),'start_clear',bool(checker.clear(pose)),flush=True)
            if escape is not None:
                events[-1].update(start_inside_margin=True,escape=escape.tolist())
                print('COOKWARE_REFINE_STRAIGHT_ESCAPE',round(float(np.linalg.norm(escape[:2]-pose[:2])),3),flush=True)
                docking.path=[escape]
            elif abs(turn)<.08 and checker.swept(pose,goal):
                docking.path=[goal];stand_used=first_stand
            else:
                # A counter corner may block a straight advance. Keep the same
                # sensed target and try nearby facing docks on either side.
                bearing0=math.atan2(forward[1],forward[0])
                def stand_candidates(stand):
                    out=[np.r_[centre[:2]-stand*forward,bearing0]]
                    # wider headings too (2026-11-24 warm_two_pots_000033 pot 2: arrived at 45 deg to the counter; only
                    # the square-on stand, 0.8 rad off the bearing, is clear of the counter front, as the oracle stands)
                    for offset in (.15,-.15,.30,-.30,.45,-.45,.60,-.60,.75,-.75,.90,-.90):
                        yaw=bearing0+offset
                        out.append(np.r_[centre[:2]-stand*np.array([math.cos(yaw),math.sin(yaw)]),yaw])
                    return out
                candidates=stand_candidates(COOKWARE_STAND_M)
                # 2026-11-24 (warm_two_pots_000033, Pot_22): at the closer centre stand the base arrives diagonal to the
                # counter; turning to the stand heading AT the stand sweeps a base corner into the counter, and the
                # lattice found no route. Stop short of a clear stand pose, turn there, then drive straight in.
                staged=None
                # a pot ON A BURNER: 0.54 m first (2026-11-24: the checked grasp pass found a grasp from 0.54 m in runs 13-15
                # and refused ~800 from 0.50 m in pots9; oracle 0.49-0.59 m there); the counter keeps 0.46 m first
                on_burner='stove burner' in str(label).lower()
                order=((.54,)+tuple(x for x in (COOKWARE_STAND_M,)+COOKWARE_FARTHER_STANDS_M if x!=.54)) if on_burner else (COOKWARE_STAND_M,)+COOKWARE_FARTHER_STANDS_M
                for stand in order:
                    if staged is not None:break
                    for candidate in stand_candidates(stand):
                        if staged is not None:break
                        if not checker.clear(candidate):continue
                        cturn=(candidate[2]-pose[2]+np.pi)%(2*np.pi)-np.pi
                        if abs(cturn)<.08 and checker.swept(pose,candidate):
                            staged=[np.asarray(candidate,float)];stand_used=stand;break
                        along=np.array([math.cos(candidate[2]),math.sin(candidate[2])])
                        heading_now=np.array([math.cos(pose[2]),math.sin(pose[2])])
                        # back straight out first when turning in place here sweeps a base corner into the counter
                        # (2026-11-24 pot 2: every first turn hit the counter corner 0.3 m ahead-right)
                        for reverse in (0.,.10,.20,.30):
                            if staged is not None:break
                            first=np.r_[pose[:2]-reverse*heading_now,pose[2]]
                            if reverse and not checker.swept(pose,first):break
                            for back in (.15,.25):
                                short=candidate[:2]-back*along;bearing=math.atan2(*(short-first[:2])[::-1])
                                legs=[(first,np.r_[first[:2],bearing]),(np.r_[first[:2],bearing],np.r_[short,bearing]),
                                      (np.r_[short,bearing],np.r_[short,candidate[2]]),(np.r_[short,candidate[2]],candidate)]
                                ok=[bool(checker.swept(a,b)) for a,b in legs]
                                if all(ok):
                                    staged=([first] if reverse else [])+[np.r_[short,candidate[2]],np.asarray(candidate,float)]
                                    stand_used=stand
                                    print('COOKWARE_STAGE_LEGS',json.dumps(dict(stand=stand,yaw=round(float(candidate[2]),3),reverse=reverse,back=back)),flush=True)
                                    break
                if staged is not None:
                    events[-1].update(staged_stand=[x.tolist() for x in staged],stand_m=stand_used)
                    print('COOKWARE_REFINE_STAGED',stand_used,[np.round(x,3).tolist() for x in staged],flush=True)
                    docking.path=staged
                elif not docking.plan_candidates(pose,packet.arm.position,candidates,source='fresh_cookware_depth'):
                    raise RuntimeError('No checked cookware refinement: '+str(docking.events[-2:]))
                else:
                    from .observed_undock import departure_route
                    docking.path=departure_route(pose,docking.path,docking.checker,keep_final_rotation=True)
            follower=DockFollower(docking.path,position_tolerance=.01,max_speed=.12,max_yaw_speed=.4)
            # 15 s as before, longer only for a route that needs it (the staged stand: ~18 s ideal).
            from .dock_follower import tick_budget
            ticks=tick_budget(docking.path,pose,0,15*50,ceiling=60.)
            for tick in range(ticks):
                if tick%5==0:
                    packet=runtime.sensors.capture(include_robot_mask=False)
                    estimate=tracker.update(packet.cameras['navigation'],packet.timestamp)
                    rec.record_map(runtime.observed_navigation_map,estimate,'fresh_cookware_dock')
                if not estimate.accepted:base.step(0.,0.);continue
                P=estimate.T_map_base;p=np.r_[P[:2,3],math.atan2(P[1,0],P[0,0])]
                speed,yaw=follower.command(p);base.step(speed,yaw)
                if follower.complete and base.stationary():break
            else:raise RuntimeError('Cookware refinement motion budget exhausted')
        else:raise RuntimeError('Fresh cookware depth remains outside the pickup band')
    report=json.loads(artifacts.read_text(source/'report.json'))
    report['selected_object']['position']=target.tolist()
    report['fresh_cookware_alignment']=events
    artifacts.write_text(source/'report.json',json.dumps(report))
    tracker.save(source/'slam')
    # Publish the new live handoff after actual base motion. This does not
    # restore or reset physics; capture refuses stale state snapshots.
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(runtime.model,spec))
    mujoco.mj_getState(runtime.model,runtime.data,state,spec)
    artifacts.save_array(source/'private_checkpoint/state.npy',state)
    result=dict(success=True,events=events,simulated_seconds=float(runtime.data.time)-start,
        provenance='fresh wrist RGB-D and continuous estimated SLAM poses')
    (output/'report.json').write_text(json.dumps(result,indent=2));return result
