"""Local semantic approach/redock from RGB-D and continuous estimated poses."""
import json
import math
import time
import numpy as np
# Half the spacing of a double door's two handles (~9 cm).
SAME_HANDLE=.045
from PIL import Image
from cross_episode_sim.navigation.shaped_docking import RobotGeometry, ShapedDocking, wrap
from cross_episode_sim.navigation.dock_follower import DockFollower
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.recording.deferred import StateVideoRecorder
from cross_episode_sim.control.sensor_appliance_backend import camera_points, pixel_point
from cross_episode_sim.perception.molmo_pointing import MolmoPointer


# A bowl is wider than it is tall. Recorded bowls measured 0.20 m wide and
# 0.14 m tall at the pickup dock; a green cup the verifier twice named a bowl
# beside a bowl of the same colour measured 0.08 m wide and 0.14 m tall.
TALL_VESSEL_RATIO=1.3


def measured_vessel(category, extent):
    """A named bowl whose measured depth surface is clearly taller than wide is a cup."""
    if category=='bowl' and extent[2]>=TALL_VESSEL_RATIO*max(extent[0],extent[1]):return 'cup'
    return category


# A microwave handle point off the bar handle looks for the bar this far to either side (m; the door is about 0.6 m wide).
MICROWAVE_BAR_REACH=.70
# Longest time one redock motion may take (s).
REDOCK_BUDGET_CEILING=90.

# Leaving a start pose that is inside an observed obstacle's safety margin
# (shared with the navigation undock).
from ..navigation.observed_undock import straight_escape as blocked_start_escape,ESCAPE_LIMIT as BLOCKED_START_LIMIT


# Planning margins (m) tried after the nominal one for a straight back-off.
BACK_OFF_MARGINS=(.02,.01)
# Open floor in front of a fixture: cells within this radius (m) of the pointed spot, at least
# this many, whose mean offset is at least this share of the radius.
OPEN_FLOOR_RADIUS=.9
OPEN_FLOOR_MIN_CELLS=20
OPEN_FLOOR_MIN_STRENGTH=.25

# In-place look-around when a fixture is not in view: this many right turns of this size (rad).
LOOK_AROUND_STEPS=7
LOOK_AROUND_STEP=math.pi/4
LOOK_AROUND_LABEL='look around'

# Bearings (rad, from facing the leaf) of stand spots past an opened door's free
# edge, tried only when no spot facing the leaf is clear.
# Up to 1.4 rad the base still stands on the closing side; this fridge door leaves
# 0.45 m to the wall there and those spots were 7 cm inside the base margin
# (fix105), so spots on the leaf's own line (1.57) and just past it follow.
CLOSING_TIP_BEARINGS=(1.2,-1.2,1.4,-1.4,1.57,-1.57,1.75,-1.75)


# Bearings (rad) around the measured door-front normal tried for the fridge handle stand spot.
FRIDGE_HANDLE_FRONT_OFFSETS=(0.,.15,-.15,.30,-.30)

class SensorRedocking:
    """Retain the arm/gripper posture when its swept shape permits local travel.

    No object truth, reference docking poses, or reference actions are inputs.
    A fresh candidate/point is required after motion; callers invalidate grasps.
    This component's navigation is local visual approach, not exploration or
    replacement of the existing cross-room navigation skill.
    """
    def __init__(self, runtime, sensors, mapper):
        self.runtime, self.sensors, self.mapper = runtime, sensors, mapper
        self.geometry = RobotGeometry(runtime.model)
        self.pointer = MolmoPointer()
        self.grocery_pointers = {}
        self.target_point = None;self.target_label = None

    def observe(self, target):
        self.target_confirmed=False
        packet = self.sensors.capture(include_robot_mask=False)
        tracking = self.mapper.update(packet.cameras['navigation'], packet.timestamp)
        if not tracking.accepted:
            raise RuntimeError('Local redock needs an estimated camera pose')
        if getattr(self,'target_point_base',None) is not None:
            # Seeded from the robot's own memory in its base frame at seed time;
            # placed into this map once, at the first accepted pose.
            T0=tracking.T_map_base;self.target_point=T0[:3,:3]@self.target_point_base+T0[:3,3]
            self.target_point_base=None
        if getattr(self,'trust_memory',False) and self.target_label==target and self.target_point is not None:
            # A door this robot opened itself: its opened-handle position was
            # measured while opening and is reanchored by SLAM. It is often
            # beside the robot, out of view (fix63: 0.8 m to the left), so it
            # is docked to directly instead of requiring a fresh sighting.
            print('REDOCK_REMEMBERED_TARGET',json.dumps(dict(target=target,map_point=np.round(self.target_point,3).tolist())),flush=True)
            self.target_confirmed=True
            return packet, tracking, self.target_point.copy(), 'navigation'
        matches=[]
        for key in ('navigation', 'wrist'):
            camera = packet.cameras[key]
            pointer=self.pointer
            # Ownership vote for appliances only (it stopped a microwave redock
            # docking at a cabinet). For storage cabinets it rejected the big
            # cabinet's correct handle (fix59/fix60 replay), so it is skipped.
            if target.startswith('handle on ') and any(w in target for w in ('microwave','fridge')):
                # Same appliance check the fixture skill applies before grasping:
                # each pointed handle is shown to the VLM and kept only if it
                # belongs to the named appliance. Without it the redock docked at
                # a cabinet/drawer handle under the counter for "handle on microwave".
                from ..perception.fixture_pointing import ApplianceAssociatedPointer
                if not hasattr(self,'appliance_pointers'):self.appliance_pointers={}
                key=(target,id(self.pointer))
                if key not in self.appliance_pointers:
                    self.appliance_pointers[key]=ApplianceAssociatedPointer(self.pointer,target.removeprefix('handle on '))
                pointer=self.appliance_pointers[key]
            if target in ('packaged drink','packaged snack'):
                from ..perception.portable_object_pointing import GroceryPointer
                if target not in self.grocery_pointers:
                    self.grocery_pointers[target]=GroceryPointer(self.pointer,target)
                pointer=self.grocery_pointers[target]
            if target.startswith('handle on ') and 'cabinet' in target and pointer is self.pointer:
                # Same prompt as the cabinet survey. The single-handle prompt
                # returns the left handle of a double door, 9 cm from the
                # surveyed right handle, inside the confirmation radius, so the
                # grasp took the left door the survey had rejected (fix70).
                from ..navigation.cabinet_survey import SURVEY_PROMPT
                result = pointer.remote.call(image=np.asarray(camera.rgb),target=SURVEY_PROMPT) if getattr(pointer,'remote',None) is not None else pointer.point(Image.fromarray(camera.rgb),SURVEY_PROMPT)
            else:
                result = pointer.point(Image.fromarray(camera.rgb), target)
            if target=='handle on fridge':
                # Also ask for the door's vertical handle by name (same prompt
                # as the opening skill), with the same appliance vote.
                from ..perception.fridge_handle import FridgeDoorHandlePointer
                from ..perception.fixture_pointing import ApplianceAssociatedPointer
                if not hasattr(self,'fridge_door_pointers'):self.fridge_door_pointers={}
                if id(self.pointer) not in self.fridge_door_pointers:
                    self.fridge_door_pointers[id(self.pointer)]=ApplianceAssociatedPointer(FridgeDoorHandlePointer(self.pointer),'fridge')
                extra=self.fridge_door_pointers[id(self.pointer)].point(Image.fromarray(camera.rgb),target).get('points',[])
                result=dict(result,points=list(extra)+list(result.get('points',[])))
            if 'category_verification' in result:
                print('GROCERY_CATEGORY_EVIDENCE',json.dumps(dict(camera=key,target=target,
                    proposals=result['category_verification'])),flush=True)
            if result['points']:
                _, _, valid = camera_points(camera)
                T = tracking.T_map_base
                candidates=[]
                for pixel in result['points']:
                    local = pixel_point(camera, valid, pixel)
                    if np.isfinite(local).all():
                        # A point on the robot itself is never the target. The
                        # stowed arm can hide a fixture edge (the microwave's
                        # right edge from its navigation dock) and pointing then
                        # lands on the wrist, 0.13 m from the base.
                        from ..sensors.onboard import without_robot_points
                        if not len(without_robot_points(local[None],camera)):
                            print('REDOCK_POINT_ON_ROBOT',json.dumps(dict(camera=key,target=target,
                                base_point=np.round(local,3).tolist())),flush=True)
                            continue
                        candidates.append(T[:3,:3] @ local + T[:3,3])
                if target=='handle on fridge':
                    # The door handle is the vertical bar. Plain pointing (and
                    # the appliance vote, correctly) also returns the freezer
                    # pull under the door; docking there left the door handle
                    # out of the wrist view and the pull was grasped instead.
                    from ..perception.handle_surface import vertical_bar_handle
                    cloud=camera_points(camera)[0]
                    def bars(pixels):
                        found=[]
                        for pixel in pixels:
                            try:local=pixel_point(camera,valid,pixel)
                            except ValueError:continue
                            if np.isfinite(local).all() and vertical_bar_handle(cloud,local):found.append(T[:3,:3]@local+T[:3,3])
                        return found
                    pointed=len(candidates);candidates=bars(result['points'])
                    self.fridge_front_normal=None
                    if candidates:
                        # Stand square to the door front at the bar, not beside
                        # the fridge: from 0.57 m to the side the bar was seen
                        # edge-on and no grasp was reachable (fix150 call 24 and
                        # its replay), while every opening that worked had stood
                        # in front of the door first (fix143-fix145).
                        from ..perception.handle_surface import front_normal_toward_robot
                        local_bar=T[:3,:3].T@(candidates[0]-T[:3,3])
                        normal=front_normal_toward_robot(cloud,local_bar);source='door front'
                        direction=None if normal is None else T[:3,:3]@normal
                        if direction is None:
                            # Seen from the side the door front is too thin to
                            # fit (replay of the fix150 state); the open floor
                            # in front of the fridge gives the same direction.
                            floor=self.open_floor_direction(candidates[0][:2]);source='open floor'
                            direction=None if floor is None else np.r_[floor,0.]
                        if direction is not None:
                            self.fridge_front_normal=np.asarray(direction,float)
                            print('REDOCK_FRIDGE_HANDLE_FRONT',json.dumps(dict(source=source,normal_map=np.round(self.fridge_front_normal[:2],2).tolist())),flush=True)
                        else:print('REDOCK_FRIDGE_HANDLE_FRONT',json.dumps(dict(source=None)),flush=True)
                    if not candidates:
                        # Pointed on the door but not measurable as a bar from
                        # here: remember a stand direction square to the door
                        # front for the caller's facing step.
                        from ..perception.handle_surface import front_normal_toward_robot
                        for pixel in result['points']:
                            try:local=pixel_point(camera,valid,pixel)
                            except ValueError:continue
                            if not np.isfinite(local).all():continue
                            spot=T[:3,:3]@local+T[:3,3];source='door front'
                            normal=front_normal_toward_robot(cloud,local)
                            direction=None if normal is None else T[:3,:3]@normal
                            if direction is None:
                                floor=self.open_floor_direction(spot[:2]);source='open floor'
                                direction=None if floor is None else np.r_[floor,0.]
                            if direction is not None:
                                self.front_view_hint=(spot,direction)
                                print('REDOCK_FRIDGE_FRONT_HINT',json.dumps(dict(camera=str(key)[:40],point=np.round(local,3).tolist(),
                                    source=source,direction_map=np.round(direction[:2],2).tolist())),flush=True)
                                break
                    print('REDOCK_FRIDGE_DOOR_HANDLE',json.dumps(dict(camera=str(key)[:40],pointed=pointed,vertical_bars=len(candidates),
                        heights=[round(float(p[2]),2) for p in candidates])),flush=True)
                if candidates and target=='handle on microwave':
                    # The handle must be on the microwave the same view shows.
                    # The appliance vote above passed the cabinet-door handle
                    # under the microwave (fix102: that cabinet was opened and
                    # the call counted as an opened microwave).
                    from ..perception.fixture_pointing import appliance_body_points,on_appliance_body
                    bodies=[T[:3,:3]@b+T[:3,3] for b in appliance_body_points(self.pointer,camera,valid,'microwave')]
                    if bodies:
                        kept=[p for p in candidates if on_appliance_body(p,bodies)]
                        print('REDOCK_HANDLE_ON_APPLIANCE',json.dumps(dict(camera=key,target=target,
                            body_heights=[round(float(b[2]),2) for b in bodies],
                            handle_heights=[round(float(p[2]),2) for p in candidates],kept=len(kept))),flush=True)
                        candidates=kept
                        if not candidates:
                            # Nothing pointed was on the microwave: look for the
                            # door's free edge inside the microwave's own enlarged
                            # crop, as the opening skill does before grasping.
                            from ..perception.fixture_contact import FixtureContactPointer
                            from ..runtime.fixture_controls import handle_description
                            if not hasattr(self,'contact_pointers'):self.contact_pointers={}
                            if id(self.pointer) not in self.contact_pointers:
                                self.contact_pointers[id(self.pointer)]=FixtureContactPointer(self.pointer,'microwave')
                            answer=self.contact_pointers[id(self.pointer)].point(Image.fromarray(camera.rgb),handle_description('microwave',True))
                            for pixel in answer.get('points',[]):
                                try:local=pixel_point(camera,valid,pixel)
                                except ValueError:continue
                                p=T[:3,:3]@local+T[:3,3]
                                if np.isfinite(local).all() and on_appliance_body(p,bodies):candidates.append(p)
                            print('REDOCK_MICROWAVE_DOOR_EDGE',json.dumps(dict(camera=key,pointed=len(answer.get('points',[])),
                                on_body=len(candidates),heights=[round(float(p[2]),2) for p in candidates])),flush=True)
                if not candidates and target=='handle on microwave':
                    # Pointing gave nothing usable (fix126, second opening: from
                    # 1.0 m in front of the microwave the only point was on the
                    # robot's own arm; the grasp then ran from 0.85 m away and
                    # the pull was lost at 11 deg). The microwave itself is in
                    # view: take the bar handle measured on its front.
                    from ..perception.fixture_pointing import appliance_body_points,on_appliance_body
                    from ..perception.handle_surface import nearby_vertical_bar,vertical_bar_middle,bar_matches_point
                    from ..sensors.onboard import without_robot_points
                    bodies=[T[:3,:3]@b+T[:3,3] for b in appliance_body_points(self.pointer,camera,valid,'microwave')]
                    cloud=without_robot_points(camera_points(camera)[0],camera);Tinv=np.linalg.inv(T)
                    for body in (bodies if len(cloud) else []):
                        local=Tinv[:3,:3]@body+Tinv[:3,3]
                        bar=nearby_vertical_bar(cloud,local,reach=MICROWAVE_BAR_REACH,
                            accept=lambda seed:bar_matches_point(cloud,seed,float(local[2])))
                        if bar is not None:
                            middle=vertical_bar_middle(cloud,bar);candidates.append(T[:3,:3]@middle+T[:3,3])
                            print('REDOCK_MICROWAVE_BAR_FROM_BODY',json.dumps(dict(camera=key,body=np.round(local,3).tolist(),
                                bar=np.round(middle,3).tolist())),flush=True)
                            break
                if candidates and target=='handle on microwave':
                    # A point that is not on a vertical bar (the door's hinge
                    # edge from the door-edge fallback, isolated test from the
                    # fix123 state: docked 0.57 m beside the handle) moves to
                    # the bar handle on the same microwave front, when one is
                    # measured in this view.
                    from ..perception.handle_surface import vertical_bar_handle,nearby_vertical_bar,vertical_bar_middle,bar_matches_point
                    from ..sensors.onboard import without_robot_points
                    cloud=without_robot_points(camera_points(camera)[0],camera);Tinv=np.linalg.inv(T);moved=[]
                    for p in candidates:
                        local=Tinv[:3,:3]@p+Tinv[:3,3]
                        if len(cloud) and vertical_bar_handle(cloud,local):
                            # On the bar: stand for (and later grasp) the middle
                            # of its whole measured height, not where the pointer
                            # happened to mark it (top end, fix135).
                            from ..perception.handle_surface import bar_middle_point
                            middle=bar_middle_point(cloud,local)
                            if abs(float(middle[2]-local[2]))>.05:
                                print('REDOCK_MICROWAVE_BAR_MIDDLE',json.dumps(dict(camera=key,pointed=np.round(local,3).tolist(),
                                    middle=np.round(middle,3).tolist())),flush=True)
                                p=T[:3,:3]@middle+T[:3,3]
                        elif len(cloud):
                            on_microwave=lambda seed:(bool(bodies) and on_appliance_body(T[:3,:3]@vertical_bar_middle(cloud,seed)+T[:3,3],bodies)
                                and bar_matches_point(cloud,seed,float(local[2])))
                            bar=nearby_vertical_bar(cloud,local,reach=MICROWAVE_BAR_REACH,accept=on_microwave)
                            if bar is not None:
                                from ..perception.handle_surface import bar_middle_point
                                q=T[:3,:3]@bar_middle_point(cloud,bar)+T[:3,3]
                                print('REDOCK_MICROWAVE_BAR_HANDLE',json.dumps(dict(camera=key,pointed=np.round(local,3).tolist(),
                                    bar=np.round(Tinv[:3,:3]@q+Tinv[:3,3],3).tolist())),flush=True)
                                p=q
                            elif getattr(self.runtime,'require_bar_handle',False):
                                # First look only: with no bar measured here the
                                # handle is probably out of this view (navigation
                                # ended beside the microwave, hinge edge in frame,
                                # handle 0.57 m out of it). Report it unseen so the
                                # caller faces the microwave and looks again.
                                print('REDOCK_MICROWAVE_NO_BAR_IN_VIEW',json.dumps(dict(camera=key,pointed=np.round(local,3).tolist())),flush=True)
                                continue
                        moved.append(p)
                    candidates=moved
                rejected=getattr(getattr(self,'runtime',None),'rejected_handles',None)
                if rejected and target.startswith('handle on '):
                    # A cabinet already opened and measured too narrow for
                    # storage is not offered again (20 cm neighbourhood), and
                    # the previous handle target no longer anchors the choice.
                    candidates=[p for p in candidates if all(np.linalg.norm(p[:2]-np.asarray(r)[:2])>.2 for r in rejected)]
                    if self.target_point is not None and any(np.linalg.norm(self.target_point[:2]-np.asarray(r)[:2])<=.2 for r in rejected):
                        self.target_point=None
                if candidates and target.startswith('handle on ') and 'cabinet' in target:
                    # Wall cabinets are not storage targets: a handle pointed
                    # above arm reach is dropped, as if not seen (live fix56
                    # pointed only at a wall-cabinet handle 1.58 m up and spent
                    # the open-door call on unreachable grasps).
                    from ..navigation.hybrid_navigation import HANDLE_HEIGHT
                    T0=tracking.T_map_base
                    high=[p for p in candidates if float((p-T0[:3,3])@T0[:3,2]+T0[2,3])>HANDLE_HEIGHT[1]]
                    if high:print('REDOCK_WALL_CABINET_HANDLE_IGNORED',json.dumps(dict(camera=key,heights=[round(float((p-T0[:3,3])@T0[:3,2]+T0[2,3]),2) for p in high])),flush=True)
                    candidates=[p for p in candidates if float((p-T0[:3,3])@T0[:3,2]+T0[2,3])<=HANDLE_HEIGHT[1]]
                if candidates:
                    if getattr(self,'preserve_target',False) and self.target_label==target:
                        matches.extend((p,key) for p in candidates)
                        continue
                    if self.target_label==target and self.target_point is not None:
                        point=min(candidates,key=lambda p:np.linalg.norm(p-self.target_point))
                    elif target.startswith('handle on '):
                        # Several identical cabinets can be pointed at once and any
                        # one is eligible: take a reachable-height handle, nearest first.
                        from ..navigation.hybrid_navigation import HANDLE_HEIGHT
                        T0=tracking.T_map_base
                        # Keep the pointer's own ranking (its first point is its best
                        # match for the named fixture); only skip handles out of reach.
                        # Taking the nearest instead chose a drawer under a microwave.
                        reachable=[p for p in candidates if HANDLE_HEIGHT[0]<=float((p-T0[:3,3])@T0[:3,2]+T0[2,3])<=HANDLE_HEIGHT[1]]
                        point=(reachable or candidates)[0]
                    else:point=candidates[0]
                    self.target_point=point.copy();self.target_label=target
                    self.target_confirmed=True
                    return packet, tracking, point, key
        if matches:
            point,key=min(matches,key=lambda item:np.linalg.norm(item[0]-self.target_point))
            # The surveyed handle is kept: the nearest fresh point to it, so a
            # double door's other handle (9 cm away) is not substituted. (A
            # room-centre re-choice here was dropped: from the redock pose the
            # two handles' distances to the centre differ by ~6 cm.)
            print('REDOCK_HANDLE_MATCH',json.dumps(dict(target=target,point=np.round(point,3).tolist(),
                gaps=[round(float(np.linalg.norm(m[0]-self.target_point)),3) for m in matches])),flush=True)
            if target.startswith('handle on ') and 'cabinet' in target and getattr(self,'preserve_target',False) and np.linalg.norm(point-self.target_point)>SAME_HANDLE:
                # No fresh point on the surveyed handle: the nearest one is the
                # other door's handle (9 cm away; fix72 switched doors this way).
                # Keep the surveyed handle rather than change doors.
                print('REDOCK_KEPT_SURVEYED_HANDLE',json.dumps(dict(target=target,map_point=np.round(self.target_point,3).tolist(),
                    nearest_fresh_gap=round(float(np.linalg.norm(point-self.target_point)),3))),flush=True)
                point=self.target_point.copy()
            if np.linalg.norm(point-self.target_point)<=.25:
                return self.confirm(packet,tracking,point,key,target,matches)
        if getattr(self,'preserve_target',False) and self.target_label==target and self.target_point is not None:
            # Pointing can miss an instance in one frame. The object navigation
            # observed is still confirmed by its own fresh depth surface, which
            # cannot select another instance.
            T=tracking.T_map_base;local=(self.target_point-T[:3,3])@T[:3,:3]
            print('REDOCK_TARGET_EVIDENCE',json.dumps(dict(target=target,pointed=[dict(camera=key,
                gap_m=float(np.linalg.norm(p-self.target_point))) for p,key in matches])),flush=True)
            for key,camera,support,component,measured in self.tracked_surface(packet,local):
                gap=float(np.min(np.linalg.norm(component-local,axis=1)))
                associated=bool(local[2]>support+.012 and gap<.05)
                print('REDOCK_TRACK_CONFIRMATION',json.dumps(dict(camera=key,target=target,surface_gap_m=gap,
                    observed_support_z=support,object_surface_points=len(component),accepted=associated)),flush=True)
                if associated:
                    return self.confirm(packet,tracking,T[:3,:3]@measured+T[:3,3],key,target,matches)
        if matches:
            raise RuntimeError('Navigation target not confirmed in fresh views; refusing to switch objects')
        raise RuntimeError('Target not visible; use navigate/explore before redocking')

    def confirm(self, packet, tracking, point, key, target, matches):
        """Keep the navigation target unless a close look names it the other vessel.

        Navigation classifies a pointed bowl or cup from metres away, where a
        cup standing beside a bowl is a few pixels wide and can pass as a bowl.
        At the pickup dock the same verifier sees it large. When it now names
        the other vessel, the target never was the requested category: take
        the nearest freshly pointed instance the close look does confirm, or
        refuse. An unclear answer changes nothing.
        """
        from ..perception.vessel_verification import VESSELS
        if target in VESSELS:
            T=tracking.T_map_base
            seen=self.close_look(packet,(point-T[:3,3])@T[:3,:3],key)
            if seen in VESSELS and seen!=target:
                replacement=None
                for other,view in sorted(matches,key=lambda item:np.linalg.norm(item[0]-self.target_point)):
                    if np.linalg.norm(other-point)<.06:continue
                    if self.close_look(packet,(other-T[:3,3])@T[:3,:3],view)==target:
                        replacement=(other,view);break
                print('REDOCK_CATEGORY_CORRECTION',json.dumps(dict(target=target,navigation_target_is=seen,
                    replaced=replacement is not None,shift_m=None if replacement is None else
                    float(np.linalg.norm(replacement[0]-point)))),flush=True)
                if replacement is None:
                    raise RuntimeError(f'Navigation target is a {seen}, not a {target}, and no {target} is confirmed in fresh views')
                point,key=replacement
        self.target_point=point.copy();self.target_confirmed=True
        return packet,tracking,point,key

    def close_look(self, packet, local, key):
        """Category the vessel verifier names for the depth surface at a base-frame point, or None without one.

        The wrist camera is the manipulation view; the confirming camera is the
        fallback. Onboard RGB-D only.
        """
        from ..perception.daaam_grounding import LocalVLM,PROMPT,masked_views,parse_label
        for view,camera,_,component,_ in self.tracked_surface(packet,local,views=tuple(dict.fromkeys(('wrist',key)))):
            optical=(component-camera.T_base_optical[:3,3])@camera.T_base_optical[:3,:3]
            u=np.round(optical[:,0]*camera.K[0,0]/optical[:,2]+camera.K[0,2]).astype(int)
            v=np.round(optical[:,1]*camera.K[1,1]/optical[:,2]+camera.K[1,2]).astype(int)
            mask=np.zeros(camera.depth_m.shape,bool)
            inside=(optical[:,2]>0)&(u>=0)&(u<mask.shape[1])&(v>=0)&(v<mask.shape[0]);mask[v[inside],u[inside]]=True
            if not mask.any():continue
            if getattr(self,'verifier',None) is None:self.verifier=LocalVLM(device='cuda:0')
            answer=self.verifier.label_with_prompt(masked_views(camera.rgb,mask),PROMPT,max_new_tokens=120)
            try:category=parse_label(answer)['category']
            except (ValueError,KeyError):category='unknown'
            extent=np.ptp(component,axis=0)
            named=category;category=measured_vessel(category,extent)
            print('REDOCK_CLOSE_LOOK',json.dumps(dict(camera=view,category=category,named=named,response=answer,
                surface_points=len(component),extent_m=extent.tolist())),flush=True)
            return category
        return None

    def tracked_surface(self, packet, local, views=('navigation', 'wrist')):
        """Fresh above-support depth surface at a previously observed target, per camera."""
        from ..control.observed_grasp_planner import observed_target_mask
        from ..sensors.onboard import without_robot_points
        for key in views:
            camera=packet.cameras[key];cloud,_,_=camera_points(camera)
            cloud=without_robot_points(cloud,camera)
            if not len(cloud):continue
            near=cloud[np.linalg.norm(cloud[:,:2]-local[:2],axis=1)<.28]
            if len(near)<30:continue
            levels,counts=np.unique(np.round(near[:,2]/.005),return_counts=True)
            support=float(levels[counts.argmax()]*.005)
            try:component=cloud[observed_target_mask(cloud,local,support)]
            except RuntimeError:continue
            measured=np.median(component,axis=0)
            # Choose an actual surface sample, not a centroid in empty air.
            measured=component[np.argmin(np.linalg.norm(component-measured,axis=1))]
            yield key,camera,support,component,measured

    def open_floor_direction(self, point_xy, radius=None, minimum=None):
        """Unit direction (map frame) from a point on a fixture front toward the open floor before it, or None.

        Mean offset of the observed, obstacle-free floor cells within `radius`
        of the point. Used when the front's own normal cannot be measured: the
        fridge door seen from beside the microwave shows mostly the fridge's
        side (fix138 state: the points round the pointed handle were not one
        flat face), but the floor in front of it is in the map.
        """
        grid=getattr(self.runtime,'observed_navigation_map',None)
        if grid is None or getattr(grid,'floor_views',None) is None:return None
        radius=OPEN_FLOOR_RADIUS if radius is None else radius;minimum=OPEN_FLOOR_MIN_STRENGTH if minimum is None else minimum
        occupied=(grid.evidence>=2).any(axis=2);free=(grid.floor_views>=grid.config.minimum_floor_views)&~occupied
        rows,cols=np.nonzero(free)
        if not len(rows):return None
        delta=grid.cell_to_world(np.stack([cols,rows],axis=-1))-np.asarray(point_xy,float)[:2]
        distance=np.linalg.norm(delta,axis=1);near=(distance<radius)&(distance>.15)
        if near.sum()<OPEN_FLOOR_MIN_CELLS:return None
        mean=delta[near].mean(0);strength=float(np.linalg.norm(mean))/radius
        if strength<minimum:return None
        return mean/np.linalg.norm(mean)

    def back_off(self, metres, folder):
        """Drive straight back `metres` from where the robot stands (an ordinary checked redock), keeping its heading."""
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        if not tracking.accepted:raise RuntimeError('Back-off needs current estimated pose')
        T=tracking.T_map_base;yaw=math.atan2(T[1,0],T[0,0]);guided=getattr(self.runtime,'qwen_docking',False)
        kept=dict(normal=getattr(self,'approach_normal_map',None),strict=getattr(self,'strict_frontal',False),tiers=getattr(self,'no_closing_tiers',False),
            point=getattr(self,'target_point',None),label=getattr(self,'target_label',None),expected=getattr(self,'expected',None))
        self.target_point=np.r_[T[:2,3]+np.array([math.cos(yaw),math.sin(yaw)]),1.];self.target_label=LOOK_AROUND_LABEL
        self.preserve_target=True;self.trust_memory=True;self.runtime.qwen_docking=False
        self.approach_normal_map=None;self.strict_frontal=True;self.no_closing_tiers=True;self.plain_observe=True
        # Standing 5 cm beside an opened door leaf and 3 deg off its line, a
        # straight 0.4 m reverse drifts 2 cm toward the leaf and into the
        # nominal 3.5 cm margin, so even the back-off had no route (replay of
        # the fix140 state). Like the aperture dock's standing margins, plan
        # this short straight move with the largest smaller margin that works.
        used=None
        try:
            for index,margin in enumerate((None,)+BACK_OFF_MARGINS):
                self.planning_margin=margin
                result=SensorRedocking.move(self,LOOK_AROUND_LABEL,'approach',1.+float(metres),folder if index==0 else folder/f'margin_{index}')
                used=margin
                if result.get('success') or result.get('reason')!='no_observed_collision_clear_local_route':break
        finally:
            self.plain_observe=False;self.planning_margin=None
            self.runtime.qwen_docking=guided;self.trust_memory=False;self.preserve_target=False
            self.target_point=kept['point'];self.target_label=kept['label'];self.target_confirmed=False
            self.approach_normal_map=kept['normal'];self.strict_frontal=kept['strict'];self.no_closing_tiers=kept['tiers']
        print('REDOCK_BACK_OFF',json.dumps(dict(metres=metres,success=bool(result.get('success')),reason=str(result.get('reason'))[:60],
            planning_margin_m=used)),flush=True)
        return result

    def front_view_dock(self, distance, folder):
        """Stand `distance` square in front of the door front remembered by the last look, or None without one."""
        hint=getattr(self,'front_view_hint',None);self.front_view_hint=None
        if hint is None:return None
        point,normal=hint;guided=getattr(self.runtime,'qwen_docking',False)
        kept=dict(strict_frontal=getattr(self,'strict_frontal',False),no_closing_tiers=getattr(self,'no_closing_tiers',False))
        self.target_point=np.asarray(point,float).copy();self.target_label=LOOK_AROUND_LABEL
        self.preserve_target=True;self.trust_memory=True;self.runtime.qwen_docking=False
        self.approach_normal_map=np.asarray(normal,float);self.strict_frontal=True;self.no_closing_tiers=True
        try:result=self.move(LOOK_AROUND_LABEL,'approach',distance,folder)
        finally:
            self.runtime.qwen_docking=guided;self.trust_memory=False;self.preserve_target=False
            self.target_point=None;self.target_label=None;self.target_confirmed=False
            self.approach_normal_map=None;self.strict_frontal=kept['strict_frontal'];self.no_closing_tiers=kept['no_closing_tiers']
        print('REDOCK_FRONT_VIEW',json.dumps(dict(success=bool(result.get('success')),reason=str(result.get('reason'))[:60],
            final_distance_m=result.get('final_distance_m'))),flush=True)
        return result

    def look_around_for(self, target, purpose, distance, folder, steps=None, step=None):
        """Turn in place in steps until `target` comes into view, then dock to it.

        Navigation can end with the fixture behind the robot (fix121: 'found but
        not confirmed', facing the far wall with the microwave out of frame).
        Each step is an ordinary checked redock to a point one metre away in the
        new direction, standing where the robot already stands, so it only
        turns. `folder(name)` gives a fresh output folder per step.
        """
        steps=LOOK_AROUND_STEPS if steps is None else steps;step=LOOK_AROUND_STEP if step is None else step
        for index in range(steps):
            packet=self.sensors.capture(include_robot_mask=False)
            tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
            if not tracking.accepted:raise RuntimeError('Look-around needs current estimated pose')
            T=tracking.T_map_base;yaw=math.atan2(T[1,0],T[0,0])-step     # turn right, as the in-room scan does
            guided=getattr(self.runtime,'qwen_docking',False)
            self.target_point=np.r_[T[:2,3]+np.array([math.cos(yaw),math.sin(yaw)]),1.];self.target_label=LOOK_AROUND_LABEL
            self.preserve_target=True;self.trust_memory=True;self.runtime.qwen_docking=False
            try:turn=self.move(LOOK_AROUND_LABEL,'approach',1.,folder(f'turn_{index}'))
            finally:
                self.runtime.qwen_docking=guided;self.trust_memory=False;self.preserve_target=False
                self.target_point=None;self.target_label=None;self.target_confirmed=False
            print('REDOCK_LOOK_AROUND',json.dumps(dict(target=target,step=index+1,turned=bool(turn.get('success')),
                reason=str(turn.get('reason'))[:60])),flush=True)
            # At the requested distance, not the ranked closest stand-off: from
            # 0.66 m the microwave filled the frame and the handle search went
            # wrong (isolated, fix121 state).
            self.runtime.qwen_docking=False
            try:return self.move(target,purpose,distance,folder(f'view_{index}'))
            except RuntimeError as exc:
                if 'not visible' not in str(exc):raise
            finally:self.runtime.qwen_docking=guided
        raise RuntimeError('Target not visible after a full look around')

    def move(self, target, purpose, distance, output, lateral_offset=None):
        if not isinstance(target, str) or not target.strip():
            raise ValueError('Provide a semantic target')
        if purpose not in ('pick', 'place', 'handle', 'approach'):
            raise ValueError('Unsupported docking purpose')
        distance = (.55 if purpose == 'pick' else .65) if distance is None else float(distance)
        if not np.isfinite(distance) or not .35 <= distance <= 1.5:
            raise ValueError('Local standoff must be 0.35–1.5 m')
        if lateral_offset is not None and (not np.isfinite(lateral_offset) or abs(lateral_offset)>.6):
            raise ValueError('lateral_offset must be within -0.6 to 0.6 m')
        output.mkdir(parents=True, exist_ok=False)
        wall = time.monotonic();start = float(self.runtime.data.time)
        packet, tracking, point, camera = self.observe(target)
        grid = self.runtime.observed_navigation_map
        # Accumulate actual measured depth, without a scanning/tuck cycle.
        if grid.last_timestamp is None or tracking.timestamp > grid.last_timestamp:
            grid.update(packet.cameras['navigation'], tracking)
            grid.update_support(packet.cameras['wrist'], tracking)
        T = tracking.T_map_base
        pose = np.r_[T[:2,3], math.atan2(T[1,0], T[0,0])]
        delta = point[:2] - pose[:2];initial_distance = float(np.linalg.norm(delta))
        bearing = math.atan2(delta[1], delta[0])
        normal=getattr(self,'approach_normal_map',None)
        frontal_ok=(not getattr(self,'strict_frontal',False) or
            (normal is not None and abs(wrap(math.atan2(-normal[1],-normal[0])-pose[2]))<.06))
        if (frontal_ok and purpose=='pick' and getattr(self,'preserve_target',False)
                and .45<=initial_distance<=.65 and abs(wrap(bearing-pose[2]))<.12):
            # No base motion: navigation already supplies a useful dock for
            # this freshly confirmed object. Arm planning remains mandatory.
            result=dict(success=True,reason='retained_navigation_dock',target=target,
                initial_distance_m=initial_distance,final_distance_m=initial_distance,
                simulated_seconds=0.,wall_seconds=time.monotonic()-wall,
                retained_posture=True,pre_dock_arm_ik=False,
                fresh_manipulation_observation_required=True)
            (output/'report.json').write_text(json.dumps(result,indent=2))
            return result
        # Fixture closing can require the opposite side of the swung panel.
        # This normal is supplied only by the observed-door subclass, in the
        # same estimated map frame; ordinary object docking is unchanged.
        normal=getattr(self,'approach_normal_map',None)
        fridge_front=None
        if normal is None and target=='handle on fridge' and purpose=='handle':
            fridge_front=getattr(self,'fridge_front_normal',None);normal=fridge_front
        if normal is not None:
            bearing=math.atan2(-normal[1],-normal[0])
        docking = ShapedDocking(grid, self.geometry, max_nodes=1800)
        if getattr(self,'planning_margin',None) is not None:
            # A back-off beside an opened door may plan with a smaller safety
            # margin (see back_off); the shape check itself always stays on.
            nominal_checker=docking.make_checker;reduced_margin=float(self.planning_margin)
            def reduced_checker(p,q):
                c=nominal_checker(p,q);c.margin=reduced_margin;return c
            docking.make_checker=reduced_checker
        # User decision: black/unknown floor is not a local-motion veto.
        # Preserve the full retained-robot sweep against all observed obstacles;
        # do not paint unknown cells free or apply this to cross-room exploration.
        docking.require_observed_support = False
        checker = docking.make_checker(pose, packet.arm.position)
        offsets=(0.,) if getattr(self,'strict_frontal',False) else (0., .15, -.15, .30, -.30)
        candidates = [np.r_[point[:2] - distance*np.array([math.cos(bearing+a), math.sin(bearing+a)]), bearing+a]
                      for a in offsets]
        if lateral_offset is not None:
            tangent=np.array([-math.sin(bearing),math.cos(bearing)])
            candidates=[np.r_[point[:2]-distance*np.array([math.cos(bearing),math.sin(bearing)])+lateral_offset*tangent,bearing]]
        route = None
        guided = bool(getattr(self.runtime,'qwen_docking',False)) and lateral_offset is None
        escape=None if getattr(self,'escaped_start',False) else blocked_start_escape(checker,pose)
        if escape is not None:
            # Back out of the margin first; the dock search runs again from there.
            print('REDOCK_BLOCKED_START_ESCAPE',json.dumps(dict(target=target,blocked_by=(checker.detail or {}).get('part'),
                back_m=round(float((escape[:2]-pose[:2])@np.array([math.cos(pose[2]),math.sin(pose[2])])),3))),flush=True)
            route=[escape];candidates=[];guided=False
        if guided:
            from ..navigation.qwen_docks import rank_docks,plan_ranked_docks
            # Same observed target, but varying standoff and bearing rather than
            # assuming every fixture can be approached at one fixed radius.
            distances=tuple(dict.fromkeys((distance,.65,.75,.85,1.0)))
            offsets=(0.,) if getattr(self,'strict_frontal',False) else FRIDGE_HANDLE_FRONT_OFFSETS if fridge_front is not None else (0.,.30,-.30,.60,-.60,.90,-.90)
            def dock(r,a):return np.r_[point[:2]-r*np.array([math.cos(bearing+a),math.sin(bearing+a)]),bearing+a]
            feasible=[];rejected=[];nearest=None
            if purpose=='pick':
                # Arm reach binds a pickup. Where furniture blocks the requested
                # radius, take each bearing's closest clear standoff instead of
                # the next 0.1 m ring, and never rank a dock that gives up reach.
                broad=[];radii={}
                for a in offsets:
                    for r in np.arange(distance,1.+1e-9,.025):
                        g=dock(float(r),a);broad.append(g)
                        if checker.clear(g):radii[len(broad)-1]=float(r);break
                        rejected.append(dict(index=len(broad)-1,goal=g.tolist(),detail=checker.detail))
                if radii:
                    nearest=min(radii.values())
                    feasible=[broad[i] for i,r in radii.items() if r<=nearest+.1]
                    if abs(wrap(bearing-pose[2]))<.12 and initial_distance<=nearest+.03:
                        # Already facing the target as close as the observed
                        # obstacles allow: moving could only lose reach.
                        result=dict(success=True,reason='retained_closest_clear_dock',target=target,
                            initial_distance_m=initial_distance,final_distance_m=initial_distance,
                            requested_distance_m=distance,closest_clear_distance_m=nearest,
                            simulated_seconds=0.,wall_seconds=time.monotonic()-wall,
                            retained_posture=True,pre_dock_arm_ik=False,
                            fresh_manipulation_observation_required=True)
                        (output/'endpoint_checks.json').write_text(json.dumps(rejected,indent=2))
                        (output/'report.json').write_text(json.dumps(result,indent=2))
                        return result
            else:
                broad=[dock(r,a) for r in distances for a in offsets]
                for i,g in enumerate(broad):
                    if not checker.clear(g):
                        rejected.append(dict(index=i,goal=g.tolist(),detail=checker.detail));continue
                    feasible.append(g)
                if normal is not None:
                    # Closing side of an opened door with no room to stand
                    # facing the leaf (this fridge door opens 0.56 m from the
                    # side wall): stand past the leaf's free edge instead,
                    # still on the closing side, and reach along the leaf.
                    # First where the oracle stands (facing the leaf as it will
                    # be mid-stroke), then past the leaf's free edge.
                    closing_tiers=[] if getattr(self,'no_closing_tiers',False) else [('CLOSING_DOCK_MID_STROKE',lambda:list(self.mid_stroke_docks(distances)) if hasattr(self,'mid_stroke_docks') else []),
                        ('CLOSING_DOCK_PAST_LEAF_EDGE',lambda:[dock(r,a) for r in distances for a in CLOSING_TIP_BEARINGS])]
                    def next_closing_tier():
                        name,goals=closing_tiers.pop(0);goals=goals();clear=[]
                        for g in goals:
                            broad.append(g)
                            if checker.clear(g):clear.append(g)
                            else:rejected.append(dict(index=len(broad)-1,goal=g.tolist(),detail=checker.detail))
                        print(name,json.dumps(dict(offered=len(goals),clear=len(clear))),flush=True)
                        return clear
                    while not feasible and closing_tiers:feasible=next_closing_tier()
                if purpose=='handle' and feasible:
                    # Arm reach binds a handle grasp as it does a pickup: never
                    # rank a dock that gives up reach for a frontal view. Keep
                    # the clear docks within 10 cm of the closest one.
                    reach=[float(np.linalg.norm(g[:2]-point[:2])) for g in feasible]
                    nearest=min(reach)
                    feasible=[g for g,r in zip(feasible,reach) if r<=nearest+.1]
            from ..navigation.qwen_docks import candidate_image
            candidate_image(grid,pose,broad,[point]*len(broad)).save(output/'all_candidates.png')
            np.savez_compressed(output/'observed_map.npz',origin=grid.origin,resolution=grid.config.resolution,
                occupied=(grid.evidence>=2).any(axis=2),floor=grid.floor_views,pose=pose,candidates=broad,target=point)
            (output/'endpoint_checks.json').write_text(json.dumps(rejected,indent=2))
            candidates,selection=rank_docks(grid,pose,feasible,[point]*len(feasible),packet,output/'qwen',
                purpose+' '+target)
            if plan_ranked_docks(docking,pose,packet.arm.position,candidates):
                checker=docking.checker;route=docking.path
            # Face-on spots can test clear and still have no route: behind the
            # side wall nothing was ever observed, so a spot there has no
            # obstacle evidence (fridge, fix119 replay: three 'clear' spots
            # 0.5 m behind the wall, no route, close abandoned). Go on to the
            # mid-stroke and past-edge spots instead of giving up.
            tier=0
            while route is None and normal is not None and closing_tiers:
                tier+=1;feasible=next_closing_tier()
                if not feasible:continue
                candidate_image(grid,pose,broad,[point]*len(broad)).save(output/'all_candidates.png')
                (output/'endpoint_checks.json').write_text(json.dumps(rejected,indent=2))
                candidates,selection=rank_docks(grid,pose,feasible,[point]*len(feasible),packet,output/f'qwen_tier_{tier}',
                    purpose+' '+target)
                if plan_ranked_docks(docking,pose,packet.arm.position,candidates):
                    checker=docking.checker;route=docking.path
                print('CLOSING_DOCK_TIER_ROUTE',json.dumps(dict(tier=tier,clear=len(feasible),route=route is not None)),flush=True)
        # Prefer a continuous direct approach before lattice/A* fallback.
        for goal in ([] if guided else candidates):
            travel_heading=math.atan2(goal[1]-pose[1],goal[0]-pose[0]) if np.linalg.norm(goal[:2]-pose[:2])>.015 else pose[2]
            if abs(wrap(travel_heading-pose[2]))>math.pi/2:travel_heading=wrap(travel_heading+math.pi)
            direct=[np.r_[pose[:2],travel_heading],np.r_[goal[:2],travel_heading],goal]
            if all(checker.swept(a,b) for a,b in zip([pose]+direct,direct)):
                route = direct;break
        if not guided and route is None and docking.plan_candidates(pose, packet.arm.position, candidates, source='qwen_sensor_redock'):
            checker = docking.checker;route = docking.path
        if route is None:
            result = dict(success=False, reason='no_observed_collision_clear_local_route',
                initial_distance_m=initial_distance, planning_events=docking.events, collision_detail=checker.detail,
                pre_dock_arm_ik=False,fresh_manipulation_observation_required=True,
            unknown_floor_policy='allowed_for_local_redock',observed_obstacle_checks=True)
            (output/'report.json').write_text(json.dumps(result, indent=2));return result
        path_length = sum(np.linalg.norm(b[:2]-a[:2]) for a,b in zip([pose]+route, route))
        turns = sum(abs(wrap(b[2]-a[2])) for a,b in zip([pose]+route, route))
        # Route-dependent speed under calibrated actuator limits; executed at 50 Hz.
        speed = float(np.clip(path_length / max(3., 10.-turns/.5), .12, .35))
        follower = DockFollower(route, max_speed=speed, max_yaw_speed=.6,
                                approach_gain=1.5, recover_turn_drift=True)
        base = OnboardBaseActuator(self.runtime.model, self.runtime.data,
            motion_guard=lambda f,w,dt: abs(f)<=speed+1e-8 and abs(w)<=.6+1e-8 and dt<=.02)
        # The guard authorizes the swept-checked retained posture, not arbitrary travel.
        previous = np.zeros(2); success = False
        budget=float(getattr(self,'motion_budget_seconds',30.))
        if getattr(self,'approach_normal_map',None) is not None:
            budget=max(budget,5.+2.*(path_length/speed+turns/.6))
        # A fixed limit ends a winding route just before it is done: a 1.24 m
        # closing-side route (fix125 and its replays) was 1-2 s short at 30 s
        # and the push then started from an unfinished stand pose; a 0.87 m
        # pick route (replay of the fix124 state) stopped 6 mm from its dock.
        # Time the route with the follower itself, which counts its slowing
        # into every lattice waypoint: 1.5 times that plus 10 s, at most 90 s.
        from ..navigation.dock_follower import route_seconds
        timed=route_seconds([np.asarray(g,float) for g in route],pose,max_speed=speed,max_yaw_speed=.6,
            approach_gain=1.5,limit=REDOCK_BUDGET_CEILING)
        budget=max(budget,min(REDOCK_BUDGET_CEILING,1.5*timed+10.))
        with StateVideoRecorder(self.runtime.model, self.runtime.data, output/'state_recording') as rec:
            for tick in range(int(math.ceil(budget*50))):
                if tick % 5 == 0:
                    packet = self.sensors.capture(include_robot_mask=False)
                    tracking = self.mapper.update(packet.cameras['navigation'], packet.timestamp)
                    rec.record_map(grid, tracking, 'redock' if purpose != 'approach' else 'navigate')
                if not tracking.accepted:
                    base.step(0., 0.);continue
                T = tracking.T_map_base;p = np.r_[T[:2,3], math.atan2(T[1,0], T[0,0])]
                command = np.asarray(follower.command(p))
                # Smooth actual commands, preserving terminal stopping and physics.
                command = previous + np.clip(command-previous, [-.45*.02,-1.2*.02], [.45*.02,1.2*.02])
                base.step(*command);previous = command
                if follower.complete and np.max(np.abs(command)) < .001 and base.stationary():
                    success=True;break
                if tick % 250 == 0:
                    print(json.dumps(dict(stage=purpose, simulated_seconds=float(self.runtime.data.time)-start,
                        target_distance_m=float(np.linalg.norm(point[:2]-p[:2])))), flush=True)
            base.step(0.,0.)
        # Refresh both view and SLAM after motion; no old observation is handed off.
        packet=self.sensors.capture(include_robot_mask=False)
        tracking=self.mapper.update(packet.cameras['navigation'],packet.timestamp)
        grid.update(packet.cameras['navigation'],tracking);grid.update_support(packet.cameras['wrist'],tracking)
        final=float(np.linalg.norm(point[:2]-tracking.T_map_base[:2,3]))
        result=dict(success=success, reason='' if success else 'local_motion_budget',
            initial_distance_m=initial_distance,final_distance_m=final,requested_distance_m=distance,lateral_offset_m=lateral_offset,
            target=target,purpose=purpose,camera=camera,simulated_seconds=float(self.runtime.data.time)-start,
            wall_seconds=time.monotonic()-wall,path_length_m=float(path_length),speed_limit_mps=speed,
            pre_dock_arm_ik=False,fresh_manipulation_observation_required=True,
            unknown_floor_policy='allowed_for_local_redock',observed_obstacle_checks=True,
            retained_posture=True,posture_reason='swept robot geometry checked on observed map',
            duration_exception=None if self.runtime.data.time-start<=15 else 'route turning/settling exceeded target; physical execution retained')
        (output/'report.json').write_text(json.dumps(result,indent=2))
        if escape is not None:
            self.escaped_start=True
            try:return self.move(target,purpose,distance,output/'after_escape',lateral_offset)
            finally:self.escaped_start=False
        return result
