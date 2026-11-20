"""Sensor-geometry approach choices shared by pickup and placement.

An approach is a sequence of target poses, not a collision-free trajectory.
Every segment must still pass the native arm and carried-object planner.
"""
import json
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ApproachOptions:
    approach_mode: str = 'auto'
    front_wrist: str = 'object_level'
    entry_clearance: float = .10
    entry_height: float = .04
    wrist_yaw_degrees: float = 0.
    source_receptacle: str | None = None
    wrist_tilt_degrees: float = 20.
    rim_height_offset: float = -.001
    placement_preference: str = 'nearest'
    object_tilt_degrees: float = 0.
    wrist_roll_degrees: float | None = None

    def __post_init__(self):
        if self.placement_preference not in ('nearest','center'):
            raise ValueError('Unknown placement_preference')
        if self.wrist_roll_degrees is not None and (not np.isfinite(self.wrist_roll_degrees) or abs(self.wrist_roll_degrees)>180):
            raise ValueError('wrist_roll_degrees must be finite within -180 to 180')
        if self.front_wrist not in ('object_level','horizontal','learned','tilted_rim'):
            raise ValueError('Unknown front_wrist')
        if self.approach_mode not in ('auto','top_down','front_entry'):
            raise ValueError('Unknown approach_mode')
        for name,lo,hi in [('entry_clearance',.03,.30),('entry_height',0.,.15),('wrist_yaw_degrees',-90.,90.),('wrist_tilt_degrees',0.,65.),('rim_height_offset',-.02,.03),('object_tilt_degrees',0.,45.)]:
            v=getattr(self,name)
            if isinstance(v,bool) or not np.isfinite(v) or not lo<=v<=hi:
                raise ValueError(f'{name} must be finite and within {lo}–{hi}')
        if self.source_receptacle is not None and (not isinstance(self.source_receptacle,str) or not self.source_receptacle.strip()):
            raise ValueError('source_receptacle must name the visible source')

    def as_dict(self):return asdict(self)


def resolve_mode(requested, surface, scene, *, observed_aperture=False):
    """Recognize an overhead obstruction, not a receptacle-name lookup.

    A remembered front opening takes precedence in auto mode. Otherwise count
    local overhead coverage of the observed support; Qwen can override this
    geometric heuristic. Sparse or occluded reconstruction is not proof of free
    space, and the downstream collision planner remains authoritative.
    """
    if requested!='auto':return requested
    if observed_aperture:return 'front_entry'
    surface=np.asarray(surface);scene=np.asarray(scene)
    if not len(surface) or not len(scene):return 'top_down'
    z=float(np.median(surface[:,2]));over=scene[(scene[:,2]>z+.12)&(scene[:,2]<z+.65)]
    if not len(over):return 'top_down'
    from scipy.spatial import cKDTree
    _,ids=np.unique(np.floor(surface[:,:2]/.04).astype(int),axis=0,return_index=True)
    fraction=np.mean(cKDTree(over[:,:2]).query(surface[ids,:2])[0]<.04)
    return 'front_entry' if fraction>.4 else 'top_down'


def observed_support_below_target(points, target):
    """Nearest broad horizontal RGB-D surface below a confined target.

    A shelf above the object can dominate the depth histogram. It cannot be
    its support. Require a dense, spatially extended level below the sensed
    target; narrow bottle surfaces must not become support planes.
    """
    points=np.asarray(points,float);target=np.asarray(target,float)
    local=points[(np.linalg.norm(points[:,:2]-target[:2],axis=1)<.20)
                 & (points[:,2]<target[2]-.01)]
    if len(local)<25:raise RuntimeError('No observed support below target')
    bins=np.round(local[:,2]/.005).astype(int)
    levels,counts=np.unique(bins,return_counts=True)
    minimum=max(25,int(counts.max()*.15))
    for level,count in sorted(zip(levels,counts),reverse=True):
        if count<minimum:continue
        surface=local[bins==level]
        if np.min(np.ptp(surface[:,:2],axis=0))<.08:continue
        return float(level*.005)
    raise RuntimeError('No broad observed support below target')


def opening_frame(surface, aperture=None):
    """Outward direction and front plane from observed geometry in base frame."""
    surface=np.asarray(surface,float)
    if aperture is not None:
        n=np.asarray(aperture['closed_normal'],float).copy();n[2]=0.;n/=np.linalg.norm(n)
        anchor=(np.asarray(aperture['hinge'])+np.asarray(aperture['closed_handle']))/2
    else:
        center=np.median(surface,axis=0)
        # Rectangular shelf axes; use the axis facing the current base.
        _,_,axes=np.linalg.svd(surface[:,:2]-center[:2],full_matrices=False)
        toward=-center[:2]/max(np.linalg.norm(center[:2]),1e-8)
        n2=max(axes,key=lambda a:abs(a@toward));n2=n2*np.sign(n2@toward)
        n=np.r_[n2,0.];anchor=center.copy()
        anchor+=n*(np.quantile(surface@n,.98)-anchor@n)
    return anchor,n


def front_entry_poses(current, goal, shape_tool, anchor, normal, options):
    """Withdraw outside the opening, align, lower, then insert horizontally.

    The entire measured payload/gripper envelope is outside the front plane
    before lowering. A final short descent precedes release. No target pose is
    executed here and no simulator geometry is consulted.
    """
    current=np.asarray(current,float);goal=np.asarray(goal,float);shape=np.asarray(shape_tool,float)
    n=np.asarray(normal,float);n=n/np.linalg.norm(n);anchor=np.asarray(anchor,float)
    insert=goal.copy();insert[2,3]+=options.entry_height
    if options.front_wrist=='tilted_rim' and options.entry_height>0:
        # Approach a rim along the tool axis, not at floor-level throughout the
        # cavity. Preserve the selected contact pose and its orientation. This
        # is pickup-only; placement keeps its established horizontal insertion.
        downward=-float(goal[2,2])
        if downward<=.1:raise ValueError('Tilted rim entry needs a downward approach axis')
        insert[:3,3]=goal[:3,3]-goal[:3,2]*(options.entry_height/downward)
    entry=insert.copy()
    projections=shape@entry[:3,:3].T@n
    outward=max(0.,options.entry_clearance-float((entry[:3,3]-anchor)@n+projections.min()))
    entry[:3,3]+=outward*n
    outside=current.copy()
    # Reserve room for both current and aligned gripper orientation.
    radius=float(np.linalg.norm(shape,axis=1).max())
    shift=max(0.,options.entry_clearance+radius-float((outside[:3,3]-anchor)@n))
    outside[:3,3]+=shift*n
    # Lower and rotate together outside the opening. Requiring either the final
    # rotation at carry height or the carry rotation at shelf height can make
    # an otherwise reachable alignment impossible.
    aligned=entry.copy()
    shift=max(0.,options.entry_clearance+radius-float((aligned[:3,3]-anchor)@n))
    aligned[:3,3]+=shift*n
    low=entry.copy()
    return [('outside',outside),('align',aligned),('entry',low),('insert',insert),('release_pose',goal.copy())]


def plan_front_alignment(planner,q,current,goal,shape_tool,anchor,normal,options):
    waypoints=front_entry_poses(current,goal,shape_tool,anchor,normal,options)
    aligned=waypoints[1][1]
    # Joint-space motion lowers and rotates together; do not impose a high
    # horizontal translation or low carry-orientation waypoint first.
    try:path=planner.plan(q,aligned[:3,3],aligned[:3,:3])
    except RuntimeError:
        outside=waypoints[0][1]
        first=planner.plan(q,outside[:3,3],outside[:3,:3])
        last=planner.plan(first[-1],aligned[:3,3],aligned[:3,:3])
        path=np.concatenate([first,last[1:]])
    return path,waypoints


PUSH_STEP=.01


def plan_push_insertion(planner,q,target,accept_partial):
    """Oracle-style push: slide in 1 cm steps, release at the deepest step.

    Steps continue while IK holds and the observed world stays clear (no
    extra clearance margin: the 2.5 cm observed boxes already over-cover
    surfaces); the slide stops at the first blocked step. The deepest reached
    step that `accept_partial` accepts (object wholly over the observed
    support, behind the opening) is the release pose; the whole prefix is then
    retimed and swept like any insertion. Placement only.
    """
    from .local_ik import local_ik
    from .contact_retreat import tool_pose
    start=tool_pose(planner.geometry,q)
    if np.linalg.norm(Rotation.from_matrix(target[:3,:3]@start[:3,:3].T).as_rotvec())>.025:
        raise RuntimeError('Insertion requires alignment first')
    distance=float(np.linalg.norm(target[:3,3]-start[:3,3]))
    steps=max(1,int(np.ceil(distance/PUSH_STEP)));joints=np.asarray(q).copy();best=None;stop='full depth'
    for i in range(1,steps+1):
        f=i/steps;pose=target.copy();pose[:3,3]=start[:3,3]+f*(target[:3,3]-start[:3,3])
        nxt=local_ik(planner.geometry,joints,pose)
        if nxt is None:nxt=local_ik(planner.geometry,joints,pose,iterations=100)
        if nxt is None:stop='ik at %.2f'%f;break
        joints=nxt
        if i==steps or accept_partial(pose):best=(f,pose)
    if best is None:raise RuntimeError('Push insertion reached no supported depth ('+stop+')')
    f,pose=best
    path=plan_aligned_translation(planner,q,pose,margin=None)
    print('PUSH_INSERTION',json.dumps(dict(fraction=round(float(f),2),stopped_short_m=round(float(np.linalg.norm(target[:3,3]-pose[:3,3])),3),stop=stop)),flush=True)
    return path


def plan_aligned_translation(planner,q,target,accept_partial=None,margin=.005,push=False):
    """Fixed-orientation Cartesian insertion, retimed as one 50 Hz motion.

    With `accept_partial(pose)`, a translation that becomes unreachable or
    blocked partway stops at the deepest reachable, collision-free point that
    `accept_partial` accepts (the datagen oracle's rule: release at the first
    wholly supported pose; going deeper is a margin, not a requirement).
    """
    if accept_partial is not None and push:
        return plan_push_insertion(planner,q,target,accept_partial)
    if accept_partial is not None:
        try:return plan_aligned_translation(planner,q,target)
        except RuntimeError as exc:
            if 'requires alignment' in str(exc):raise
            full_error=exc
        from .contact_retreat import tool_pose
        start=tool_pose(planner.geometry,q)
        for f in np.arange(.95,.29,-.05):
            pose=target.copy();pose[:3,3]=start[:3,3]+f*(target[:3,3]-start[:3,3])
            if not accept_partial(pose):continue
            try:path=plan_aligned_translation(planner,q,pose)
            except RuntimeError:continue
            print('PARTIAL_INSERTION',json.dumps(dict(fraction=round(float(f),2),stopped_short_m=round(float(np.linalg.norm(target[:3,3]-pose[:3,3])),3),reason=str(full_error)[:120])),flush=True)
            return path
        raise full_error
    from .local_ik import local_ik
    from .contact_retreat import tool_pose
    start=tool_pose(planner.geometry,q)
    if np.linalg.norm(Rotation.from_matrix(target[:3,:3]@start[:3,:3].T).as_rotvec())>.025:
        raise RuntimeError('Insertion requires alignment first')
    distance=np.linalg.norm(target[:3,3]-start[:3,3])
    if distance<1e-6:return np.asarray(q)[None].copy()
    fractions=np.linspace(0.,1.,max(2,int(np.ceil(distance/.005))+1))
    nodes=[np.asarray(q).copy()]
    for f in fractions[1:]:
        pose=target.copy();pose[:3,3]=start[:3,3]+f*(target[:3,3]-start[:3,3])
        next_q=local_ik(planner.geometry,nodes[-1],pose)
        if next_q is None:next_q=local_ik(planner.geometry,nodes[-1],pose,iterations=100)
        if next_q is None:raise RuntimeError('Aligned insertion IK unreachable at fraction '+str(round(float(f),3))+' target '+str(pose[:3,3].tolist()))
        nodes.append(next_q)
    nodes=np.asarray(nodes)
    duration=max(.3,1.875*distance/.15,float(1.875*np.abs(np.diff(nodes,axis=0)).sum(0).max()))
    for _ in range(6):
        t=np.linspace(0.,1.,int(np.ceil(duration/.02))+1);f=10*t**3-15*t**4+6*t**5
        path=np.column_stack([np.interp(f,fractions,nodes[:,j]) for j in range(7)])
        speed=np.max(np.abs(np.diff(path,axis=0)))/.02
        acceleration=np.max(np.abs(np.diff(path,n=2,axis=0)))/.02**2 if len(path)>2 else 0.
        scale=max(speed, np.sqrt(acceleration/2.))
        if scale<=1.01:break
        duration*=scale*1.01
    else:raise RuntimeError('Unable to retime insertion within motion limits')
    # Check commands and the intervening swept joint segments.
    dense=np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.abs(b-a).sum()/.001))+1))
                          for a,b in zip(path[:-1],path[1:])])
    # margin=None: unchecked push after alignment (user decision, Nov 16:
    # once aligned outside the opening, just push in, as the oracle does).
    if margin is not None and (not planner.clear(path,margin=margin) or not planner.clear(dense,margin=margin)):
        raise RuntimeError('Aligned insertion sweep blocked: '+str(getattr(planner,'last_rejection',None)))
    return path


def plan_front_withdrawal(planner,release_q,goal,aligned,alignment_path,settle_drop=0.):
    """Exit at release height before lifting or reversing wrist alignment.

    The alignment point is already outside the observed opening by the full
    gripper/payload envelope. No simulator contact state is used as a trigger.

    settle_drop: the planned gap between the released object and the observed
    support. The open hand first lowers by up to that gap so the support, not
    the open finger pad, carries the object before the hand slides out (live
    fix55: a sideways bowl released 5 cm up hung on the pad and was dragged
    out; doc run 07 'continued riding on the finger'). Lower drops are tried
    if the full one is blocked; zero keeps the old exit.
    """
    start_q=release_q;height=goal[2,3];settle=[]
    for drop in ([settle_drop,.6*settle_drop,.3*settle_drop] if settle_drop>.003 else []):
        lowered=np.asarray(goal,float).copy();lowered[2,3]-=drop
        try:settle=plan_aligned_translation(planner,release_q,lowered)
        except RuntimeError as exc:
            print('WITHDRAWAL_PLAN','settle_rejected',round(float(drop),3),str(exc)[:160],flush=True);continue
        print('WITHDRAWAL_PLAN','settle_open_hand',round(float(drop),3),lowered[:3,3].tolist(),flush=True)
        start_q=settle[-1];height=lowered[2,3];break
    exit_pose=np.asarray(aligned).copy()
    exit_pose[2,3]=height
    exit_pose[:3,:3]=goal[:3,:3]
    print('WITHDRAWAL_PLAN','horizontal_exit',exit_pose[:3,3].tolist(),flush=True)
    horizontal=plan_aligned_translation(planner,start_q,exit_pose)
    if len(settle):horizontal=np.concatenate([settle,horizontal[1:]])
    print('WITHDRAWAL_PLAN','restore_alignment_outside',aligned[:3,3].tolist(),flush=True)
    # The object is now clear: restore the original branch, rather than forcing
    # a straight vertical lift through a different elbow/wrist configuration.
    rejoin=planner.plan_joint_goal(horizontal[-1],alignment_path[-1])
    path=np.concatenate([horizontal,rejoin[1:],alignment_path[::-1][1:]])
    return path,[('horizontal_exit',exit_pose),('restore_alignment_outside',np.asarray(aligned).copy())]


def plan_front_entry(planner, q, current, goal, shape_tool, anchor, normal, options, contact_planner=None, seed_contact_branches=False, retreat_planner=None, release_supported=None, settle_drop=0.):
    exit_options={} if retreat_planner is None else dict(retreat_planner=retreat_planner)
    if settle_drop:exit_options['settle_drop']=settle_drop
    if release_supported is not None:exit_options['release_supported']=release_supported
    if contact_planner is None and not seed_contact_branches:
        return _plan_front_entry_branch(planner,q,current,goal,shape_tool,anchor,normal,options,**exit_options)
    # An independently solved outside pose can land on an elbow branch that
    # cannot extend to the grasp. Seed the SAME alignment/entry routine from
    # collision-checked contact IK branches before executing anything.
    from itertools import islice
    endpoint_planner=contact_planner or planner
    endpoints=list(islice(endpoint_planner.iter_solutions(q,goal[:3,3],goal[:3,:3]),3))
    if not endpoints:raise RuntimeError('No reachable horizontal contact pose at this dock: '+str(getattr(endpoint_planner,'ik_diagnostics',[])[:3]))
    previous=getattr(planner,'preferred_ik_seeds',None);errors=[]
    try:
        for index,endpoint in enumerate(endpoints):
            planner.preferred_ik_seeds=[endpoint]
            print('ENTRY_CONTACT_BRANCH',index,flush=True)
            try:
                return _plan_front_entry_branch(planner,q,current,goal,shape_tool,anchor,normal,options,contact_planner,**exit_options)
            except RuntimeError as exc:errors.append(str(exc))
    finally:
        if previous is None:delattr(planner,'preferred_ik_seeds')
        else:planner.preferred_ik_seeds=previous
    raise RuntimeError('Horizontal entry branches rejected: '+str(errors))


def _plan_front_entry_branch(planner, q, current, goal, shape_tool, anchor, normal, options, contact_planner=None, retreat_planner=None, release_supported=None, settle_drop=0.):
    alignment,waypoints=plan_front_alignment(planner,q,current,goal,shape_tool,anchor,normal,options)
    segments=[alignment];joints=alignment[-1];shift=np.zeros(3)
    for index,(phase,pose) in enumerate(waypoints[2:],2):
        solver=contact_planner if phase in ('insert','release_pose') and contact_planner is not None else planner
        pose=np.asarray(pose,float).copy();pose[:3,3]+=shift
        print('ALIGNMENT_PLAN',phase,pose[:3,3].tolist(),flush=True)
        # Placement only: stop the slide-in where the held object is already
        # wholly over the observed support, if the full depth is not reachable.
        accept=None
        if phase=='release_pose' and contact_planner is None and release_supported is not None:
            try:path=plan_aligned_translation(solver,joints,pose,margin=None)
            except RuntimeError as exc:raise RuntimeError(f'{phase}: {exc}') from exc
            segments.append(path);joints=path[-1];continue
        if phase=='insert' and contact_planner is None and release_supported is not None:
            full=pose[:3,3].copy();accept=lambda candidate:release_supported(candidate[:3,3]-full+shift)
        try:path=plan_aligned_translation(solver,joints,pose,accept_partial=accept,push=accept is not None)
        except RuntimeError as exc:raise RuntimeError(f'{phase}: {exc}') from exc
        if accept is not None:
            from .contact_retreat import tool_pose
            reached=tool_pose(solver.geometry,path[-1])[:3,3]
            shift=shift+(reached-pose[:3,3])
            waypoints[index]=(phase,np.vstack([np.column_stack([pose[:3,:3],reached]),[0,0,0,1]]))
        segments.append(path);joints=path[-1]
    # Loaded entry and empty-hand exit have different contact boundaries.
    approach=np.concatenate([v if i==0 else v[1:] for i,v in enumerate(segments[:-2])])
    insertion=np.concatenate([segments[-2],segments[-1][1:]])
    if contact_planner is None:
        retreat,withdrawal=plan_front_withdrawal(retreat_planner or planner,insertion[-1],goal,waypoints[1][1],alignment,settle_drop)
        waypoints+=withdrawal
    else:
        # Pickup callers plan their separate loaded withdrawal after grasping.
        retreat=insertion[::-1].copy()
    return approach,insertion,retreat,waypoints


def approach_config(value):
    value=dict(value)
    aperture=value.pop('observed_aperture',None)
    return ApproachOptions(**value),aperture


def horizontal_entry_rotation(normal, yaw_degrees=0.):
    """Shared pick/place tool frame: approach into opening, closing line level."""
    z=-np.asarray(normal,float).copy();z[2]=0.
    if not np.isfinite(z).all() or np.linalg.norm(z)<1e-8:
        raise ValueError('Need an observed horizontal opening normal')
    z/=np.linalg.norm(z)
    z=Rotation.from_euler('z',yaw_degrees,degrees=True).apply(z)
    x=np.array([0.,0.,1.]);y=np.cross(z,x)
    return np.column_stack([x,y,z])


def horizontal_reach_probe(geometry, joints, target_base):
    """Kinematic endpoint probe only; never substitutes for swept planning."""
    from .local_ik import local_ik
    from ..robot.profile import MOBILE_FRANKA
    pose=np.eye(4);pose[:3,3]=target_base
    rotation=horizontal_entry_rotation([-1.,0.,0.])
    for roll in (0.,180.):
        pose[:3,:3]=rotation@Rotation.from_euler('z',roll,degrees=True).as_matrix()
        for seed in (joints,np.asarray(MOBILE_FRANKA.travel_posture),geometry.limits.mean(axis=1)):
            result=local_ik(geometry,seed,pose,iterations=100)
            if result is not None:return result
    return None


def support_break_clearance(payload, path, obstacle_points, support_z, padding=.015):
    """Permit initial shelf contact only during monotone vertical lift-off.

    Side/overhead obstacles remain checked throughout. Once lifted, the caller
    checks the complete obstacle cloud for the horizontal withdrawal.
    """
    from .observed_payload import payload_clearance
    points=np.asarray(obstacle_points)
    non_support=points[np.abs(points[:,2]-support_z)>.01]
    poses=[payload.geometry.tcp_pose(q) for q in path]
    first=poses[0];last_z=first[2,3]
    for q,pose in zip(path,poses):
        if np.linalg.norm(pose[:2,3]-first[:2,3])>.003:return False
        if pose[2,3]<last_z-.002:return False
        if np.linalg.norm(Rotation.from_matrix(pose[:3,:3]@first[:3,:3].T).as_rotvec())>.02:return False
        if not payload_clearance(payload,q,non_support,padding=padding,surface_hull=True):return False
        last_z=pose[2,3]
    return payload_clearance(payload,path[-1],points,padding=padding,surface_hull=True)


# A proposal within this height (m) of the object's middle counts as a mid-height grasp.
FRONT_PICK_MIDDLE_BAND=.02
# How far short of the object's measured centre (m, along the entry) the level pinch is offered.
FRONT_PICK_SHORT_OF_CENTRE=(.015,.025,.005)


def front_pick_candidates(candidates, normal, target_surface, support_z, yaw_degrees=0.):
    """Adapt sensed M2T2 contact centers to the shared horizontal entry frame.

    These are explicit geometric adaptations, not untouched learned grasps.
    Endpoint linkage, robot IK, insertion/withdrawal sweeps and physical grip
    verification remain mandatory. Do not rotate a pose after validating it.
    """
    surface=np.asarray(target_surface,float)
    if surface.ndim!=2 or surface.shape[1]!=3 or not len(surface):return []
    rotation=horizontal_entry_rotation(normal,yaw_degrees)
    height=float(np.ptp(surface[:,2]));minimum=support_z+min(.03,height*.25)
    rows=[];seen=set()
    for row in candidates:
        tcp=np.asarray(row['tcp'],float)
        if tcp.shape!=(4,4) or not np.isfinite(tcp).all():continue
        if tcp[2,3]<minimum:continue
        key=tuple(np.floor(tcp[:3,3]/.008).astype(int))
        if key in seen:continue
        seen.add(key)
        for roll in (0.,180.):
            goal=tcp.copy();goal[:3,:3]=rotation@Rotation.from_euler('z',roll,degrees=True).as_matrix()
            rows.append(dict(row,tcp=goal.tolist(),model_tcp=row['tcp'],
                proposal_source='observed_horizontal_entry_adaptation',learned_orientation=False,
                wrist_roll_degrees=roll))
    middle=float((surface[:,2].min()+surface[:,2].max())/2)
    if not any(abs(np.asarray(row['tcp'])[2,3]-middle)<=FRONT_PICK_MIDDLE_BAND for row in rows):
        # The proposals all sit near the object's foot (GraspGen-X on the
        # fridge bottle: every TCP 1-3 cm above the shelf, where the level
        # hand's own body met the shelf edge, fix105). A level entry grasps
        # an upright object at mid-height: offer each proposed contact there.
        raised=[];seen=set()
        # First where the object's own measured shape puts a level pinch: the
        # middle of its width, a little short of its centre along the entry.
        # With the TCP at the centre of this 8 cm bottle its near side reached
        # 4 cm back into the open jaw, onto the finger linkage (fix105 state).
        entry=rotation[:,2];lateral=np.cross([0.,0.,1.],entry)
        band=surface[abs(surface[:,2]-middle)<=FRONT_PICK_MIDDLE_BAND+.005]
        if len(band)>=20 and candidates:
            across=band@lateral;depth=band@entry
            half=float(np.ptp(across))/2;centre=float(depth.min())+half
            template=dict(candidates[0])
            for short in FRONT_PICK_SHORT_OF_CENTRE:
                position=entry*(centre-short)+lateral*float((across.min()+across.max())/2);position[2]=middle
                for roll in (0.,180.):
                    goal=np.eye(4);goal[:3,3]=position
                    goal[:3,:3]=rotation@Rotation.from_euler('z',roll,degrees=True).as_matrix()
                    raised.append(dict(template,tcp=goal.tolist(),model_tcp=template['tcp'],
                        proposal_source='observed_shape_level_pinch_mid_height',learned_orientation=False,
                        wrist_roll_degrees=roll,short_of_centre_m=short))
        for row in candidates:
            tcp=np.asarray(row['tcp'],float)
            if tcp.shape!=(4,4) or not np.isfinite(tcp).all():continue
            key=tuple(np.floor(tcp[:2,3]/.008).astype(int))
            if key in seen:continue
            seen.add(key)
            for roll in (0.,180.):
                goal=tcp.copy();goal[2,3]=middle
                goal[:3,:3]=rotation@Rotation.from_euler('z',roll,degrees=True).as_matrix()
                raised.append(dict(row,tcp=goal.tolist(),model_tcp=row['tcp'],
                    proposal_source='observed_horizontal_entry_adaptation_mid_height',learned_orientation=False,
                    wrist_roll_degrees=roll))
        rows=raised+rows
    rows.sort(key=lambda row:abs(np.asarray(row['tcp'])[2,3]-middle))
    return rows


def front_placement_candidates(surface, scene, held, tcp, yaw_degrees=0., opening_normal=None, opening_anchor=None, wrist_roll_degrees=None, object_tilt_degrees=0., tilt_outward=None):
    from .upright_cookware import upright_candidates
    if opening_normal is not None:
        # Drawer-style front reach: the tool points into the opening. Roll is
        # explicit robot posture, not a claim that a partial bowl wall is its base.
        from .observed_placement import unoccupied_contacts
        from ..perception.grate_support import supported_interior
        rotation=horizontal_entry_rotation(opening_normal,yaw_degrees)
        surface=np.asarray(surface);held=np.asarray(held);tcp=np.asarray(tcp)
        tool=(held-tcp[:3,3])@tcp[:3,:3]
        _,ids=np.unique(np.floor(surface/.025).astype(int),axis=0,return_index=True)
        contacts=surface[ids];rows=[]
        rolls=(0.,180.,90.,-90.) if wrist_roll_degrees is None else (wrist_roll_degrees,)
        # Level entry first. Then the same rolls with the approach pitched 20
        # degrees down (closing line still horizontal): on a low shelf
        # (0.375 m) all 320 level poses folded the elbow into the arm.
        variants=[(roll,0.) for roll in rolls]+[(roll,-20.) for roll in rolls[:2]]
        for group,(roll,pitch) in enumerate(variants):
            R=rotation@Rotation.from_euler('y',pitch,degrees=True).as_matrix()@Rotation.from_euler('z',roll,degrees=True).as_matrix()
            relative=tool@R.T
            bottom=np.r_[(relative[:,:2].min(0)+relative[:,:2].max(0))/2,relative[:,2].min()]
            radius=float(np.linalg.norm(relative[:,:2]-bottom[:2],axis=1).max()+.03)
            valid=supported_interior(surface,contacts,radius)
            if opening_anchor is not None:
                # Keep the observed object's footprint behind the opening plane,
                # not just on a thin observed strip at the shelf's front edge.
                depth=-(contacts-np.asarray(opening_anchor))@np.asarray(opening_normal)
                valid &= depth>=radius+.03
            valid &= unoccupied_contacts(contacts,scene,radius,float(np.ptp(relative[:,2])+.05))
            # Prefer the nearest fully supported spot, not a deep center target.
            selected=contacts[valid];selected=selected[np.argsort(np.linalg.norm(selected[:,:2],axis=1))]
            for point in selected:
                pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=point-bottom
                rows.append(dict(group=group,confidence=0.,contact=point.tolist(),tcp=pose.tolist(),
                    proposal_source='observed_horizontal_front' if not pitch else 'observed_pitched_front',learned_proposal=False,
                    support_radius_m=radius,wrist_roll_degrees=roll,approach_pitch_degrees=pitch))
        return rows
    rows=upright_candidates(surface,scene,held,tcp,tilt_degrees=object_tilt_degrees,outward=tilt_outward)
    if yaw_degrees:
        # Select the closest valid orientation; do not rotate a checked object
        # footprint after its support/occupancy tests.
        yaw=Rotation.from_euler('z',yaw_degrees,degrees=True).as_matrix()
        desired=yaw@np.asarray(tcp)[:3,:3]
        rows.sort(key=lambda r:np.linalg.norm(np.asarray(r['tcp'])[:3,:3]-desired))
    return rows


def gripper_points_tool(geometry,q,tcp):
    import itertools
    corners=np.array(list(itertools.product([-1.,1.],repeat=3)))
    values=[]
    for box in geometry.boxes(q):
        if 'gripper/' in box.name:values.append(corners*box.half@box.axes.T+box.center)
    if not values:raise ValueError('No calibrated gripper envelope')
    return (np.concatenate(values)-tcp[:3,3])@tcp[:3,:3]


def observed_rim_candidates(candidates, surface, outward, tilt_degrees, height_offset, yaw_degrees=0.):
    """Adapt sensed contacts to a raised rim and an outward-leaning wrist.

    No object mesh/pose/annotation is used. These hypotheses are unqualified
    until the existing IK, swept geometry, contact and retention checks pass.
    Tilt is absolute degrees from vertically downward, not an added rotation.
    """
    surface=np.asarray(surface,float);normal=np.asarray(outward,float).copy();normal[2]=0.
    if len(surface)<20 or np.linalg.norm(normal)<1e-8:return []
    normal/=np.linalg.norm(normal)
    axis=np.cross([0.,0.,-1.],-normal)
    lean=Rotation.from_rotvec(axis*np.radians(tilt_degrees)).as_matrix()
    yaw=Rotation.from_euler('z',yaw_degrees,degrees=True).as_matrix()
    top=float(np.quantile(surface[:,2],.98));rows=[]
    for row in candidates:
        pose=np.asarray(row['tcp'],float).copy();closing=pose[:3,1].copy();closing[2]=0.
        if np.linalg.norm(closing)<1e-8:continue
        closing/=np.linalg.norm(closing);down=np.array([0.,0.,-1.])
        pose[:3,:3]=lean@yaw@np.column_stack([np.cross(closing,down),closing,down])
        pose[2,3]=top+height_offset
        rows.append(dict(row,tcp=pose.tolist(),model_tcp=row['tcp'],
            proposal_source='observed_tilted_rim',learned_orientation=False,
            wrist_tilt_degrees=tilt_degrees,rim_height_offset=height_offset,observed_rim_height=top))
    return rows


def rank_centered_contacts(rows, center, tolerance=.04):
    """Preserve qualified poses, selecting contacts near a sensor-derived center."""
    center=np.asarray(center,float)
    ranked=sorted(rows,key=lambda row:np.linalg.norm(np.asarray(row['contact'])[:2]-center[:2]))
    return [dict(row,center_distance_m=float(np.linalg.norm(np.asarray(row['contact'])[:2]-center[:2])))
            for row in ranked if np.linalg.norm(np.asarray(row['contact'])[:2]-center[:2])<=tolerance]


def observed_support_center(surface, hint, outward):
    """Center of measured horizontal support extents, not the pointing pixel."""
    surface=np.asarray(surface,float);hint=np.asarray(hint,float)
    patch=surface[np.abs(surface[:,2]-hint[2])<.008]
    if len(patch)<20:raise ValueError('Insufficient horizontal support for center estimate')
    normal=np.asarray(outward,float)[:2];normal=normal/np.linalg.norm(normal)
    tangent=np.array([-normal[1],normal[0]]);axes=np.column_stack([normal,tangent])
    projected=patch[:,:2]@axes
    center=(np.quantile(projected,.02,axis=0)+np.quantile(projected,.98,axis=0))/2
    return np.r_[axes@center,np.median(patch[:,2])]


def pushable_placed_points(points, surface, anchor, normal, inset=.03, height=.20, behind=.02, side=.04):
    """Observed points of objects resting on a shelf behind the opening plane.

    `surface` is the observed shelf; a point counts when it lies over the
    shelf footprint, behind the opening by `inset`, and between 1 cm and
    `height` above the shelf. Walls beside/above the shelf are outside that
    box, so they remain obstacles.
    """
    points=np.asarray(points,float);surface=np.asarray(surface,float)
    if not len(points) or not len(surface):return np.zeros(len(points),bool)
    shelf=float(np.median(surface[:,2]))
    n=np.asarray(normal,float).copy();n[2]=0.;n/=np.linalg.norm(n)
    across=np.array([-n[1],n[0],0.])
    sd=(surface-np.asarray(anchor))@n;sa=surface@across
    depth=(points-np.asarray(anchor))@n;lateral=points@across
    # `behind` stays 2 cm. Widening it to 20 cm (to catch a bowl sitting past
    # the visible shelf) counted the cabinet's back and side walls on an EMPTY
    # shelf (fix88: 131 points) and sent the first bowl in on the low entry.
    # `side`/`inset`: the cabinet frame beside the opening is a 1 cm strip of
    # points 2-5 cm inside, right at the shelf's side edge. On an empty shelf
    # it gave 131 (fix88 live) and 266 (isolated) "placed object" points.
    return ((depth<=-inset)&(depth>=sd.min()-behind)&(lateral>=sa.min()+side)&(lateral<=sa.max()-side)
            &(points[:,2]>shelf+.01)&(points[:,2]<shelf+height))


def _door_frame(aperture):
    hinge=np.asarray(aperture['hinge'],float);n=np.asarray(aperture['closed_normal'],float).copy();n[2]=0.;n/=np.linalg.norm(n)
    t=np.asarray(aperture['closed_handle'],float)-hinge;t=t-n*(t@n);t[2]=0.;t/=np.linalg.norm(t)
    return hinge,np.column_stack([n,t,[0.,0.,1.]])


def base_to_aperture(aperture, point):
    """Point in the observed door frame (origin hinge; axes outward normal, hinge->handle, up)."""
    hinge,R=_door_frame(aperture);return (R.T@(np.asarray(point,float)-hinge)).tolist()


def aperture_to_base(aperture, local):
    hinge,R=_door_frame(aperture);return R@np.asarray(local,float)+hinge
