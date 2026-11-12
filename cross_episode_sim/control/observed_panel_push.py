"""Push targets from observed panel points and an estimated vertical hinge.

No simulator state or fixture identities enter this component. A candidate is
not permission to move: the caller must validate approach/sweep against observed
geometry and monitor contact and panel motion during execution.
"""
from dataclasses import dataclass
import numpy as np
from .coordinated_contact import panel_contact_pose


@dataclass(frozen=True)
class PanelPushTarget:
    surface: np.ndarray
    direction: np.ndarray
    precontact: np.ndarray
    contact: np.ndarray
    patch: np.ndarray


def preflight_panel_push_step(planner, joints, target, hinge_xy, gripper_vertices,
                              angle, support, orientation_gain=.5):
    """Require a reachable contact to admit an initial coordinated push sweep.

    All geometry is observed; this plans without executing or changing state.
    A contact pose alone does not prove that the arm can follow the panel.
    """
    from .coordinated_contact import panel_arc_geometry
    from .contact_retreat import ContactRetreatPlanner, base_offset_for_tool_motion
    patch, normal, wrist = panel_arc_geometry(
        target.surface, target.direction, np.r_[hinge_xy, 0.], [0., 0., 1.],
        angle, target.contact[:3, :3], orientation_gain)
    goal = panel_contact_pose(patch, normal, wrist, gripper_vertices)
    turn = goal[:3, :3] @ target.contact[:3, :3].T
    yaw = float(np.arctan2(turn[1, 0], turn[0, 0]))
    contact = ContactRetreatPlanner(
        planner.geometry, planner.clear_contact_push, support,
        pose_clear=lambda q, offset, theta: planner.clear_contact_push(q, offset, base_yaw=theta),
        support_pose_clear=support)
    failures = []
    for fraction in (1., .75, .5, .25, 0.):
        theta = yaw * fraction
        offset = base_offset_for_tool_motion(target.contact, goal, theta)
        offset *= min(1., .10 / max(np.linalg.norm(offset), 1e-9))
        if np.linalg.norm(offset) <= .005:
            continue
        try:
            return contact.plan(joints, target.contact, distance=-float(np.linalg.norm(offset)),
                                tcp_end=goal, base_offset_end=offset, base_yaw_end=theta)
        except RuntimeError as exc:
            failures.append(str(exc))
    raise RuntimeError('Panel contact cannot continue into push: ' + str(failures))


def refresh_observed_push_surface(target, points, gripper_vertices_tcp):
    """Refit a newly exposed rear face locally after reaching around an edge.

    A fridge liner can protrude beyond the thin edge visible from the front.
    Keep the contact orientation, but measure the hand-facing surface again.
    An occluded/unsupported patch retains its previously observed target.
    """
    points=np.asarray(points,float);normal=target.direction
    across=np.cross(normal,[0.,0.,1.]);across/=np.linalg.norm(across)
    delta=points-target.surface
    signed=delta@normal
    local=points[(abs(delta@across)<.035)&(abs(delta[:,2])<.035)&
                 (signed>-.045)&(signed<.02)]
    if len(local)<8:return target
    layers=np.floor(((local-target.surface)@normal)/.003).astype(int)
    for layer in np.unique(layers):
        face=local[(layers==layer)|(layers==layer+1)]
        if len(face)<8 or np.ptp(face[:,2])<.01:continue
        # Require a planar patch aligned with the tracked door, rather than
        # selecting a protruding handle or a single nearer depth pixel.
        _,singular,axes=np.linalg.svd(face-face.mean(0),full_matrices=False)
        if singular[1]<.003 or abs(axes[-1]@normal)<.95:continue
        surface=face[np.argmin(np.linalg.norm(face-target.surface,axis=1))]
        contact=panel_contact_pose(surface,normal,target.contact[:3,:3],gripper_vertices_tcp)
        precontact=contact.copy();precontact[:3,3]-=.06*normal
        patch=observed_contact_footprint(points,surface,normal,
            np.asarray(gripper_vertices_tcp)@contact[:3,:3].T+contact[:3,3])
        return PanelPushTarget(surface,normal,precontact,contact,patch)
    return target


def transport_panel_push_target(target,hinge_xy,previous_normal,current_normal):
    """Retain an observed contact patch through self-occlusion, tracking its panel.

    Inputs are sensed normals and estimated hinge geometry in the same base
    frame. Transporting observed points does not fabricate a new contact face.
    """
    from .sensor_hinged import arc_pose
    angle=float(np.arctan2(np.cross(previous_normal,current_normal)[2],np.dot(previous_normal,current_normal)))
    turn=arc_pose(np.eye(4),hinge_xy,angle)
    R=turn[:3,:3];offset=turn[:3,3]
    return PanelPushTarget(target.surface@R.T+offset,R@target.direction,
        turn@target.precontact,turn@target.contact,target.patch@R.T+offset)


def retain_observed_panel_face(previous,observed,hinge_xy,*,layer_tolerance=.012):
    """Keep a tracked face's identity when the camera sees a parallel liner.

    The normal is freshly fitted; an occluded front-face anchor is transported
    from its previous observation, not replaced by a rear-face depth sample.
    This does not add points to the obstacle cloud or remove world geometry.
    """
    from .sensor_hinged import arc_pose,PanelObservation
    angle=float(np.arctan2(np.cross(previous.normal,observed.normal)[2],
                          previous.normal@observed.normal))
    pose=np.eye(4);pose[:3,3]=previous.anchor
    predicted=arc_pose(pose,hinge_xy,angle)[:3,3]
    if abs((observed.anchor-predicted)@observed.normal)<=layer_tolerance:return observed
    return PanelObservation(predicted,observed.normal,observed.inliers,observed.residual)


def track_push_panel(points,previous,hinge_xy,angle,contact_height,*,preserve_face=False):
    """Track the visible panel above a pushing hand, not behind a held handle."""
    from .sensor_hinged import arc_pose,fit_panel
    pose=np.eye(4);pose[:3,3]=previous.anchor
    prediction=arc_pose(pose,hinge_xy,angle)
    normal=prediction[:3,:3]@previous.normal
    near=prediction[:3,3]+.045*normal
    near[2]=contact_height+.25
    if not preserve_face:return fit_panel(points,near,normal,radius=.3)
    # A fixed patch can be hidden by the hand/liner, and a passive panel can
    # travel farther than one commanded increment. Search neighboring observed
    # patches and bounded orientation hypotheses; never synthesize a fit.
    for extra in (0.,5.,-5.,10.,-10.,20.,-20.):
        prediction=arc_pose(pose,hinge_xy,angle+np.deg2rad(extra))
        normal=prediction[:3,:3]@previous.normal
        toward=np.r_[hinge_xy,prediction[2,3]]-prediction[:3,3]
        toward[2]=0.;toward/=np.linalg.norm(toward)
        for shift,height in ((0.,.25),(0.,.4),(.15,.25),(0.,.1),(.15,.4),(.15,.1),(0.,-.05)):
            near=prediction[:3,3]+.045*normal+shift*toward
            near[2]=contact_height+height
            try:observed=fit_panel(points,near,normal,radius=.3)
            except ValueError:continue
            return retain_observed_panel_face(previous,observed,hinge_xy)
    raise ValueError('Observed push panel unavailable in neighboring views/orientations')


def prepare_observed_panel_push(approach_planner, contact_planner, joints, target, hinge_xy,
                                *, arm_only_edge_route=False):
    """Use the oracle preparation routine with RGB-D collision geometry only."""
    from .coordinated_contact import prepare_panel_push
    from .local_ik import local_ik
    from .observed_grasp_planner import smooth_joint_path

    def solve_contact(pose):
        contact_planner.preferred_ik_seeds=[np.array([0.,.25,0.,-1.5,0.,3.35,0.])]
        result=next(contact_planner.iter_solutions(joints,pose[:3,3],pose[:3,:3]),None)
        if result is None:
            raise RuntimeError('Contact IK rejected: '+str(contact_planner.ik_diagnostics[:1]))
        return result

    def check(planner,path):
        if not planner.clear(path):
            raise RuntimeError('Panel transition collision: '+str(planner.last_rejection))

    def solve_precontact(pose,contact_q):
        result=local_ik(approach_planner.geometry,contact_q,pose)
        if result is None:raise RuntimeError('No local precontact IK on selected contact branch')
        check(approach_planner,result)
        return result

    def check_approach(path,pre_q):
        check(contact_planner,path)
        dense=np.linspace(path[0],path[-1],max(2,int(np.ceil(np.sum(abs(path[-1]-path[0]))/.001))+1))
        check(contact_planner,dense)

    def plan_reach(pre_q):
        # Preserve the contact IK branch. Try explicit arm-only clearance
        # waypoints before the expensive global joint-space fallback.
        graph=approach_planner.allow_curobo_fallback
        approach_planner.allow_curobo_fallback=False
        try:
            try:return approach_planner.plan_joint_goal(joints,pre_q)
            except RuntimeError:pass
            current=approach_planner.geometry.tcp_pose(joints)
            radial=target.surface-np.r_[hinge_xy,target.surface[2]]
            radial/=np.linalg.norm(radial)
            for clearance in (.12,.20):
                entry=target.precontact.copy();entry[:3,3]+=clearance*radial
                if arm_only_edge_route:
                    # Leave the front of the panel along its free-edge tangent
                    # before going behind it. A diagonal front-to-back reach
                    # can cut through the panel despite a clear endpoint.
                    corner=current[:3,3]+max(0.,float((entry[:3,3]-current[:3,3])@radial))*radial
                    try:
                        first=approach_planner.plan(joints,corner,current[:3,:3])
                        turn=approach_planner.plan(first[-1],corner,entry[:3,:3])
                        around=approach_planner.plan(turn[-1],entry[:3,3],entry[:3,:3])
                        last=approach_planner.plan_joint_goal(around[-1],pre_q)
                    except RuntimeError as exc:
                        approach_planner.events.append(dict(method='panel_edge_route_rejected',
                            clearance=clearance,reason=str(exc),collision=approach_planner.last_rejection))
                    else:
                        approach_planner.events.append(dict(method='panel_edge_corner',clearance=clearance))
                        return np.concatenate([first,turn[1:],around[1:],last[1:]])
                try:
                    first=approach_planner.plan(joints,entry[:3,3],current[:3,:3])
                    turn=approach_planner.plan(first[-1],entry[:3,3],entry[:3,:3])
                    last=approach_planner.plan_joint_goal(turn[-1],pre_q)
                except RuntimeError:continue
                approach_planner.events.append(dict(method='arm_only_panel_reach',clearance=clearance))
                return np.concatenate([first,turn[1:],last[1:]])
        finally:approach_planner.allow_curobo_fallback=graph
        return approach_planner.plan_joint_goal(joints,pre_q)

    def prepare(solver):
        return prepare_panel_push(target.contact,
        direction=target.direction,
        solve_contact=solver,
        check_contact=lambda q:check(contact_planner,q),
        solve_precontact=solve_precontact,
        plan_contact=smooth_joint_path,
        check_approach=check_approach,
        plan_reach=plan_reach)
    if not arm_only_edge_route:
        return prepare(solve_contact)
    # A feasible contact on one IK branch may have no clear approach. Try the
    # other collision-valid branches before rejecting the observed contact.
    contact_planner.preferred_ik_seeds=[np.array([0.,.25,0.,-1.5,0.,3.35,0.])]
    solutions=sorted(contact_planner.iter_solutions(joints,target.contact[:3,3],target.contact[:3,:3]),
                     key=lambda q:np.linalg.norm(q-joints))
    errors=[];tried=[]
    for solution in solutions:
        if any(np.max(abs(solution-old))<.02 for old in tried):continue
        tried.append(solution)
        try:return prepare(lambda pose:solution)
        except RuntimeError as exc:errors.append(str(exc))
    if not solutions:raise RuntimeError('Contact IK rejected: '+str(contact_planner.ik_diagnostics[:1]))
    raise RuntimeError('Panel reach branches rejected: '+str(errors))



class ObservedPanelPush:
    def candidates(self, points, panel, hinge_xy, handle_position, opening_sign,
                   gripper_vertices_tcp, *, edge_margins=(.14,.10,.06), clearance=.06, oracle_contact_poses=False,
                   preferred_rotation=None, use_observed_edge=False, height_offsets=(0.,)):
        points=np.asarray(points,float);handle=np.asarray(handle_position,float)
        center=np.r_[np.asarray(hinge_xy,float),handle[2]]
        radial=handle-center;radial[2]=0.;radius=np.linalg.norm(radial)
        if radius<.12 or opening_sign not in (-1,1):
            raise ValueError('Insufficient observed hinge leverage or opening direction')
        radial/=radius
        direction=opening_sign*np.cross([0.,0.,1.],radial)
        normal=np.asarray(panel.normal,float)
        if abs(normal@direction)<.9:raise ValueError('Panel and estimated hinge disagree')
        direction=normal*np.sign(normal@direction)
        edge_radius=radius
        if use_observed_edge:
            # A handle is inboard of the free edge. Measure the panel extent
            # itself, as the cabinet push does, using only the sensed plane.
            along=(points-center)@radial
            plane=(abs((points-panel.anchor)@normal)<.012)&(abs(points[:,2]-handle[2])<.14)
            plane &= (along>.12)&(along<radius+.2)
            if plane.sum()<30:raise ValueError('Observed free edge is unavailable')
            edge_radius=float(np.quantile(along[plane],.995))
        # Side contact keeps the wrist outside the free edge and avoids an
        # unreachable face-on wrist reversal on pull-side doors.
        up=np.array([0.,0.,1.])
        pads=np.asarray(gripper_vertices_tcp,float)
        if pads.ndim!=2 or pads.shape[1]!=3 or not len(pads):
            raise ValueError('Missing calibrated closed-gripper contact vertices')
        rotations=[]
        for tool_axis in (-radial, -radial+direction, direction):
            tool_axis=tool_axis/np.linalg.norm(tool_axis)
            rotations.append(np.column_stack((up,np.cross(tool_axis,up),tool_axis)))
        side=rotations[0]
        for angle in (np.pi/4,-np.pi/4):
            co,si=np.cos(angle),np.sin(angle)
            rotations.insert(1,side@np.array([[co,0,si],[0,1,0],[-si,0,co]]))
        if oracle_contact_poses:
            # Transfer the oracle's tilt/roll family, not simulator geometry.
            from scipy.spatial.transform import Rotation
            frontal=np.column_stack((up,np.cross(direction,up),direction))
            rotations=[frontal@Rotation.from_euler('z',roll).as_matrix()@
                       Rotation.from_euler('y',tilt,degrees=True).as_matrix()
                       for roll in (0.,-opening_sign*np.pi/2,opening_sign*np.pi/2)
                       for tilt in (-75.,-70.,-60.,-90.,-45.,90.,60.,0.,-30.,30.,45.)]+rotations
        if preferred_rotation is not None:
            preferred=np.asarray(preferred_rotation,float)
            if preferred.shape!=(3,3) or not np.allclose(preferred.T@preferred,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(preferred),1.,atol=1e-5):
                raise ValueError('Expected a calibrated tool rotation')
            # Try retaining the released handle posture before large wrist
            # reversals. Contact is established by the same closed-finger hull.
            rotations.insert(0,preferred.copy())
        candidates=[]
        from itertools import product
        for height_offset,margin in product(height_offsets,edge_margins):
            wanted=center+max(.08,edge_radius-margin)*radial
            wanted[2]+=height_offset
            delta=points-wanted
            mask=(abs(delta@radial)<.035)&(abs(delta[:,2])<.035)
            mask &= abs((points-panel.anchor)@normal)<.045
            local=points[mask]
            if len(local)<30:continue
            # Use the observed face facing the approaching hand, not an
            # invented opposite surface or a simulator mesh intersection.
            # A newly exposed rear face can be a small fraction of this local
            # patch. Preserve its observed edge instead of selecting the much
            # denser front face and placing the hand through the panel.
            axial=local@direction
            if use_observed_edge:
                # An oblique rear strip may contain <1% of the points but span
                # many image rows. Select the outermost supported depth layer;
                # reject isolated speckle instead of discarding a real edge by
                # its fraction of the much denser front-face sampling.
                bins=np.floor((axial-float(panel.anchor@direction))/.003).astype(int)
                face=None
                for layer in np.unique(bins):
                    observed=local[(bins==layer)|(bins==layer+1)]
                    if len(observed)>=8 and np.ptp(observed[:,2])>=.01:
                        face=observed;break
                if face is None:continue
            else:face=local[axial<np.quantile(axial,.1)+.003]
            surface=face[np.argmin(np.linalg.norm(face-wanted,axis=1))]
            for rotation in rotations:
                contact=panel_contact_pose(surface,direction,rotation,pads)
                precontact=contact.copy();precontact[:3,3]-=clearance*direction
                patch=observed_contact_footprint(points,surface,direction,pads@rotation.T+contact[:3,3])
                candidates.append(PanelPushTarget(surface,direction,precontact,contact,patch))
        if not candidates:raise ValueError('No observed panel patch at reachable handle height')
        return candidates


def observed_contact_footprint(points, surface, normal, gripper_vertices, padding=.006):
    """Observed plane points under the closed-finger projected contact hull.

    The caller retains physical collision and separation from every other
    surface. Protruding handles and geometry behind the plane are not exempt.
    """
    from scipy.spatial import ConvexHull
    normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
    up=np.array([0.,0.,1.]);across=np.cross(normal,up);across/=np.linalg.norm(across)
    axes=np.column_stack((across,up))
    footprint=(np.asarray(gripper_vertices)-surface)@axes
    hull=ConvexHull(footprint)
    delta=np.asarray(points)-surface;uv=delta@axes
    inside=np.all(uv@hull.equations[:,:2].T+hull.equations[:,2]<=padding,axis=1)
    return np.asarray(points)[inside & (abs(delta@normal)<.009)]
