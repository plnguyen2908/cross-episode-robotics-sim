import numpy as np
import pytest
from cross_episode_sim.control.manipulation_approach import ApproachOptions,resolve_mode,opening_frame,front_entry_poses


def plane(z):
    x,y=np.meshgrid(np.linspace(.5,1.,25),np.linspace(-.25,.25,25))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]


def test_accessible_opening_not_receptacle_name():
    shelf=plane(.4)
    walls=np.array([[.5,-.3,.6],[1.,.3,.7]])
    assert resolve_mode('auto',shelf,np.vstack([shelf,walls]))=='top_down'
    assert resolve_mode('auto',shelf,np.vstack([shelf,plane(.75)]))=='front_entry'
    assert resolve_mode('top_down',shelf,plane(.75),observed_aperture=True)=='top_down'
    assert resolve_mode('front_entry',shelf,shelf)=='front_entry'


def test_whole_payload_outside_before_lowering_and_horizontal_entry():
    current=np.eye(4);current[:3,3]=[.2,0.,1.2]
    goal=np.eye(4);goal[:3,3]=[.8,0.,.45]
    shape=np.array([[-.12,-.1,-.04],[.12,.1,.1]])
    anchor=np.array([.55,0.,.4]);normal=np.array([-1.,0.,0.]);options=ApproachOptions()
    before=goal.copy();rows=dict(front_entry_poses(current,goal,shape,anchor,normal,options))
    entry=rows['entry'];world=shape@entry[:3,:3].T+entry[:3,3]
    assert np.min((world-anchor)@normal)>=options.entry_clearance-1e-9
    assert rows['entry'][2,3]==rows['insert'][2,3]
    assert np.allclose(rows['release_pose'],goal) and np.allclose(goal,before)
    assert rows['align'][2,3]==entry[2,3]
    assert np.allclose(rows['align'][:3,:3],goal[:3,:3])
    assert rows['outside'][2,3]==current[2,3]


def test_opening_estimate_uses_observed_support():
    anchor,n=opening_frame(plane(.4))
    assert n[0]<-.9 and abs(anchor[0]-.5)<.03


@pytest.mark.parametrize('kw',[{'entry_height':float('nan')},{'entry_clearance':-.2},{'wrist_yaw_degrees':200},{'approach_mode':'magic'}])
def test_invalid_alignment_parameters(kw):
    with pytest.raises(ValueError):ApproachOptions(**kw)


def test_options_are_per_call_and_do_not_modify_native_schema(tmp_path):
    from tests.test_qwen_household import session
    s=session(tmp_path,'bowl');seen=[]
    def place(**kwargs):
        from cross_episode_sim.skills.m2t2_interface import SkillOutcome
        seen.append(s.approach_options.as_dict());return SkillOutcome(False,'probe')
    s.place=place
    result=s.execute_skill('place',dict(receptacle='open drawer',approach_mode='top_down',entry_height=.06))
    assert not result['success'] and seen[0]['approach_mode']=='top_down' and seen[0]['entry_height']==.06
    assert s.approach_options is None
    native=s.skill_interface().skills.list_skills()
    assert 'approach_mode' not in next(r for r in native if r['name']=='place')['parameters']['properties']


def test_front_withdrawal_exits_at_release_height_before_lifting(monkeypatch):
    import cross_episode_sim.control.manipulation_approach as mod
    from types import SimpleNamespace
    current=np.zeros(7);alignment=np.stack([current,current+.1,current+.2])
    goal=np.eye(4);goal[:3,3]=[.8,0.,.45]
    aligned=goal.copy();aligned[:3,3]=[.35,0.,.49]
    calls=[]
    def translate(planner,q,pose):
        calls.append(pose.copy())
        return np.stack([q,q+.1])
    monkeypatch.setattr(mod,'plan_aligned_translation',translate)
    planner=SimpleNamespace(plan_joint_goal=lambda a,b:np.stack([a,b]))
    path,phases=mod.plan_front_withdrawal(planner,current+.6,goal,aligned,alignment)
    assert np.allclose(calls[0][:3,3],[.35,0.,.45])
    assert len(calls)==1
    assert np.allclose(calls[0][:3,:3],goal[:3,:3])
    assert [p for p,_ in phases]==['horizontal_exit','restore_alignment_outside']
    assert np.allclose(path[-1],current)



def test_front_withdrawal_first_lowers_the_open_hand_onto_the_support(monkeypatch):
    """Live fix55: a bowl released 5 cm above the shelf hung on the open pad."""
    import cross_episode_sim.control.manipulation_approach as mod
    from types import SimpleNamespace
    current=np.zeros(7);alignment=np.stack([current,current+.1,current+.2])
    goal=np.eye(4);goal[:3,3]=[.8,0.,.45]
    aligned=goal.copy();aligned[:3,3]=[.35,0.,.49]
    calls=[]
    def translate(planner,q,pose,blocked=(.05,)):
        calls.append(pose.copy())
        if any(abs(goal[2,3]-pose[2,3]-b)<1e-9 for b in blocked):raise RuntimeError('blocked')
        return np.stack([q,q+.1])
    monkeypatch.setattr(mod,'plan_aligned_translation',translate)
    planner=SimpleNamespace(plan_joint_goal=lambda a,b:np.stack([a,b]))
    path,phases=mod.plan_front_withdrawal(planner,current+.6,goal,aligned,alignment,settle_drop=.05)
    # Full 5 cm drop blocked -> 3 cm drop, then exit at that lowered height.
    assert np.allclose(calls[1][:3,3],[.8,0.,.42])
    assert np.allclose(calls[2][:3,3],[.35,0.,.42])
    assert np.allclose(path[0],current+.6) and np.allclose(path[-1],current)


def test_front_candidates_face_opening_and_keep_supported_contacts():
    from cross_episode_sim.control.manipulation_approach import front_placement_candidates
    shelf=plane(.4);tcp=np.eye(4);tcp[:3,3]=[.2,0.,1.]
    x,y,z=np.meshgrid(np.linspace(-.02,.02,5),np.linspace(-.03,.03,5),np.linspace(0.,.03,3))
    held=np.c_[x.ravel(),y.ravel(),z.ravel()]+tcp[:3,3]
    rows=front_placement_candidates(shelf,shelf,held,tcp,opening_normal=[-1.,0.,0.])
    assert rows
    pitched=[row.get('approach_pitch_degrees',0.) for row in rows]
    # Level entry poses come first; 20-degree downward-pitched ones only after.
    assert pitched[0]==0. and pitched==sorted(pitched,reverse=True) and min(pitched)==-20.
    for row in rows:
        R=np.asarray(row['tcp'])[:3,:3]
        assert np.allclose(R.T@R,np.eye(3)) and np.linalg.det(R)>0.
        if row['approach_pitch_degrees']==0.:
            assert np.allclose(R[:,2],[1.,0.,0.])
        else:
            # Into the opening and 20 degrees down; jaws still close horizontally.
            assert np.allclose(R[:,2],[np.cos(np.radians(20.)),0.,-np.sin(np.radians(20.))])
            assert abs(R[2,1])<1e-9
        assert row['contact'][2]==pytest.approx(.4)
        assert row['learned_proposal'] is False


def test_front_place_footprint_is_inside_observed_opening():
    from cross_episode_sim.control.manipulation_approach import front_placement_candidates
    shelf=plane(.4);tcp=np.eye(4);tcp[:3,3]=[.2,0.,1.]
    x,y,z=np.meshgrid(np.linspace(-.02,.02,5),np.linspace(-.03,.03,5),np.linspace(0.,.03,3))
    held=np.c_[x.ravel(),y.ravel(),z.ravel()]+tcp[:3,3]
    anchor=np.array([.55,0.,.4]);normal=np.array([-1.,0.,0.])
    rows=front_placement_candidates(shelf,shelf,held,tcp,opening_normal=normal,opening_anchor=anchor)
    assert rows
    for row in rows:
        depth=-(np.asarray(row['contact'])-anchor)@normal
        assert depth>=row['support_radius_m']+.03-1e-9


def test_horizontal_pick_reuses_place_frame_without_moving_contact_centers():
    from cross_episode_sim.control.manipulation_approach import front_pick_candidates,horizontal_entry_rotation
    tcp=np.eye(4);tcp[:3,3]=[.6,.02,1.12]
    rows=[dict(tcp=tcp.tolist(),contact=[.6,.02,1.12],confidence=.8,group=0)]
    surface=np.array([[.58,0.,1.07],[.62,.04,1.18]])
    picks=front_pick_candidates(rows,[-1.,0.,0.],surface,1.07)
    assert len(picks)==2
    for row in picks:
        pose=np.array(row['tcp'])
        assert np.allclose(pose[:3,3],tcp[:3,3])
        assert np.allclose(pose[:3,2],[1.,0.,0.])
        assert abs(pose[2,1])<1e-9
        assert np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3))
        assert not row['learned_orientation']
    assert np.allclose(np.array(picks[0]['tcp'])[:3,:3],horizontal_entry_rotation([-1.,0.,0.]))


def test_bad_predicted_rotation_does_not_abort_remaining_grasps():
    from cross_episode_sim.control.m2t2_policy import valid_grasp_poses
    good=np.eye(4);bad=good.copy();bad[0,0]=-1
    scale=good.copy();scale[0,0]=.5
    rows=[dict(tcp=bad.tolist()),dict(tcp=good.tolist()),dict(tcp=scale.tolist())]
    kept,rejected=valid_grasp_poses(rows)
    assert kept==[rows[1]] and [r['index'] for r in rejected]==[0,2]


def test_confined_support_is_below_target_not_denser_upper_shelf():
    from cross_episode_sim.control.manipulation_approach import observed_support_below_target
    target=np.array([.75,0.,1.12])
    lower=plane(1.07)
    upper=np.tile(plane(1.665),(8,1))
    # Narrow object surfaces above the support must not be mistaken for shelves.
    x,y=np.meshgrid(np.linspace(.735,.765,25),np.linspace(-.015,.015,25))
    bottle=np.c_[x.ravel(),y.ravel(),np.full(x.size,1.10)]
    points=np.vstack([lower,upper,bottle])
    assert observed_support_below_target(points,target)==pytest.approx(1.07)
    with pytest.raises(RuntimeError,match='support below target'):
        observed_support_below_target(upper,target)


def test_front_entry_tries_contact_branch_and_restores_planner_state(monkeypatch):
    from types import SimpleNamespace
    import cross_episode_sim.control.manipulation_approach as mod
    initial=[np.full(7,9.)];planner=SimpleNamespace(preferred_ik_seeds=initial)
    contact=SimpleNamespace(iter_solutions=lambda *args:iter([np.ones(7),np.full(7,2.)]))
    tried=[]
    def branch(planner,*args):
        seed=planner.preferred_ik_seeds[0].copy();tried.append(seed)
        if seed[0]==1:raise RuntimeError('branch cannot extend')
        return 'validated'
    monkeypatch.setattr(mod,'_plan_front_entry_branch',branch)
    result=mod.plan_front_entry(planner,np.zeros(7),np.eye(4),np.eye(4),np.zeros((2,3)),np.zeros(3),[-1,0,0],ApproachOptions(),contact)
    assert result=='validated' and len(tried)==2
    assert planner.preferred_ik_seeds is initial


def test_support_break_permits_lift_off_but_not_side_collision_or_downward_motion():
    from types import SimpleNamespace
    from cross_episode_sim.control.observed_payload import ObservedPayload
    from cross_episode_sim.control.manipulation_approach import support_break_clearance
    def pose(q):
        T=np.eye(4);T[2,3]=q[0];return T
    geometry=SimpleNamespace(tcp_pose=pose,fk=lambda q:{0:pose(q)},tcp_body=0,tcp_local=np.eye(4))
    x,y,z=np.meshgrid(np.linspace(-.02,.02,3),np.linspace(-.02,.02,3),np.linspace(.004,.10,3))
    payload=ObservedPayload(geometry,np.c_[x.ravel(),y.ravel(),z.ravel()])
    x,y=np.meshgrid(np.linspace(-.1,.1,15),np.linspace(-.1,.1,15))
    shelf=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    path=np.linspace(0.,.025,20)[:,None]
    assert support_break_clearance(payload,path,shelf,0.)
    assert not support_break_clearance(payload,path,np.vstack([shelf,[.02,0.,.04]]),0.)
    assert not support_break_clearance(payload,np.array([[0.],[-.01],[.025]]),shelf,0.)
    assert not support_break_clearance(payload,path[:2],shelf,0.)



def test_observed_rim_tilt_height_and_contact_preservation():
    from cross_episode_sim.control.manipulation_approach import observed_rim_candidates
    tcp=np.eye(4);tcp[:3,3]=[.7,.1,.8]
    original=tcp.copy()
    row=observed_rim_candidates([{'tcp':tcp,'score':.7}],plane(.95),[-1,0,0],55,-.001,20)[0]
    pose=np.asarray(row['tcp']);R=pose[:3,:3]
    assert np.allclose(R.T@R,np.eye(3)) and np.linalg.det(R)==pytest.approx(1)
    assert np.degrees(np.arccos(-R[2,2]))==pytest.approx(55)
    assert R[0,2]>0  # Approach toward cavity; wrist leans outward.
    assert np.allclose(pose[:2,3],tcp[:2,3]) and pose[2,3]==pytest.approx(.949)
    assert np.allclose(tcp,original) and row['proposal_source']=='observed_tilted_rim'


def test_tilted_pick_enters_high_then_descends_along_tool_axis():
    from cross_episode_sim.control.manipulation_approach import observed_rim_candidates
    current=np.eye(4);current[:3,3]=[.2,0,1.2]
    goal=current.copy();goal[:3,3]=[.8,0,.95]
    goal=np.asarray(observed_rim_candidates([dict(tcp=goal)],plane(.95),[-1,0,0],55,-.001)[0]['tcp'])
    shape=np.array([[-.05,-.08,-.12],[.05,.08,.02]])
    rows=dict(front_entry_poses(current,goal,shape,[.55,0,.95],[-1,0,0],
        ApproachOptions(front_wrist='tilted_rim',entry_height=.04)))
    delta=rows['release_pose'][:3,3]-rows['insert'][:3,3]
    assert np.allclose(delta/np.linalg.norm(delta),goal[:3,2])
    assert rows['insert'][2,3]==pytest.approx(goal[2,3]+.04)
    assert rows['entry'][2,3]==rows['insert'][2,3]
    assert np.allclose(rows['release_pose'],goal)


@pytest.mark.parametrize('kw',[{'wrist_tilt_degrees':66},{'rim_height_offset':-.03},{'wrist_tilt_degrees':float('nan')}])
def test_rim_parameter_limits(kw):
    with pytest.raises(ValueError):ApproachOptions(**kw)


def test_calibration_feedback_boundary():
    from cross_episode_sim.code_policy.capx import validate_feedback_cell
    functions={'pick':None,'redock':None,'finish':None,'get_robot_state':None}
    validate_feedback_cell('print(get_robot_state())\nprint(pick(target="food"))',functions)
    for code in ['pick(target="food")\nfinish()', 'redock(target="food")\npick(target="food")', 'for i in range(2):\n pick(target="food")']:
        with pytest.raises(ValueError,match='one action'):validate_feedback_cell(code,functions)


def test_centered_placement_preserves_qualified_poses_and_excludes_edge_fallback():
    from cross_episode_sim.control.manipulation_approach import rank_centered_contacts
    rows=[dict(contact=[x,0.,.9],tcp=np.eye(4).tolist()) for x in (.55,.6,.61,.85)]
    ranked=rank_centered_contacts(rows,[.6,0.,.9],.04)
    assert [r['contact'][0] for r in ranked]==[.6,.61]
    assert ranked[0]['tcp']==rows[1]['tcp']
    assert rank_centered_contacts(rows,[1.2,0.,.9],.04)==[]


def test_center_estimate_uses_support_extent_not_pointing_pixel():
    from cross_episode_sim.control.manipulation_approach import observed_support_center
    surface=plane(.95)
    center=observed_support_center(surface,[.55,-.1,.95],[-1,0,0])
    assert np.allclose(center,[.75,0,.95],atol=.025)


def test_centered_entry_retries_contact_branch_without_changing_placement_exit(monkeypatch):
    import cross_episode_sim.control.manipulation_approach as mod
    from types import SimpleNamespace
    q=np.zeros(7);attempts=[]
    planner=SimpleNamespace(iter_solutions=lambda *args:iter([q+.1,q+.2]))
    def branch(*args):
        attempts.append(args[0].preferred_ik_seeds[0].copy())
        assert args[-1] is None  # Keep placement's validated empty-hand withdrawal.
        if len(attempts)==1:raise RuntimeError('unreachable branch')
        return 'validated_placement_with_withdrawal'
    monkeypatch.setattr(mod,'_plan_front_entry_branch',branch)
    result=mod.plan_front_entry(planner,q,np.eye(4),np.eye(4),np.zeros((2,3)),np.zeros(3),[-1,0,0],mod.ApproachOptions(),seed_contact_branches=True)
    assert result=='validated_placement_with_withdrawal' and len(attempts)==2
    assert not hasattr(planner,'preferred_ik_seeds')


def test_tilted_placement_preserves_attachment_and_support_contact():
    from cross_episode_sim.control.upright_cookware import upright_candidates
    x,y=np.meshgrid(np.linspace(-.035,.035,12),np.linspace(-.035,.035,12))
    held=np.c_[x.ravel()+.25,y.ravel(),np.ones(x.size)]
    tcp=np.eye(4);tcp[:3,3]=[.25,0.,1.1]
    rows=upright_candidates(plane(.4),plane(.4),held,tcp,tilt_degrees=40,outward=[-1,0,0])
    assert rows
    tool=(held-tcp[:3,3])@tcp[:3,:3]
    for row in rows:
        goal=np.asarray(row['tcp']);placed=tool@goal[:3,:3].T+goal[:3,3]
        assert placed[:,2].min()==pytest.approx(row['contact'][2])
        assert np.degrees(np.arccos(goal[2,2]))==pytest.approx(40)
        assert goal[0,2]<0  # Object upper normal leans outward; not a 90-degree roll.
        assert row['proposal_source']=='observed_plane_tilted'


def test_shallow_axis_does_not_confuse_densely_observed_wall_with_bottom():
    from cross_episode_sim.control.upright_cookware import observed_shallow_axis
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(1)
    # A shallow vessel with many wall samples and sparse bottom samples.
    walls=[]
    for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
        p=rng.uniform([-.08,-.08,0],[.08,.08,.025],size=(800,3));p[:,axis]=sign*.08;walls.append(p)
    p=np.concatenate(walls)
    R=Rotation.from_euler('xy',[15,30],degrees=True).as_matrix()
    normal=observed_shallow_axis(p@R.T+[.5,.1,1.])
    assert normal@(R@np.array([0.,0.,1.]))>.995


def test_insertion_stops_at_the_deepest_supported_point_when_full_depth_is_unreachable(monkeypatch):
    """Oracle release rule: going deeper is a margin, not a requirement."""
    from types import SimpleNamespace
    import cross_episode_sim.control.local_ik as local_ik_module
    import cross_episode_sim.control.contact_retreat as retreat_module
    from cross_episode_sim.control.manipulation_approach import plan_aligned_translation
    # Joint 0 encodes the tool x position; IK fails beyond x=0.55.
    monkeypatch.setattr(retreat_module,'tool_pose',lambda geometry,q:np.vstack([np.c_[np.eye(3),[q[0],0.,.5]],[0,0,0,1]]))
    def ik(geometry,seed,pose,iterations=None):
        if pose[0,3]>.55:return None
        q=np.zeros(7);q[0]=pose[0,3];return q
    monkeypatch.setattr(local_ik_module,'local_ik',ik)
    planner=SimpleNamespace(geometry=None,clear=lambda path,margin=.005:True)
    target=np.eye(4);target[:3,3]=[.7,0.,.5]
    start=np.zeros(7);start[0]=.3
    with pytest.raises(RuntimeError,match='unreachable'):plan_aligned_translation(planner,start,target)
    # Accept only poses at least 0.45 m ahead (object wholly over the shelf there).
    path=plan_aligned_translation(planner,start,target,accept_partial=lambda pose:pose[0,3]>=.45)
    assert .45<=path[-1][0]<=.55
    # No accepted partial pose: the original failure is raised.
    with pytest.raises(RuntimeError,match='unreachable'):
        plan_aligned_translation(planner,start,target,accept_partial=lambda pose:pose[0,3]>=.6)


def test_objects_already_on_the_cabinet_shelf_are_pushable_not_obstacles():
    from cross_episode_sim.control.manipulation_approach import pushable_placed_points
    xs,ys=np.meshgrid(np.arange(.6,1.1,.02),np.arange(-.3,.3,.02))
    shelf=np.c_[xs.ravel(),ys.ravel(),np.full(xs.size,.375)]
    bowl=np.array([[.8,0.,.42],[.82,.05,.45]])           # placed bowl on the shelf
    wall=np.array([[.8,.32,.45],[.8,-.32,.45]])          # side walls
    top=np.array([[.8,0.,.70]]);outside=np.array([[.5,0.,.42]])
    pts=np.concatenate([bowl,wall,top,outside,shelf[:3]])
    mask=pushable_placed_points(pts,shelf,np.array([.6,0.,.5]),np.array([-1.,0.,0.]))
    assert mask.tolist()==[True,True,False,False,False,False,False,False,False]


def test_door_frame_round_trip_and_same_pose_memory():
    from cross_episode_sim.control.manipulation_approach import base_to_aperture,aperture_to_base
    a=dict(hinge=[.6,.4,.5],closed_handle=[.5,-.2,.5],closed_normal=[-1.,0.,0.])
    p=np.array([.9,.1,.375])
    assert np.allclose(aperture_to_base(a,base_to_aperture(a,p)),p)
    # The same door seen from a moved robot: shift+rotate everything.
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('z',30,degrees=True).as_matrix();t=np.array([.2,-.1,0.])
    b={k:(R@np.asarray(v)+(0 if k=='closed_normal' else t)).tolist() for k,v in a.items()}
    assert np.allclose(aperture_to_base(b,base_to_aperture(a,p)),R@p+t)


def test_push_insertion_is_unchecked_once_aligned(monkeypatch):
    """User decision (Nov 16): once aligned outside the opening, just push in."""
    from types import SimpleNamespace
    import cross_episode_sim.control.local_ik as local_ik_module
    import cross_episode_sim.control.contact_retreat as retreat_module
    from cross_episode_sim.control.manipulation_approach import plan_aligned_translation
    monkeypatch.setattr(retreat_module,'tool_pose',lambda geometry,q:np.vstack([np.c_[np.eye(3),[q[0],0.,.5]],[0,0,0,1]]))
    monkeypatch.setattr(local_ik_module,'local_ik',lambda geometry,seed,pose,iterations=None:np.r_[pose[0,3],np.zeros(6)])
    def clear(path,margin=.005):raise AssertionError('push must not run collision checks')
    planner=SimpleNamespace(geometry=None,clear=clear)
    target=np.eye(4);target[:3,3]=[.7,0.,.5];start=np.zeros(7);start[0]=.3
    path=plan_aligned_translation(planner,start,target,accept_partial=lambda pose:pose[0,3]>=.45,push=True)
    assert abs(path[-1][0]-.7)<1e-6



def test_cabinet_frame_strip_beside_the_opening_is_not_a_placed_object():
    import numpy as np
    from cross_episode_sim.control.manipulation_approach import pushable_placed_points
    g=np.mgrid[.50:.80:.01,-.74:-.36:.01].reshape(2,-1).T
    shelf=np.column_stack([g,np.full(len(g),.375)])
    anchor=np.array([.495,-.55,.53]);normal=np.array([-1.,0.,0.])
    zz=np.linspace(.39,.57,60)
    frame=np.concatenate([np.column_stack([np.full(60,x),np.full(60,-.365),zz]) for x in (.52,.53,.54)])   # strip at the side edge
    assert pushable_placed_points(frame,shelf,anchor,normal).sum()==0
    a=np.linspace(0,2*np.pi,300);bowl=np.column_stack([.66+.085*np.cos(a),-.55+.085*np.sin(a),np.full(300,.44)])
    assert pushable_placed_points(bowl,shelf,anchor,normal).sum()>250
