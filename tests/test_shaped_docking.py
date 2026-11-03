from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
import pytest
from cross_episode_sim.navigation.shaped_docking import RobotBox,RobotGeometry,ObservedShapeChecker,ShapedDocking
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig


def boxes():return [RobotBox('base',np.array([0.,0.,.17]),np.eye(3),np.array([.24,.24,.17])),RobotBox('mast',np.array([-.4,0.,1.]),np.eye(3),np.array([.012,.012,.7]))]
def grid():
    g=ObservedNavigationMap(MapConfig(extent_m=6));g.floor_views[:]=2;return g
def obstacle(g,p):
    c,r=g.world_to_cell(p[:2]);z=int((p[2]-g.config.min_obstacle_height)/g.config.height_resolution);g.evidence[r,c,z]=2


def test_height_and_orientation_are_not_flattened():
    g=grid();obstacle(g,np.array([.4,0,1.]))
    check=ObservedShapeChecker(g,boxes(),np.zeros(3))
    assert check.clear(np.zeros(3))  # Counter-height obstacle does not intersect base.
    assert not check.clear(np.array([0.,0.,np.pi]))  # Turning mast into it is forbidden.
    assert not check.swept(np.zeros(3),np.array([0.,0.,np.pi]))
    obstacle(g,np.array([.2,0,.2]));check.refresh()
    assert not check.clear(np.zeros(3))
    assert check.detail['part']=='base'


def test_unknown_support_stays_unknown():
    g=grid();g.floor_views[:]=0;check=ObservedShapeChecker(g,boxes(),np.zeros(3))
    assert check.clear(np.zeros(3))
    assert not check.clear(np.array([.4,0.,0.]))
    assert check.detail['reason']=='unobserved_base_support'


def test_lattice_route_checks_turns_and_arm_reach():
    g=grid();geometry=NS(boxes=lambda q:boxes(),reach=lambda p,q:np.zeros(7))
    d=ShapedDocking(g,geometry,max_nodes=500);d.reset(NS(position=np.array([1.2,0.,.9])))
    assert d.plan(np.zeros(3),np.zeros(7))
    previous=np.zeros(3)
    for pose in d.path:assert d.checker.swept(previous,pose);previous=pose
    assert .5<=np.linalg.norm(d.goal[:2]-d.target.position[:2])<=.7
    assert d.reach_joints is not None


def test_robot_only_fk_matches_calibrated_robot():
    import mujoco
    source=Path('/nobackup/le/cross-episode-robotics-sim/runs/train_test_v1/test_two_rooms/batches/round_continuous/set_table_000000/action_recording/model.mjb')
    if not source.exists():pytest.skip('qualification robot unavailable')
    m=mujoco.MjModel.from_binary_path(str(source));d=mujoco.MjData(m);geometry=RobotGeometry(m)
    from cross_episode_sim.robot.profile import MOBILE_FRANKA
    q=np.array([.1,.2,-.1,-1.,.2,1.3,.3])
    for name,value in zip(geometry.joint_names,q):d.qpos[m.joint(name).qposadr[0]]=value
    mujoco.mj_kinematics(m,d)
    root=m.body('robot_0/base').id;site=m.site('robot_0/'+MOBILE_FRANKA.tcp_site).id
    R=d.xmat[root].reshape(3,3);T=geometry.fk(q)[geometry.tcp_body]@geometry.tcp_local
    assert np.allclose(T[:3,3],R.T@(d.site_xpos[site]-d.xpos[root]),atol=1e-8)
    assert np.allclose(T[:3,:3],R.T@d.site_xmat[site].reshape(3,3),atol=1e-8)
    assert any('mast_pole' in b.name for b in geometry.boxes(q))


def test_start_collision_cannot_be_a_successful_zero_length_route():
    g=grid();obstacle(g,np.array([0.,0.,.2]))
    d=ShapedDocking(g,NS(boxes=lambda q:boxes(),reach=lambda p,q:np.zeros(7)))
    d.reset(NS(position=np.array([.6,0.,.9])))
    assert not d.plan(np.zeros(3),np.zeros(7))
    assert d.events[-1]['event']=='shape_start_blocked'


def test_final_facing_is_continuous_and_centers_observed_target():
    g=grid();geometry=NS(boxes=lambda q:boxes(),reach=lambda p,q:np.zeros(7))
    d=ShapedDocking(g,geometry,max_nodes=500)
    d.reset(NS(position=np.array([1.2,.25,.9])))
    assert d.plan(np.zeros(3),np.zeros(7))
    relative=d.target.position[:2]-d.goal[:2]
    assert abs(np.arctan2(relative[1],relative[0])-d.goal[2])<1e-9
    assert abs(d.goal[2]/(np.pi/4)-round(d.goal[2]/(np.pi/4)))>.01
    assert d.checker.swept(d.path[-2],d.path[-1])
    assert d.events[-1]['continuous_facing']


def test_initial_support_prior_is_stable_under_translated_rotated_coordinates():
    g=grid();g.floor_views[:]=0;pose=np.array([.57,-.93,.37])
    check=ObservedShapeChecker(g,boxes(),pose)
    assert check.clear(pose)
    assert not check.clear(pose+np.array([.1,0.,0.]))


def test_forward_curve_preserves_endpoints_and_clearance():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    start=np.array([0.,0.,0.]);d.goal=np.array([.5,.2,0.]);d.path=[d.goal.copy()]
    d.checker=ObservedShapeChecker(g,boxes(),start);d.smooth_docking(start)
    assert d.curve is not None
    assert np.allclose(d.curve[0],start) and np.allclose(d.curve[-1],d.goal)
    assert all(d.checker.swept(a,b) for a,b in zip(d.curve[:-1],d.curve[1:]))
    assert d.curve_command(start)[0]>0
    assert np.allclose(d.curve_command(d.goal),0) and not d.path


def test_curve_rejection_preserves_original_route():
    d=ShapedDocking(grid(),None);start=np.zeros(3);d.goal=np.array([.5,.2,0.]);d.path=[d.goal.copy()]
    d.checker=NS(swept=lambda a,b:False);d.smooth_docking(start)
    assert d.curve is None and len(d.path)==1
    assert d.events[-1]['event']=='smoothing_rejected'


def test_rolling_advance_requires_observed_support_and_stays_short():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([1.5,0.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    assert d.approaching and np.linalg.norm(d.goal[:2])<=.25
    g.floor_views[:]=0
    d.reset(NS(position=np.array([1.5,0.,.9])))
    assert not d.plan_advance(np.zeros(3),np.zeros(7))
    assert d.events[-1]['detail']['reason']=='unobserved_base_support'


def test_rolling_advance_does_not_replace_close_precision_docking():
    d=ShapedDocking(grid(),NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([.6,0.,.9])))
    assert not d.plan_advance(np.zeros(3),np.zeros(7))


def test_reset_clears_previous_rolling_curve():
    d=ShapedDocking(grid(),NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([1.5,0.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    d.reset(NS(position=np.array([.6,0.,.9])))
    assert d.path is None and d.curve is None and not d.approaching


def test_curve_terminal_residual_faces_object_instead_of_turning_sideways():
    d=ShapedDocking(grid(),NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([1.1,0.,.9])))
    d.goal=np.array([.5,0.,0.]);d.path=[d.goal.copy()]
    d.checker=ObservedShapeChecker(d.grid,boxes(),np.zeros(3));d.smooth_docking(np.zeros(3))
    command=d.curve_command(np.array([.5,.016,0.]))
    assert command[0]==0 and abs(command[1])<.15
    assert d.path  # Still needs the final facing correction.


def test_initial_support_uses_map_cells_but_does_not_clear_obstacles():
    g=grid();g.floor_views[:]=0;pose=np.array([.28,-.005,-.002])
    check=ObservedShapeChecker(g,boxes(),pose)
    assert check.clear(pose+np.array([.001,-.0004,.0002]))
    assert not check.clear(pose+np.array([.2,0.,0.]))
    obstacle(g,np.r_[pose[:2],.2]);check.refresh()
    assert not check.clear(pose)
    assert check.detail['reason']=='observed_3d_collision'


def test_turn_rate_reserves_sensor_latency_and_braking_angle():
    from cross_episode_sim.navigation.shaped_docking import turn_rate
    for error in [-1.,-.2,-.01,0.,.01,.2,1.]:
        rate=turn_rate(error)
        assert abs(rate)*.25+rate**2/(2*.8)<=abs(error)+1e-12
        assert rate*error>=0


def test_centimetre_connector_does_not_create_a_curved_loop():
    d=ShapedDocking(grid(),None);d.goal=np.array([.05,.03,.3]);d.path=[d.goal.copy()]
    d.smooth_docking(np.zeros(3))
    assert d.curve is None and len(d.path)==1


def test_observed_retreat_is_bounded_and_collision_checked():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([.7,0.,.9])))
    pose=np.zeros(3)
    assert d.plan_retreat(pose,np.zeros(7))
    assert d.goal[0]<0 and d.checker.swept(pose,d.goal)
    assert d.plan_retreat(pose,np.zeros(7))
    assert not d.plan_retreat(pose,np.zeros(7))
    d.reset(NS(position=np.array([.7,0.,.9])))
    obstacle(g,np.array([-.3,0.,.2]))
    assert not d.plan_retreat(pose,np.zeros(7))


def test_rolling_advance_must_reduce_target_distance():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([1.,0.,.9])))
    pose=np.array([0.,0.,np.pi])
    assert d.plan_advance(pose,np.zeros(7))
    assert np.linalg.norm(d.goal[:2]-d.target.position[:2])<.98


def test_replanning_retains_occupied_support_without_clearing_obstacles():
    g=grid();g.floor_views[:]=0
    d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    first=np.zeros(3);later=np.array([.2,0.,0.])
    d.make_checker(first,np.zeros(7))
    check=d.make_checker(later,np.zeros(7))
    assert check.swept(later,first)
    assert not check.clear(np.array([-.4,0.,0.]))
    obstacle(g,np.array([0.,0.,.2]));check.refresh()
    assert not check.clear(first)


def test_observation_turn_reveals_target_side_without_translation():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([.8,-.5,.9])))
    pose=np.zeros(3)
    assert d.plan_observe(pose,np.zeros(7))
    assert np.allclose(d.goal[:2],pose[:2]) and -.31<d.goal[2]<0
    assert d.checker.swept(pose,d.goal)
    obstacle(g,np.array([0.,0.,.2]))
    assert not d.plan_observe(pose,np.zeros(7))


def test_floor_support_uses_calibrated_base_not_mast_or_fixed_clearance_square():
    g=ObservedNavigationMap(MapConfig(extent_m=3.,resolution=.02))
    yy,xx=np.indices(g.floor_views.shape)
    world=g.origin+(np.stack((xx,yy),-1)+.5)*g.config.resolution
    g.floor_views[np.max(np.abs(world),axis=-1)<.15]=2
    small_base=RobotBox('base',np.array([0.,0.,.1]),np.eye(3),np.array([.1,.1,.1]))
    mast=RobotBox('mast',np.array([-.4,0.,1.]),np.eye(3),np.array([.02,.02,.5]))
    check=ObservedShapeChecker(g,[small_base,mast],np.zeros(3))
    assert check.clear(np.array([0.,0.,.3]))
    assert not check.clear(np.array([.3,0.,.3]))


def test_sensor_dock_smoothing_keeps_counter_heading_when_target_map_moves():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([1.,.2,1.])))
    d.goal=np.array([.5,0.,0.]);d.path=[d.goal.copy()]
    d.checker=ObservedShapeChecker(g,boxes(),np.zeros(3));d.smooth_docking(np.zeros(3))
    d.surface_outward=np.array([-1.,0.])
    assert np.allclose(d.curve_command(d.goal),0.)


def test_curve_brakes_for_latency_and_stops_translation_inside_goal_tolerance():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([1.,0.,1.])))
    d.goal=np.array([.5,0.,0.]);d.path=[d.goal.copy()]
    d.checker=ObservedShapeChecker(g,boxes(),np.zeros(3));d.smooth_docking(np.zeros(3))
    for x in (.3,.4,.45,.47):
        command=d.curve_command(np.array([x,0.,0.]))
        v=command[0]
        assert v*.35+v*v/(2*.4)<=.5-x-.022+1e-8
    d.surface_outward=np.array([-1.,0.])
    command=d.curve_command(np.array([.48,.01,.1]))
    assert command[0]==0 and command[1]<0


def test_rolling_advance_prefers_target_heading_when_swept_turn_is_clear():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([2.,1.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    assert abs(d.goal[2]-np.arctan2(1.,2.))<1e-8
    assert d.goal[1]>0


def test_distant_advance_uses_long_observed_corridor_and_preserves_close_band():
    g=grid();d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([3.,0.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    assert np.isclose(np.linalg.norm(d.goal[:2]),1.)
    assert d.curve is not None
    d.reset(NS(position=np.array([1.7,0.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    assert np.isclose(np.linalg.norm(d.goal[:2]),.4)


def test_distant_advance_cannot_cross_unknown_floor_or_intermediate_obstacle():
    g=grid();g.floor_views[:]=0
    yy,xx=np.indices(g.floor_views.shape)
    world=g.origin+(np.stack((xx,yy),-1)+.5)*g.config.resolution
    g.floor_views[(world[:,:,0]<.65)&(np.abs(world[:,:,1])<.5)]=2
    d=ShapedDocking(g,NS(boxes=lambda q:boxes()))
    d.reset(NS(position=np.array([3.,0.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    assert np.linalg.norm(d.goal[:2])<.5
    g.floor_views[:]=2
    obstacle(g,np.array([.5,0.,.2]))
    d.reset(NS(position=np.array([3.,0.,.9])))
    assert d.plan_advance(np.zeros(3),np.zeros(7))
    assert np.linalg.norm(d.goal[:2])<=.25
    assert d.checker.swept(np.zeros(3),d.goal)


def test_blocked_zero_command_requests_reobservation_not_infinite_motion():
    from types import SimpleNamespace as NS
    from cross_episode_sim.navigation.local_scan import ScanConfig
    grid=ObservedNavigationMap(MapConfig(extent_m=4))
    dock=ShapedDocking(grid,None)
    dock.reset(NS(position=np.array([1.,0.,.8])))
    dock.path=[np.array([.3,0,0])];dock.goal=dock.path[0]
    dock.last_frame=0.;dock.started=0.
    dock.checker=NS(clear=lambda p:np.linalg.norm(p)<1e-8,detail={'reason':'unobserved_base_support'})
    packet=NS(timestamp=0.,arm=NS(position=np.array(ScanConfig().stowed_arm)))
    tracking=NS(accepted=True,T_map_base=np.eye(4))
    result=dock.step(packet,tracking,0.)
    assert result['status']=='clearance_reobserve'
    assert result['forward']==0 and dock.clearance_retries==1


def test_three_centimeter_waypoint_residual_still_commands_translation():
    from types import SimpleNamespace as NS
    from cross_episode_sim.navigation.local_scan import ScanConfig
    grid=ObservedNavigationMap(MapConfig(extent_m=4));dock=ShapedDocking(grid,None)
    dock.reset(NS(position=np.array([1.,0.,.8])))
    dock.path=[np.array([.3,0,0])];dock.goal=dock.path[0];dock.last_frame=0.;dock.started=0.
    dock.checker=NS(clear=lambda p:True,detail=None)
    packet=NS(timestamp=0.,arm=NS(position=np.array(ScanConfig().stowed_arm)))
    T=np.eye(4);T[0,3]=.27
    result=dock.step(packet,NS(accepted=True,T_map_base=T),0.)
    assert result['forward']>0


def test_spatial_collision_query_matches_exhaustive_boxes():
    from cross_episode_sim.navigation.shaped_docking import rotation
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(41);g=grid()
    for point in rng.uniform([-1.5,-1.5,.05],[1.5,1.5,1.8],(150,3)):obstacle(g,point)
    shapes=boxes()+[RobotBox('tilted_arm',np.array([.5,.1,1.1]),Rotation.from_euler('xyz',[.4,.7,.2]).as_matrix(),np.array([.06,.15,.08]))]
    check=ObservedShapeChecker(g,shapes,np.zeros(3))
    pad=check.margin+np.linalg.norm([g.config.resolution,g.config.resolution,g.config.height_resolution])/2
    for pose in rng.uniform([-.7,-.7,-np.pi],[.7,.7,np.pi],(100,3)):
        points=check.points[np.linalg.norm(check.points[:,:2]-pose[:2],axis=1)<1.2].copy()
        points[:,:2]=(points[:,:2]-pose[:2])@rotation(pose[2])
        collision=any(np.any(np.all(np.abs((points-b.center)@b.axes)<=b.half+pad,axis=1)) for b in shapes)
        assert check.clear(pose)==(not collision)
