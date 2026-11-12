from cross_episode_sim.perception.molmo_pointing import parse_points,PointingSupervisor

def test_native_multi_point_format():
    assert parse_points('<points coords="1 1 424 302 2 468 296">bowls</points>',1000,1000)==[(424.,302.),(468.,296.)]

def test_absence_and_bad_coordinates_not_motion_targets():
    assert parse_points('There are none.',640,480)==[]
    assert parse_points('<points coords="1 1 1001 020">bowl</points>',640,480)==[]

def test_confirmation_rate_fits_freshness_window():
    from cross_episode_sim.navigation.readiness import ManipulationReadiness
    supervisor=PointingSupervisor(None,None,'bowl')
    assert 2*supervisor.period<ManipulationReadiness().max_age


def pointed_scene():
    import numpy as np
    from types import SimpleNamespace
    K=np.array([[300.,0,320],[0,300.,240],[0,0,1]]);depth=np.full((480,640),2.,np.float32)
    cup=np.zeros((480,640),bool);cup[200:216,396:408]=True;depth[cup]=1.5
    sink=np.zeros((480,640),bool);sink[150:230,150:396]=True;depth[150:230,150:396]=np.linspace(1.2,2.6,246,dtype=np.float32)
    # Camera looks along map +x, so image rows are height and a vessel has vertical extent.
    mount=np.array([[0.,0,1,0],[-1,0,0,0],[0,-1,0,1.2],[0,0,0,1]])
    camera=SimpleNamespace(depth_m=depth,valid_depth=np.ones_like(cup),robot_mask=None,K=K,T_base_optical=mount)
    return camera,cup,sink,SimpleNamespace(T_map_base=np.eye(4))


def test_point_beside_small_object_uses_adjacent_object_mask():
    camera,cup,sink,tracking=pointed_scene();supervisor=PointingSupervisor(None,None,'cup')
    # Four pixels left of the cup: the only mask under the point is the wide basin.
    assert sink[208,392] and not cup[208,392]
    assert supervisor.pointed_mask(camera,[sink,cup],392,208,tracking) is cup
    assert supervisor.pointed_mask(camera,[sink,cup],400,208,tracking) is cup


def test_point_far_from_any_object_mask_is_unchanged():
    camera,cup,sink,tracking=pointed_scene();supervisor=PointingSupervisor(None,None,'cup')
    assert supervisor.pointed_mask(camera,[sink,cup],380,208,tracking) is sink
    assert supervisor.pointed_mask(camera,[cup],380,208,tracking) is None
    supervisor.refine_mask=lambda camera,mask:None
    assert supervisor.pointed_mask(camera,[sink,cup],392,208,tracking) is None


def test_adjacent_mask_inside_robot_volume_is_not_adopted():
    import numpy as np
    camera,cup,sink,tracking=pointed_scene();supervisor=PointingSupervisor(None,None,'cup')
    # The neighbouring mask lies inside a calibrated robot link volume: it is the arm, not a cup.
    camera.robot_bounds_base=[(np.eye(4),np.array([1.5,-.4,1.3]),np.array([.3,.3,.3]))]
    assert supervisor.pointed_mask(camera,[sink,cup],392,208,tracking) is sink
