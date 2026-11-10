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
