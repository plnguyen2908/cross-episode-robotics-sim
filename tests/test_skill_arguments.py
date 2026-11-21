from cross_episode_sim.evaluation.skill_arguments import (ArgumentReading, interpret_navigate, interpret_place,
                                                          clean_target, station_of)
from cross_episode_sim.skills.household_sensor_plans import household_plan, TASKS


def read(call, held):
    if call.skill == 'navigate':
        args, transit, notes, error = interpret_navigate(call.arguments, held)
        assert error is None and transit is None, (call, notes)
        return args
    if call.skill == 'place':return interpret_place(call.arguments)[0]
    return clean_target(call.arguments)


def test_recipe_arguments_pass_through_unchanged():
    for task in TASKS:
        held = None
        for call in household_plan(task):
            assert read(call, held) == call.arguments, (task, call)
            if call.skill == 'pick':held = call.arguments['target']
            if call.skill == 'place':held = None


def test_place_purpose_while_holding_is_a_carry():
    args, transit, notes, error = interpret_navigate(dict(target='dining table', purpose='place'), 'cup')
    assert args == dict(target='cup', destination='dining table', purpose='place') and transit is None and notes


def test_room_target_is_a_room_transit_and_unknown_rooms_fail():
    args, transit, _, _ = interpret_navigate(dict(target='kitchen', purpose='pick'), None)
    assert transit == 'kitchen'
    assert interpret_navigate(dict(target='dining_room'), None)[1] == 'dining table'
    assert interpret_navigate(dict(target='bedroom'), None)[3] == 'unsupported_room:bedroom'
    # holding something: a room target is not turned into a transit
    assert interpret_navigate(dict(target='kitchen', purpose='place'), 'packaged drink')[0]['destination'] == 'kitchen'
    assert interpret_navigate(dict(target='cup', search_at='kitchen', mode='transit'), None)[1] == 'kitchen'


def test_names_and_search_places():
    args, _, _, error = interpret_navigate(dict(target='cold_drink', search_at='fridge'), None)
    assert args == dict(target='cold drink', search_at='kitchen') and error is None
    assert interpret_navigate(dict(target='dining_table', search_at='dining_room'), None)[0]['search_at'] == 'dining table'
    assert interpret_navigate(dict(target='cup', search_at='garage'), None)[3] == 'unknown_search_station:garage'
    assert station_of('Kitchen counter') == 'kitchen' and station_of('living room') is None
    assert clean_target(dict(target='stove_knob', state='on')) == dict(target='stove knob', state='on')


def test_place_regions():
    assert interpret_place(dict(receptacle='purple basket', region='default'))[0]['region'] == 'interior'
    assert interpret_place(dict(receptacle='kitchen drawer', region='inside'))[0]['region'] == 'interior'
    assert interpret_place(dict(receptacle='dining_table'))[0] == dict(receptacle='dining table')
    assert 'region' not in interpret_place(dict(receptacle='front left stove burner'))[0]


def test_wrapper_logs_and_delegates():
    class Inner:
        history = ['x']
        def __init__(self):self.calls = []
        def execute_skill(self, name, args):self.calls.append((name, args));return dict(success=True, reason='ok', wall_seconds=0.)
    inner = Inner();w = ArgumentReading(inner, lambda: 'cup')
    w.execute_skill('navigate', dict(target='dining table', purpose='place'))
    assert inner.calls[-1] == ('navigate', dict(target='cup', destination='dining table', purpose='place')) and w.history == ['x']
    r = ArgumentReading(inner, lambda: None).execute_skill('navigate', dict(target='kitchen'))
    assert inner.calls[-1] == ('navigate', dict(target='kitchen', mode='transit', search_at='kitchen')) and r['success']
    assert not ArgumentReading(inner, lambda: None).execute_skill('navigate', dict(target='bedroom'))['success']


def test_empty_hand_destination_that_names_the_target_is_a_go_to():
    a, t, _, e = interpret_navigate(dict(target='dining table', destination='dining table', search_at='dining table', purpose='pick'), None)
    assert 'destination' not in a and t is None and e is None
    a, t, _, _ = interpret_navigate(dict(target='kitchen', destination='kitchen', purpose='place'), None)
    assert 'destination' not in a and t == 'kitchen'
    # holding: the same call is a carry; a recipe carry with an empty hand stays a carry (and fails)
    assert interpret_navigate(dict(target='kitchen', destination='kitchen', purpose='place'), 'packaged drink')[0]['destination'] == 'kitchen'
    assert interpret_navigate(dict(target='bowl', destination='dining table'), None)[0] == dict(target='bowl', destination='dining table')


def test_place_as_redock_target_uses_the_default():
    assert 'redock_target' not in clean_target(dict(target='packaged drink', redock=True, redock_target='dining table'))
    assert 'redock_target' not in interpret_place(dict(receptacle='purple basket', redock_target='kitchen'))[0]
    assert clean_target(dict(target='fridge', redock_target='fridge handle'))['redock_target'] == 'fridge handle'


def test_station_target_searches_in_its_own_room():
    a, t, notes, e = interpret_navigate(dict(target='dining table', purpose='place'), None)
    assert a['search_at'] == 'dining table' and t is None and e is None and notes
    a, _, _, _ = interpret_navigate(dict(target='dining table', search_at='kitchen'), None)
    assert a['search_at'] == 'dining table'
    # plain objects keep the catalog default; holding: a carry, not a search
    assert 'search_at' not in interpret_navigate(dict(target='cup'), None)[0]
    assert interpret_navigate(dict(target='dining table', purpose='place'), 'cup')[0]['destination'] == 'dining table'
