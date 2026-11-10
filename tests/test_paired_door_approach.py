from cross_episode_sim.navigation.paired_door_approach import adjacent_closed_leaf


def test_pair_only_closed_same_plane_same_room_handles():
    a=dict(portal='a',position=[0.,0.,1.],source='r',closed=False)
    b=dict(portal='b',position=[.01,.2,1.],source='r',closed=True)
    assert adjacent_closed_leaf([a,b],a,[1,0]) is b
    for changes in [dict(closed=False),dict(source='elsewhere'),dict(position=[.4,.2,1.]),dict(duplicate_of='a'),dict(visual_category='other'),dict(visual_category='open_room_door')]:
        assert adjacent_closed_leaf([a,dict(b,**changes)],a,[1,0]) is None


def test_pair_order_is_relative_to_observed_facing_direction():
    from cross_episode_sim.navigation.paired_door_approach import observed_side_leaf
    a=dict(portal='a',position=[-3.,-.2,1.],source='r',observations=2,closed=True)
    b=dict(portal='b',position=[-3.,.2,1.],source='r',observations=2,closed=True)
    assert observed_side_leaf([a,b],a,[0,0],'left') is a
    assert observed_side_leaf([a,b],a,[0,0],'right') is b
    assert observed_side_leaf([a,b],a,[-6,0],'left') is b
