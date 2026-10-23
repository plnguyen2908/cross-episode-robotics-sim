from cross_episode_sim.generation import asset_suitability
from cross_episode_sim.generation.household_scene import select_assets


def test_unavailable_asset_is_audited_and_does_not_abort_valid_pool(monkeypatch):
    def geometry(source,asset,key):
        if key=='missing':raise FileNotFoundError('missing mesh')
        return 2,(.1,.1,.1)
    monkeypatch.setattr(asset_suitability,'top_down_geometry',geometry)
    rows=[dict(source='robocasa',asset='lightwheel/pot/'+key,key=key,placement_passes=1)
          for key in ('missing','available')]
    rejected=[]
    selected=select_assets({'objects':{'pot':dict(category='pot',support='kitchen_counter')}},
        dict(seed=7,asset_instance_set='baseline'),rows,rejected)
    assert selected['pot']['asset_row']['key']=='available'
    assert rejected==[{'key':'missing','reason':'missing mesh'}]


def test_rejects_side_only_grasps_and_saucepan_lids(monkeypatch):
    def geometry(source, asset, key):
        return (0 if key == 'side_only' else 2), (.2,.2,.08)
    monkeypatch.setattr(asset_suitability,'top_down_geometry',geometry)
    rows=[dict(source='robocasa',asset='lightwheel/'+category+'/'+key,key=key,
               placement_passes=1,cross_room_passes=1)
          for category,key in [('pot','side_only'),('saucepan','Lid'),('pot','top_down')]]
    spec={'objects':{'pot_01':dict(category='pot',support='kitchen_counter')}}
    selected=select_assets(spec,dict(seed=7,asset_instance_set='baseline'),rows)
    assert selected['pot_01']['asset_row']['key']=='top_down'


def test_prefers_shorter_instances_without_changing_category(monkeypatch):
    monkeypatch.setattr(asset_suitability,'top_down_geometry',
        lambda source,asset,key:(2,(.05,.05,.12 if key=='short' else .3)))
    rows=[dict(source='robocasa',asset='lightwheel/juice/'+key,key=key,
               placement_passes=1,cross_room_passes=1) for key in ('tall','short')]
    spec={'objects':{'drink':dict(category='drink_bottle',support='kitchen_counter')}}
    params=dict(seed=7,asset_instance_set='baseline')
    assert select_assets(spec,params,rows)['drink']['asset_row']['key']=='short'
    params['asset_instance_set']='same_category_alternatives'
    assert select_assets(spec,params,rows)['drink']['asset_row']['key']=='tall'


def test_two_pot_selection_rejects_overlapping_burner_footprints(monkeypatch):
    from cross_episode_sim.generation import asset_suitability
    monkeypatch.setattr(asset_suitability, 'top_down_geometry',
                        lambda source, asset, key: (1, (.40, .31, .08) if key == 'wide' else (.20, .20, .12)))
    rows = [dict(key=k, source='robocasa', asset='pot/'+k, category='pot',
                 placement_passes=1) for k in ('wide', 'small')]
    # Category extraction uses the asset path; follow native lightwheel naming.
    for row in rows:
        row['asset'] = 'lightwheel/pot/'+row['key']
    audit = []
    selected = select_assets(dict(objects={'pot': dict(category='pot', support='kitchen_counter')},
                                  fixtures=['burner_1', 'burner_2']),
                             dict(seed=7, asset_instance_set='baseline'), rows, audit)
    assert selected['pot']['asset_row']['key'] == 'small'
    assert audit[0]['key'] == 'wide'


def test_stove_allows_qualified_pan_but_excludes_lid_and_failed_pot(monkeypatch):
    monkeypatch.setattr(asset_suitability, 'top_down_geometry',
                        lambda *args: (2, (.2, .18, .08)))
    rows = [dict(source='robocasa', asset=asset, key=asset, placement_passes=1)
            for asset in ('lightwheel/pot/Pot057', 'lightwheel/saucepan/Saucepan002_Lid',
                          'lightwheel/pan/PanGood')]
    spec = dict(objects={'pot_01': dict(category='pot', support='kitchen_counter')},
                fixtures=['burner_1', 'burner_2'])
    audit = []
    chosen = select_assets(spec, dict(seed=7, asset_instance_set='baseline'), rows, audit)
    assert chosen['pot_01']['asset_row']['asset'] == 'lightwheel/pan/PanGood'
    assert len(audit) == 2


def test_explicit_asset_selection_keeps_qualification_checks(monkeypatch):
    import pytest
    monkeypatch.setattr(asset_suitability, 'top_down_geometry', lambda *a: (2, (.2,.2,.1)))
    rows = [dict(source='robocasa', asset='lightwheel/pot/'+k, key=k, placement_passes=n)
            for k,n in [('first',1),('chosen',1),('unqualified',0)]]
    spec = {'objects': {'pot_01': dict(category='pot',support='kitchen_counter')}}
    params = dict(seed=7,asset_instance_set='baseline',asset_keys={'pot_01':'chosen'})
    assert select_assets(spec,params,rows)['pot_01']['asset_row']['key']=='chosen'
    params['asset_keys']['pot_01']='unqualified'
    with pytest.raises(ValueError,match='qualification/fit'):
        select_assets(spec,params,rows)
