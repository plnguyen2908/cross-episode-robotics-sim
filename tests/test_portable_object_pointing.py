from types import SimpleNamespace as NS
from PIL import Image
from cross_episode_sim.perception.portable_object_pointing import PortableFoodPointer, household_pointing_supervisor


def test_appliance_proposal_is_rejected_but_food_container_is_kept():
    raw = dict(points=[[20, 20], [60, 60]], raw='proposals', wall_seconds=0.)
    answers = iter(['{"category":"other","description":"toaster"}',
                    '{"category":"food container","description":"meal box"}'])
    pointer = PortableFoodPointer(NS(point=lambda *a: raw),
                                  NS(label_with_prompt=lambda *a, **k: next(answers)))
    assert pointer.point(Image.new('RGB', (100, 100)), 'food containers')['points'] == [[60, 60]]
    assert raw['points'] == [[20, 20], [60, 60]]


def test_unparseable_confirmation_does_not_authorize_approach():
    pointer = PortableFoodPointer(NS(point=lambda *a: dict(points=[[20, 20]])),
                                  NS(label_with_prompt=lambda *a, **k: 'incomplete'))
    assert pointer.point(Image.new('RGB', (100, 100)), 'food containers')['points'] == []


def test_other_household_targets_keep_existing_pointer():
    pointer = object()
    for label in ('cup', 'bowl', 'cookware on kitchen counter'):
        assert household_pointing_supervisor(pointer, None, label).pointer is pointer


def test_location_qualified_food_targets_use_verifier(monkeypatch):
    import cross_episode_sim.perception.portable_object_pointing as module
    monkeypatch.setattr(module, 'LocalVLM', lambda **kwargs: object())
    for label in ('food container', 'food container on kitchen counter', 'food container inside microwave'):
        assert isinstance(household_pointing_supervisor(object(), None, label).pointer, PortableFoodPointer)


def test_grocery_query_distinguishes_bottles_and_boxes_without_changing_target_label():
    calls=[]
    pointer=NS(point=lambda image,target: calls.append(target) or {'points': []})
    for label,word in [('packaged drink','drink bottles'),('packaged snack','rectangular blocks')]:
        supervisor=household_pointing_supervisor(pointer,None,label)
        supervisor.pointer.point(Image.new('RGB',(100,100)), label+'s')
        assert word in calls[-1]
        assert supervisor.label==label


def test_grocery_wrong_category_is_not_relabelled_as_requested_drink():
    from cross_episode_sim.perception.portable_object_pointing import GroceryPointer
    raw=dict(points=[[20,20],[60,60]],raw='two proposals')
    answers=iter(['{"category":"packaged snack","description":"plain block"}',
                  '{"category":"packaged drink","description":"printed beverage bottle"}'])
    pointer=GroceryPointer(NS(point=lambda *args:raw),'packaged drink',
                           NS(label_with_prompt=lambda *args,**kw:next(answers)))
    result=pointer.point(Image.new('RGB',(100,100)),'packaged drink')
    assert result['points']==[[60,60]]
    assert [r['accepted'] for r in result['category_verification']]==[False,True]
    assert len(raw['points'])==2


def test_grocery_unknown_and_malformed_responses_do_not_invent_targets():
    from cross_episode_sim.perception.portable_object_pointing import GroceryPointer
    answers=iter(['{"category":"unknown","description":"occluded"}','broken JSON'])
    pointer=GroceryPointer(NS(point=lambda *args:dict(points=[[20,20],[60,60]])),
                           'packaged drink',NS(label_with_prompt=lambda *args,**kw:next(answers)))
    assert pointer.point(Image.new('RGB',(100,100)),'packaged drink')['points']==[]
