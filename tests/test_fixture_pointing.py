from types import SimpleNamespace
from PIL import Image
from cross_episode_sim.perception.fixture_pointing import ApplianceAssociatedPointer,fixture_pointing_supervisor


def test_appliance_association_rejects_other_handle_and_retains_correct_point(monkeypatch):
    import cross_episode_sim.perception.fixture_pointing as module
    answers=iter(['{"category":"other","description":"drawer handle"}',
                  '{"category":"microwave","description":"microwave door frame"}'])
    monkeypatch.setattr(module,'LocalVLM',lambda **kw:SimpleNamespace(label_with_prompt=lambda *a,**k:next(answers)))
    raw=dict(points=[[10,20],[30,40]],raw='two candidate points',wall_seconds=.1)
    pointer=ApplianceAssociatedPointer(SimpleNamespace(point=lambda *a:raw))
    result=pointer.point(Image.new('RGB',(100,100)),'microwave edge')
    assert result['points']==[[30,40]]
    assert raw['points']==[[10,20],[30,40]]
    assert [event['accepted'] for event in pointer.events]==[False,True]


def test_uncertain_fixture_output_is_not_a_grasp_authorization(monkeypatch):
    import cross_episode_sim.perception.fixture_pointing as module
    monkeypatch.setattr(module,'LocalVLM',lambda **kw:SimpleNamespace(label_with_prompt=lambda *a,**k:'truncated JSON'))
    pointer=ApplianceAssociatedPointer(SimpleNamespace(point=lambda *a:dict(points=[[10,20]],raw='point',wall_seconds=0)))
    assert not pointer.point(Image.new('RGB',(100,100)),'handle')['points']


def test_nonfixture_supervisor_preserves_existing_pointer():
    pointer=object()
    assert fixture_pointing_supervisor(pointer,None,'bowl').pointer is pointer
