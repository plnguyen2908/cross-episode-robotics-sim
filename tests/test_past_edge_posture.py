"""The arm is stowed before the drive to the free-edge stand spot (fix156 call 16)."""
import types,sys
from pathlib import Path
import cross_episode_sim.runtime.fixture_controls as fc


def test_posture_before_first_stand_drive(monkeypatch,tmp_path):
    calls=[]
    class Dock:
        def __init__(self,*a,turn_degrees=0.,**k):self.turn=turn_degrees
        def move(self,*a,**k):calls.append(('drive',self.turn));return dict(success=True)
    import cross_episode_sim.code_policy.closing_redock as cr
    monkeypatch.setattr(cr,'PastEdgeRedocking',Dock)
    monkeypatch.setattr(cr,'PUSH_OPEN_STAND_TURNS',(-60.,60.))
    import cross_episode_sim.runtime.episode as ep
    class Sensors:
        def __enter__(self):return object()
        def __exit__(self,*a):return False
    monkeypatch.setattr(ep,'episode_sensors',lambda *a:Sensors())
    owner=fc.LiveFixtureControls.__new__(fc.LiveFixtureControls)
    owner.runtime=types.SimpleNamespace(model=None,data=None,mapper=types.SimpleNamespace(view=lambda:None),qwen_docking=True)
    owner.hinges={'microwave':{}}
    owner.navigation_posture=lambda root,progress:calls.append(('posture',None))
    assert owner._stand_past_free_edge('microwave',tmp_path)==-60.
    assert calls==[('posture',None),('drive',-60.)]
    assert owner.runtime.qwen_docking is True
