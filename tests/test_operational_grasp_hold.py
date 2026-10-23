from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('operational,contacts,expected', [(True,True,.25),(False,True,2.),(True,False,None)])
def test_operational_hold_requires_sustained_stability(tmp_path, monkeypatch, operational, contacts, expected):
    from cross_episode_sim.manipulation import grasp_qualification as module
    from cross_episode_sim.control import timing
    monkeypatch.setattr(module.RoboCasaManipulation,'pick_payload',lambda self:None)
    monkeypatch.setattr(module,'model_digest',lambda p:'digest')
    monkeypatch.setattr(timing,'payload_ready',lambda c:True)
    monkeypatch.setattr(timing,'arm_ready',lambda c:True)
    c=module.GraspQualification.__new__(module.GraspQualification)
    c.args=SimpleNamespace(fixed_atomic_interface=operational,annotation_source='test',soft_finger=False)
    c.adaptive_motion_settling=True
    c.data=SimpleNamespace(time=0.,actuator=lambda name:SimpleNamespace(ctrl=[1.]))
    c.tick=lambda dt:setattr(c.data,'time',c.data.time+dt)
    pose=np.eye(4);pose[2,3]=.2
    c.tcp=lambda:np.eye(4)
    c.bread_pose=lambda:pose
    c.pickup_start_height=0.
    c.support_contacts=lambda **kw:False
    c.finger_object_contact=lambda:dict(fingers=['a','b'] if contacts else [],
        normal_force_n={'a':1.,'b':1.},depth_m=0.)
    c.asset_metadata=dict(model_xml='dummy.xml',asset='test',source='test',object_scale=1.)
    c.output=tmp_path
    c.profile=SimpleNamespace(name='test',namespace='robot/',gripper_actuator='grip')
    c.report={'annotation_selection':dict(local_transform=np.eye(4).tolist(),selected_index=0)}
    c.record=lambda **kw:None
    if expected is None:
        with pytest.raises(RuntimeError,match='sustained stability'):
            c.pick_payload()
        assert not (tmp_path/'qualified_grasp.json').exists()
    else:
        c.pick_payload()
        import json
        record=json.loads((tmp_path/'qualified_grasp.json').read_text())
        assert record['hold_seconds']==pytest.approx(expected)
