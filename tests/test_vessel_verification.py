import json
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.perception.vessel_verification import SETTLED_VOTES,VerifiedVesselSupervisor
from tests.test_molmo_pointing import pointed_scene


class Verifier:
    """Names the highlighted mask from the scripted answer for its image column."""
    def __init__(self,answers):self.answers=answers;self.calls=0
    def label_with_prompt(self,images,prompt,max_new_tokens=100):
        self.calls+=1
        overlay=np.asarray(images[0]);column=int(np.where((overlay[:,:,0]>overlay[:,:,1]).any(0))[0].min())
        return self.answers[column]


def vessel_scene():
    camera,cup,_,tracking=pointed_scene()
    bowl=np.zeros_like(cup);bowl[204:216,300:330]=True;camera.depth_m[bowl]=1.5
    camera.rgb=np.full(cup.shape+(3,),90,np.uint8);camera.frame_id='navigation'
    answers={396:json.dumps(dict(description='tall drinking vessel',category='cup')),
             300:json.dumps(dict(description='shallow wide bowl',category='bowl'))}
    return camera,cup,bowl,SimpleNamespace(T_map_base=tracking.T_map_base,accepted=True),answers


def supervisor_for(label,camera,masks,points,answers):
    pointer=SimpleNamespace(point=lambda image,target:dict(raw='',points=points,wall_seconds=0.))
    segmenter=SimpleNamespace(segment=lambda image:(None,masks))
    return VerifiedVesselSupervisor(pointer,segmenter,label,verifier=Verifier(answers))


def test_cup_pointed_as_bowl_never_becomes_a_bowl_target():
    camera,cup,bowl,tracking,answers=vessel_scene()
    supervisor=supervisor_for('bowl',camera,[cup,bowl],[(315.,210.),(402.,208.)],answers)
    for timestamp in (1.,2.):
        supervisor.observe(SimpleNamespace(timestamp=timestamp,cameras=dict(navigation=camera)),tracking)
    assert len(supervisor.targets.targets)==1 and supervisor.targets.targets[0].observations==2
    evidence=supervisor.events[-1]['category_verification']
    assert [(e['category'],e['accepted']) for e in evidence]==[('bowl',True),('cup',False)]
    assert supervisor.events[-1]['masks']==1


def test_same_objects_are_cup_targets_when_cups_are_requested():
    camera,cup,bowl,tracking,answers=vessel_scene()
    supervisor=supervisor_for('cup',camera,[cup,bowl],[(315.,210.),(402.,208.)],answers)
    supervisor.observe(SimpleNamespace(timestamp=1.,cameras=dict(navigation=camera)),tracking)
    assert [(e['category'],e['accepted']) for e in supervisor.events[-1]['category_verification']]==[('bowl',False),('cup',True)]
    assert supervisor.events[-1]['masks']==1


def test_two_points_on_one_rejected_mask_are_checked_once():
    camera,cup,bowl,tracking,answers=vessel_scene()
    supervisor=supervisor_for('bowl',camera,[cup,bowl],[(400.,206.),(402.,210.)],answers)
    supervisor.observe(SimpleNamespace(timestamp=1.,cameras=dict(navigation=camera)),tracking)
    assert supervisor.verifier.calls==1 and not supervisor.targets.targets


def test_unreadable_answer_is_not_accepted():
    camera,cup,bowl,tracking,answers=vessel_scene()
    supervisor=supervisor_for('bowl',camera,[bowl],[(315.,210.)],{300:'a bowl, probably'})
    supervisor.observe(SimpleNamespace(timestamp=1.,cameras=dict(navigation=camera)),tracking)
    assert supervisor.events[-1]['category_verification'][0]['category']=='unknown' and not supervisor.targets.targets


def test_settled_verdict_is_reused_until_a_closer_look():
    camera,cup,bowl,tracking,answers=vessel_scene()
    supervisor=supervisor_for('bowl',camera,[cup],[(402.,208.)],answers)
    for _ in range(SETTLED_VOTES+2):verdict=supervisor.verify_mask(camera,cup,tracking)
    assert supervisor.verifier.calls==SETTLED_VOTES and verdict['remembered'] and not verdict['accepted']
    # The robot has halved its distance to that object: the far verdicts are not reused.
    position=np.array(verdict['position']);closer=np.eye(4);closer[:2,3]=position[:2]/2
    moved=SimpleNamespace(**{**vars(camera),'T_base_optical':np.linalg.inv(closer)@camera.T_base_optical})
    verdict=supervisor.verify_mask(moved,cup,SimpleNamespace(T_map_base=closer))
    assert supervisor.verifier.calls==SETTLED_VOTES+1 and 'remembered' not in verdict
    assert np.allclose(verdict['position'],position,atol=1e-6)


def test_disagreeing_verdicts_keep_being_asked():
    camera,cup,bowl,tracking,answers=vessel_scene()
    supervisor=supervisor_for('bowl',camera,[cup],[(402.,208.)],answers)
    supervisor.verify_mask(camera,cup,tracking)
    supervisor.verifier.answers={396:answers[300]};supervisor.verify_mask(camera,cup,tracking)
    supervisor.verifier.answers=answers
    for _ in range(SETTLED_VOTES+1):verdict=supervisor.verify_mask(camera,cup,tracking)
    assert supervisor.verifier.calls==SETTLED_VOTES+3 and 'remembered' not in verdict
