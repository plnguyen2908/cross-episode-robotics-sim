import json
import numpy as np
from cross_episode_sim.mapping.continuity import load_memory,save_memory

def test_reanchors_only_saved_observations(tmp_path):
    (tmp_path/'slam').mkdir();pose=np.eye(4);pose[:3,3]=[2,3,0]
    np.save(tmp_path/'slam/estimated_base_poses.npy',np.array([pose]))
    save_memory(tmp_path,{'rooms':[],'tables':[]},room_events=[{'confirmed_label':'kitchen','estimated_position':[3,5,0], 'timestamp':10}])
    memory=load_memory(tmp_path)
    assert memory['rooms'][0]['position']==[1,2,0]
    assert memory['rooms'][0]['label']=='kitchen'
    assert json.loads((tmp_path/'semantic_memory.json').read_text())['rooms'][0]['position']==[3,5,0]

def test_missing_memory_does_not_invent_rooms(tmp_path):
    assert load_memory(tmp_path)==dict(rooms=[],tables=[])

def test_unknown_room_not_saved(tmp_path):
    save_memory(tmp_path,dict(rooms=[],tables=[]),room_events=[{'confirmed_label':'unknown'}])
    assert json.loads((tmp_path/'semantic_memory.json').read_text())['rooms']==[]


def test_remembered_table_retains_real_observation_timestamp():
    from cross_episode_sim.mapping.continuity import table_targets
    targets=table_targets({'tables':[dict(track_id=10,label='dining table',position=[1,2,1],score=.9,observations=3,last_seen=12.)]})
    assert targets[0].navigation_memory
    assert targets[0].last_seen==12.
    assert targets[0].observations==3


def test_vessel_hypotheses_survive_handoffs_without_freshness_inflation(tmp_path):
    from cross_episode_sim.perception.visual_targets import VisualTarget
    from cross_episode_sim.mapping.continuity import object_hypotheses
    (tmp_path/'slam').mkdir();pose=np.eye(4);pose[:3,3]=[2,3,0]
    np.save(tmp_path/'slam/estimated_base_poses.npy',np.array([pose]))
    cup=VisualTarget(4,'cup',np.array([3.,5.,.9]),.8,observations=1,last_seen=12.)
    save_memory(tmp_path,dict(rooms=[],tables=[]),targets=[cup])
    memory=load_memory(tmp_path);hypothesis=object_hypotheses(memory)[0]
    np.testing.assert_allclose(hypothesis.position,[1,2,.9])
    assert hypothesis.last_seen==12. and hypothesis.observations==1
    # A stage without object detections retains the same evidence, not a new sighting.
    save_memory(tmp_path,memory)
    saved=json.loads((tmp_path/'semantic_memory.json').read_text())['objects'][0]
    assert saved['last_seen']==12. and saved['observations']==1


def test_new_observation_replaces_nearby_hypothesis_but_preserves_other_categories(tmp_path):
    from cross_episode_sim.perception.visual_targets import VisualTarget
    cup=VisualTarget(0,'cup',np.array([1.,2.,.9]),.8,observations=2,last_seen=12.)
    bowl=VisualTarget(1,'bowl',np.array([2.,2.,.9]),.8,observations=2,last_seen=12.)
    save_memory(tmp_path,dict(rooms=[],tables=[]),targets=[cup,bowl])
    memory=json.loads((tmp_path/'semantic_memory.json').read_text())
    cup.position[0]+=.03;cup.last_seen=20.;cup.observations=3
    save_memory(tmp_path,memory,targets=[cup])
    objects=json.loads((tmp_path/'semantic_memory.json').read_text())['objects']
    assert len(objects)==2 and objects[0]['last_seen']==20. and objects[1]['label']=='bowl'
