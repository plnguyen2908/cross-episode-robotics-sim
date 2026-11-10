"""Exercise the metadata contract between observation, posture, dock and place."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cross_episode_sim.runtime.basket_placement import place_in_basket


def test_post_raise_checkpoint_retains_arrival_metadata(tmp_path,monkeypatch):
    monkeypatch.setattr('cross_episode_sim.navigation.receptacle_approach.approach_receptacle',lambda *a,**k:None)
    monkeypatch.setattr('cross_episode_sim.control.raised_payload_posture.raise_payload_clearance',lambda *a,**k:None)
    carry=tmp_path/'carry';carry.mkdir();np.save(carry/'observed_payload_tool.npy',np.zeros((8,3)))
    stages=[]
    class Pipeline:
        runtime=SimpleNamespace()
        def checkpoint(self,path,previous):
            path.mkdir();return path  # Checkpoint does not create a stage report.
        def stage(self,name,command,report,success_key):
            stages.append(name)
            if '--start' in command:
                arrival=Path(command[command.index('--start')+1])/'report.json'
                assert json.loads(arrival.read_text())['initialization_valid'] is True
            report.parent.mkdir(exist_ok=True)
            report.write_text(json.dumps({success_key:True}))
    placed,dock=place_in_basket(Pipeline(),tmp_path,carry,'purple basket',
        'inside bottom of the purple basket','kitchen',2,lambda *a:None)
    assert stages==['2_observe_receptacle','2_place_dock','2_place']
    assert placed.name=='place' and dock.name=='place_dock'
