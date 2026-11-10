import json
import h5py
import numpy as np
from cross_episode_sim.control.recorded_contact_support import recorded_contact_support


def test_resume_support_uses_only_accepted_observations_before_grasp(tmp_path):
    grasp=tmp_path/'episode/doors/door_001/grasp'
    (grasp/'state_recording').mkdir(parents=True)
    with h5py.File(grasp/'state_recording/states.h5','w') as f:f['times']=[10.,11.]
    record=tmp_path/'episode/state_recording';record.mkdir()
    np.savez(record/'past.npz',floor=np.array([[2,0],[0,0]]),
             minimum_floor_views=2,resolution=1.,origin=np.zeros(2))
    pose=np.eye(4);pose[:2,3]=[1.,2.]
    rows=[dict(timestamp=9.,accepted=True,map='past.npz',pose=pose.tolist()),
          dict(timestamp=9.5,accepted=False,map='unaccepted.npz',pose=pose.tolist()),
          dict(timestamp=12.,accepted=True,map='future.npz',pose=pose.tolist())]
    (record/'diagnostics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    region,source=recorded_contact_support(grasp)
    np.testing.assert_allclose(region.bounds,[-1.,-2.,0.,-1.])
    assert source['timestamp']==9. and source['floor_cells']==1
    explicit,explicit_source=recorded_contact_support(grasp,observed_root=tmp_path/'episode')
    np.testing.assert_allclose(explicit.bounds,region.bounds)
    assert explicit_source==source
