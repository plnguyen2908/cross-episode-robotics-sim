import numpy as np
from cross_episode_sim.control.rim_contact_refinement import outward_rim_contacts


def test_backoff_follows_local_axis_preserves_contact_and_original():
    pose = np.eye(4); pose[:3, :3] = [[0,0,1],[1,0,0],[0,1,0]]
    pose[:3, 3] = [.6,.1,.95]
    row = dict(tcp=pose.tolist(), contact=[.6,.1,.94], confidence=.9)
    result = list(outward_rim_contacts([row], offsets=(0., .009)))
    np.testing.assert_allclose(np.asarray(result[1]['tcp'])[:3,3], [.591,.1,.95])
    np.testing.assert_allclose(np.asarray(result[1]['tcp'])[:3,:3],pose[:3,:3])
    assert result[1]['contact'] == row['contact'] and result[1]['confidence'] == .9
    np.testing.assert_array_equal(row['tcp'],pose)
