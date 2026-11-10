from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.control.grasp_contact_depth import (
    calibrated_contact_variants, contact_depth_matches, pad_depth_interval,
)

class Fingers:
    def tcp_pose(self, joints):
        return np.eye(4)
    def boxes(self, joints):
        return [SimpleNamespace(name=f'robot_0/gripper/{side}_pad',
                    center=np.array([0.,y,-.015]), axes=np.eye(3),
                    half=np.array([.011,.002,.02]))
                for side,y in [('left',-.045),('right',.045)]]

def test_contact_beyond_fingertips_is_not_a_grasp():
    interval=pad_depth_interval(Fingers(),np.zeros(7))
    assert contact_depth_matches(dict(tcp=np.eye(4),contact=[0.,.03,-.01]),interval)
    assert not contact_depth_matches(dict(tcp=np.eye(4),contact=[0.,.03,.01]),interval)

def test_deeper_variant_retains_observed_contact_and_does_not_mutate_model_pose():
    pose=np.eye(4);row=dict(tcp=pose.tolist(),contact=[0.,.03,.01])
    variants=list(calibrated_contact_variants([row],Fingers(),np.zeros(7)))
    assert variants
    assert all(v['contact']==row['contact'] for v in variants)
    assert all(v['contact_backoff']<0 for v in variants)
    np.testing.assert_array_equal(row['tcp'],pose)

def test_missing_pad_calibration_is_an_explicit_error():
    class NoPads(Fingers):
        def boxes(self,joints):return []
    with pytest.raises(ValueError,match='Missing calibrated'):
        pad_depth_interval(NoPads(),np.zeros(7))
