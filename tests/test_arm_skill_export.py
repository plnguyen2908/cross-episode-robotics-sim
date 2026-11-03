import numpy as np
from scripts.export_arm_skill_lerobot import sample_segment,stationary


def test_sampling_never_crosses_call_or_invents_stretched_duration():
    times=np.arange(1000)*.002;ids,desired=sample_segment(times,.103,1.103,15)
    assert len(ids)==15 and times[ids].min()>=.103 and times[ids].max()<1.103
    assert np.max(times[ids]-desired)<.00200001


def test_stationary_filter_checks_whole_path_and_command():
    a=np.zeros((10,3));assert stationary(a,a)[0]
    b=a.copy();b[5,0]=.1;assert not stationary(b,a)[0]
    b=a.copy();b[5,2]=.1;assert not stationary(a,b)[0]
