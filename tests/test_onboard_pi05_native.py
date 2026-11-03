import numpy as np
import pytest
pytest.importorskip('openpi')
from cross_episode_sim.finetuning.onboard_pi05 import ArmSkillInputs
from openpi import transforms
from openpi.policies.droid_policy import DroidOutputs


def test_native_joint_delta_roundtrip_does_not_change_gripper_or_add_base():
    state=np.arange(8,dtype=np.float32)/10
    action=np.tile(state,(15,1));action[:,:7]+=.2;action[:,7]=1.
    raw={'observation.state':state,'action':action.copy(),'task':'Pick up a bowl.',
         'observation.images.navigation_cam':np.zeros((3,224,224),np.float32),
         'observation.images.wrist_cam':np.zeros((3,224,224),np.float32)}
    x=ArmSkillInputs()(raw);mask=transforms.make_bool_mask(7,-1)
    y=transforms.DeltaActions(mask)(x)
    assert np.allclose(y['actions'][:,:7],.2) and np.all(y['actions'][:,7]==1)
    z=transforms.AbsoluteActions(mask)(dict(state=state,actions=y['actions']))
    assert np.allclose(DroidOutputs()(z)['actions'],action)
    assert x['image']['base_0_rgb'].shape==(224,224,3)
    assert x['state'].shape==(8,) and not x['image_mask']['right_wrist_0_rgb']


def test_training_budget_and_schedule(tmp_path):
    from cross_episode_sim.finetuning.onboard_pi05 import make_config
    cfg=make_config(tmp_path,tmp_path,tmp_path)
    assert cfg.num_train_steps == 3750
    assert cfg.batch_size * cfg.num_train_steps == 30000
    assert cfg.lr_schedule.warmup_steps == 125
    assert cfg.lr_schedule.decay_steps == 3750
    for steps in (1, 2, 30000):
        cfg=make_config(tmp_path,tmp_path,tmp_path,steps=steps)
        assert cfg.lr_schedule.decay_steps == steps
        assert cfg.lr_schedule.warmup_steps < steps
        assert np.isfinite(float(cfg.lr_schedule.create()(0)))


def test_configured_output_pipeline_preserves_state_until_absolute_conversion(tmp_path):
    from cross_episode_sim.finetuning.onboard_pi05 import make_config
    cfg=make_config(tmp_path,tmp_path,tmp_path)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    state=np.arange(32,dtype=np.float32)/10
    predicted=np.zeros((15,32),np.float32)
    predicted[:,:7]=.2
    predicted[:,7]=.75
    out=transforms.compose(dc.data_transforms.outputs)({'state':state,'actions':predicted})
    assert out['actions'].shape==(15,8)
    np.testing.assert_allclose(out['actions'][:,:7],np.tile(state[:7]+.2,(15,1)))
    np.testing.assert_allclose(out['actions'][:,7],.75)
