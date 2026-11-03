"""Project data mapping for native OpenPI; model and optimizer stay upstream."""
import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model,pi0_config
from openpi.policies import droid_policy
from openpi.training import config,weight_loaders,optimizer


@dataclasses.dataclass(frozen=True)
class ArmSkillInputs(transforms.DataTransformFn):
    def __call__(self,data):
        if 'observation.state' in data:
            state=np.asarray(data['observation.state'])
            native={'observation/joint_position':state[:7], 'observation/gripper_position':state[7:8],
                'observation/exterior_image_1_left':data['observation.images.navigation_cam'],
                'observation/wrist_image_left':data['observation.images.wrist_cam'], 'prompt':data['task']}
            if 'action' in data:native['actions']=np.asarray(data['action'])
        else:native=data
        return droid_policy.DroidInputs(model.ModelType.PI05)(native)


@dataclasses.dataclass(frozen=True)
class ArmSkillData(config.DataConfigFactory):
    def create(self,assets_dirs,model_config):
        mask=transforms.make_bool_mask(7,-1)
        return dataclasses.replace(self.create_base_config(assets_dirs,model_config),
            data_transforms=transforms.Group(inputs=[ArmSkillInputs(),transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask),droid_policy.DroidOutputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),action_sequence_keys=('action',))


def make_config(checkpoint,assets,output,exp_name='v1',steps=3750,batch=8,devices=1,resume=False):
    if steps < 1 or batch < 1:
        raise ValueError('steps and batch must be positive')
    # Preserve the original 1,000/30,000 warmup fraction for any chosen budget.
    schedule=optimizer.CosineDecaySchedule(warmup_steps=min(steps-1,round(steps/30)),decay_steps=steps)
    m=pi0_config.Pi0Config(pi05=True,action_horizon=15,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    return config.TrainConfig(name='pi05_onboard_set_table_arm',exp_name=exp_name,model=m,
        data=ArmSkillData(repo_id='plnguyen2908/cross-sim-data',assets=config.AssetsConfig(assets_dir=str(assets),asset_id='set_table_arm')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint)+'/params'),
        lr_schedule=schedule,freeze_filter=m.get_freeze_filter(),ema_decay=None,batch_size=batch,num_workers=0,num_train_steps=steps,
        fsdp_devices=devices,checkpoint_base_dir=str(output),save_interval=1000,keep_period=5000,
        resume=resume,project_name='cross-sim-fine-tuning',wandb_enabled=True)
