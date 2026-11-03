"""Measure upstream OpenPI initialization/updates without changing its math."""
import functools
import json
import time
import jax
from tqdm import trange
from openpi.training import data_loader,sharding
from openpi.shared import array_typing as at


def benchmark(cfg,native,steps,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    def memory():return {str(d):d.memory_stats() for d in jax.devices()}
    report=dict(fsdp_devices=cfg.fsdp_devices,device_count=jax.device_count(),batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,steps=steps,status='initializing',trainable_parameters_unchanged=True)
    def save():output.write_text(json.dumps(report,indent=2,default=str))
    save()
    mesh=sharding.make_mesh(cfg.fsdp_devices)
    data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    iterator=iter(data_loader.create_data_loader(cfg,sharding=data_sharding,shuffle=True))
    t=time.monotonic();batch=next(iterator);jax.block_until_ready(batch);report['first_batch_seconds']=time.monotonic()-t
    train_rng,init_rng=jax.random.split(jax.random.key(cfg.seed))
    t=time.monotonic();state,state_sharding=native.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(state)
    report.update(initialization_seconds=time.monotonic()-t,memory_after_init=memory(),status='compiling');save()
    step=jax.jit(functools.partial(native.train_step,cfg),in_shardings=(replicated,state_sharding,data_sharding),
        out_shardings=(state_sharding,replicated),donate_argnums=(1,))
    t=time.monotonic()
    with sharding.set_mesh(mesh), at.disable_typechecking():
        compiled=step.lower(train_rng,state,batch).compile()
    report['compile_seconds']=time.monotonic()-t
    analysis=compiled.memory_analysis()
    report['compiled_memory']={k:getattr(analysis,k) for k in ('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes')}
    report.update(status='running',updates=[]);save()
    for i in trange(steps,desc='Native training benchmark'):
        t=time.monotonic()
        with sharding.set_mesh(mesh):state,info=compiled(train_rng,state,batch)
        jax.block_until_ready((state,info));compute=time.monotonic()-t
        t=time.monotonic();batch=next(iterator);jax.block_until_ready(batch);load=time.monotonic()-t
        row=dict(step=i,compute_seconds=compute,load_seconds=load,loss=float(info['loss']))
        report['updates'].append(row);report['memory']=memory();save();print(json.dumps(row),flush=True)
    report['status']='complete';save()
