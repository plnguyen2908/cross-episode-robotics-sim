# Current π0.5 training memory accounting

Computed from JAX abstract initialization using the actual OpenPI model, freeze filter, parameter dtypes and AdamW state. No GPU weights were allocated and the active run was not interrupted.

| Component | GiB |
|---|---:|
| All model weights | 7.209 |
| AdamW optimizer state | 3.479 |
| Gradients (same-dtype estimate) | 1.740 |
| Subtotal | 12.428 |

Total parameters: 3,403,421,456. Trainable: 466,957,072 (vision: 414,803,696; LoRA: 49,987,584; remaining projections: 2,165,792). EMA is disabled.

The subtotal excludes activations, temporary buffers, runtime/executables, initialization peaks and allocator fragmentation. It is not a measured peak or proof that replicated training fits 24 GiB. JAX reservation shown by nvidia-smi is not live usage.

[Machine-readable counts](../runs/pi05_memory_accounting.json).


## GPU benchmark — November 2, 2026

Same upstream model, freeze filter, AdamW and batch size throughout. Vision remains
fully trainable; language/action expert use LoRA. No layers were frozen to fit.
Eight GPUs, global batch 8, one sample per GPU, zero data-loader workers.

| Configuration | Result | Median seconds/update including loading |
|---|---|---:|
| Previous 8-way FSDP training log | Running before user stopped it | ~4.97 |
| 2-way FSDP, four parallel groups | 12/12 updates passed | 3.77 |
| Full replica per GPU, platform allocator | 12/12 updates passed | 3.53 |

Benchmark medians exclude the first two updates and compilation. The old log is
an approximate reference, not an interleaved controlled benchmark. Full replication
reduced update time by about 29% relative to that reference. At 3.53 s/update,
30,000 updates would take about 29.4 hours, excluding startup/checkpoint overhead;
a short benchmark is not a guarantee of sustained throughput or convergence.

The platform run reached **17,016 MiB (~16.6 GiB) observed GPU memory**, sampled
with nvidia-smi every two seconds. This includes runtime allocations but can miss
brief peaks. The paired run's JAX peak live allocation was 12.4 GiB per GPU;
its allocator pool reached 21.2 GiB. These are different memory measurements.

Full replication with the default BFC allocator failed at the first update,
both at the default 75% limit and at 90%, while allocating a ~5.37 GiB buffer.
Changing only to `XLA_PYTHON_CLIENT_ALLOCATOR=platform` released initialization
buffers and allowed all eight full replicas to train. This does not change the
model, loss, gradients, dataset or action representation.

Defaults in `scripts/train_onboard_pi05.sh` now select platform allocation and
no preallocation. `--fsdp-devices 1` is the Python default; legacy `--devices`
is an alias. This flag means **devices per sharding group**, not total GPU count.
`CUDA_VISIBLE_DEVICES` selects the GPUs. All eight participate in synchronized
training of one model; this is not eight independent experiments.

Reproduce the short benchmark without checkpoints or W&B:

```bash
bash scripts/train_onboard_pi05.sh \
  --dataset runs/arm_skill_lerobot_v1/fine_tuning \
  --assets runs/arm_skill_lerobot_v1/assets \
  --fsdp-devices 1 --batch 8 --benchmark-steps 12 \
  --benchmark-output runs/pi05_replicated_benchmark.json
```

Reports: [paired](../runs/pi05_bench_pairs.json),
[full replica](../runs/pi05_bench_replicated_platform.json),
[sampled memory](../runs/pi05_bench_replicated_platform_gpu_memory.json),
[BFC failure](../runs/pi05_bench_full_replica_single.json).
The benchmark calls upstream `init_train_state` and `train_step` directly.
Type checking is disabled only while inspecting JAX's lowered executable because
its `ArgInfo` placeholders are not arrays; upstream source is unchanged.

Regular trainer verification: the new defaults completed two native training
updates and finalized an Orbax checkpoint successfully (exit 0), with W&B disabled
for the smoke test. Checkpoint: `runs/pi05_native_training_smoke/
pi05_onboard_set_table_arm/replica_checkpoint_test/1`. The full user training was
not launched. The native action-conversion regression test also passed.

Budget update: the default is now 3,750 updates at global batch 8 (30,000
sampled windows), with 125 warmup updates and cosine decay over 3,750 updates.
The 30,000-update timing projection above describes the previous budget.
The shorter budget projects roughly 3.7 hours at the measured update rate,
plus compilation and checkpoint overhead; it does not guarantee convergence.
