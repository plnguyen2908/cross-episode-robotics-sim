# Set-table stationary-arm skills v1

LeRobot v3 fine-tuning/validation datasets for an instruction-conditioned VLA
with a separate navigation/base controller. Derived only from successful
set-table fine-tuning episodes; no evaluation/test episodes are included.

- **Actions:** 8D absolute arm joint targets (7 radians) + normalized gripper
  command (0=open, 1=closed). No base dimensions.
- **State:** 7 measured joint angles + measured normalized finger closure.
  Gripper state is NOT the previous commanded value.
- **Images:** separate navigation RGB and wrist RGB, 224×224. Original 640×480
  onboard views are rendered and padded with OpenPI's official resize helper.
  No fixed cameras, depth visualization, collision proxies or debug sites.
- **Rate:** 15 Hz, matched to the native DROID/PolaRiS checkpoint. The source
  recordings are at 500 Hz. This samples physical time; it does not stretch
  duration. Source-to-grid timestamp error is audited per segment.
- **Skills:** pick bowl, pick cup, place held bowl, place held cup. Instructions
  contain semantic categories/receptacles, never hidden IDs or coordinates.
- **Stationary filtering:** check every raw sample within each whole skill call.
  Measured base translation ≤2 cm, yaw span ≤0.03 rad; commanded translation
  ≤2 mm and yaw span ≤0.005 rad. This permits small tracking drift, not base moves.
- **Split:** 22 complete source episodes for fine-tuning, 3 for validation. Every
  skill from one source episode stays in one split. Validation is not used for
  normalization. `split_plan.json` and segment provenance make this auditable.

Each skill call is its own LeRobot episode, so action chunks cannot cross from
pick to navigate or into the next source episode. The official reader pads the
end of a chunk at the skill boundary using its normal episode behavior.

`fine_tuning/` and `validation/` are independently loadable v3 datasets. Their
standalone `info.json` uses the writer's default `train` row range; the semantic
split is identified by the parent directory and `meta/arm_skills.json`.

Layouts, object selections and source timing are in `supplementary/`, never in
model observations. Raw simulator paths in provenance identify source recordings
and are not required to load the exported data.

## Export validation

| Split | Source episodes | Skill segments | Frames | Minutes |
|---|---:|---:|---:|---:|
| Fine-tuning | 22 | 176 | 13,139 | 14.60 |
| Validation | 3 | 24 | 1,835 | 2.04 |

The four skill/category combinations are balanced: 44 clips each in fine-tuning,
6 each in validation. All segment endpoints decode with the official v3 loader.
Source-to-grid timestamp error is at most 2 ms. Native OpenPI transforms were
validated on the exported data (15×32 padded chunks, eight active dimensions);
training was not started. Three focused sampling/transform tests passed.

[Representative model inputs](../runs/arm_skill_lerobot_v1/preview.png)

## Load

```python
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo = 'plnguyen2908/cross-sim-data'
prefix = 'lerobot_atomic/set_table_arm_v1'
root = snapshot_download(repo, repo_type='dataset', allow_patterns=[prefix + '/**'])
dataset = LeRobotDataset(repo, root=f'{root}/{prefix}/fine_tuning',
    video_backend='pyav', tolerance_s=1e-3,
    delta_timestamps={'action': [i / 15 for i in range(15)]})
```

## Native OpenPI training

The project sensor branch provides `scripts/train_onboard_pi05.sh` with the
native DROID/PolaRiS joint-position checkpoint. It maps these features to the
upstream DROID input transform and trains relative arm targets internally;
upstream output transforms restore absolute targets. The pretrained shared
32-dimensional output is retained, with 8 active dimensions. No new base head.
The wrapper uses official LeRobot 0.4.3/datasets 4.1.1 for v3 data, while keeping
OpenPI model/optimizer code upstream. LoRA is configured for both language and
action-expert transformers through OpenPI's own freeze filter. Vision and the
remaining unfrozen modules follow that upstream filter. Default: 8 GPUs, global
batch 8, 3,750 updates, a full model replica per GPU (`--fsdp-devices 1`).
The shell uses the platform allocator to release temporary buffers. Twelve
benchmark updates passed; see [measured timing and memory](pi05_training_memory.md).
This is a starting configuration, not a convergence claim.

From `/nobackup/le/cross-episode-sensors`:

```bash
bash scripts/setup_onboard_pi05_training.sh
bash scripts/train_onboard_pi05.sh \
  --dataset runs/arm_skill_lerobot_v1/fine_tuning \
  --assets runs/arm_skill_lerobot_v1/assets --prepare-stats
# The command above only validates the data pipeline. Start training explicitly:
bash scripts/train_onboard_pi05.sh \
  --dataset runs/arm_skill_lerobot_v1/fine_tuning \
  --assets runs/arm_skill_lerobot_v1/assets --name v3 --train
```

Use a new experiment name for another run or `--resume` for the existing one.
The wrapper does not automatically evaluate the validation split during upstream
training; it remains separately available for checkpoint evaluation.

## Limits

Source demonstrations succeeded. The exported 15 Hz sample-and-hold actions have
not been physically requalified; do not equate source success with converted
controller replay or learned-policy success. These clips do not teach global
navigation or base-assisted manipulation. Camera format compatibility does not
establish equivalence to DROID/LIBERO camera calibration. Physics data generation
was not rerun, and no fine-tuning job was launched during this conversion.

Published: https://huggingface.co/datasets/plnguyen2908/cross-sim-data/tree/main/lerobot_atomic/set_table_arm_v1

Dataset commit: `0f2d9dd5d0d0e2e1c32c9846c28b83e3d5e4aa23`. Both remote split schemas/counts were downloaded and verified.


### Short initial fine-tuning budget

Default: 3,750 optimizer updates × global batch 8 = 30,000 sampled training
windows. This matches the sample count of 30,000 updates at batch 1, not its
optimizer dynamics. It is eight times less data exposure than the earlier
30,000-update/global-batch-8 configuration. The cosine schedule now spans the
selected `--steps` budget, with proportional warmup (125 updates by default).
Peak and final learning rates remain upstream defaults. Changing `--batch` does
not automatically change `--steps`; the launcher prints both and their product.
Training windows overlap; this count is not a count of unique trajectories.

An already running process keeps its old configuration. Start a new experiment
name (for example `v3`) to use the shorter schedule from the beginning. Resuming
an old long-budget checkpoint with the new schedule changes its learning-rate
history and is not equivalent to a fresh short-budget run.
