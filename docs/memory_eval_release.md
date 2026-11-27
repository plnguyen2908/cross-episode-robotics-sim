# Memory evaluation release: status and checklist

2026-11-27. This covers the GPT/Qwen + memory evaluation described in
the memory-eval handoff notes (`docs/memory_eval_handoff_2026-11-26.md` in the eval
trees; not on main yet). Only this checklist is on main so far: no memory code has
been pushed and no memory data has been uploaded.

**Bottom line:** a collaborator cannot reproduce the memory evaluation from
GitHub main and the Hugging Face dataset today. The code is not on main, the
memory data is not on Hugging Face, several paths point at our disks, and the
memory pipeline has never run in the self-contained release environment. Close
the gaps below, then pass the clean-room test (section 5) before anything goes
public.

## 1. What a collaborator already gets

From GitHub main (`88c35d6`) and the public dataset `plnguyen2908/cross-episodic-sim`:

| Piece | Where it comes from |
|---|---|
| External repos: DAAAM, daaam-boxmot, M2T2, GraspGen-X, MolmoSpaces, robosuite, robocasa | `scripts/setup_eval_env.sh` clones them into `external/` |
| Venvs `.venv-sift`, `.venv-m2t2`, `.venv-graspgenx` | `scripts/setup_eval_env.sh` |
| FastSAM, M2T2, GraspGen-X checkpoints; Qwen3-VL-8B and Molmo2-4B weights | `scripts/setup_eval_env.sh` (~62 GB per full install) |
| Test episodes and scene assets | `scripts/prepare_eval_data.py` (HF shards + `eval/eval_scene_assets.tar.zst`) |
| Model services and endpoints file | `scripts/start_eval_services.sh` |
| Slurm batch, results, GPT example | `scripts/slurm/household_eval.sbatch`, `collect_results.py`, `run_gpt_replan_example.sh` |
| Frozen memory lists | HF `memory/` (`FROZEN.json`, `README.md`, `episode_memory.json`, `index.json`, `lists/`) |

**Not needed for the memory eval:**
- OneRING: it runs in `"onering":"scan"` mode, so no model or venv is loaded.
- Hydra, TARE, cap-x, openpi/VLA, roboicl: the code references them, but they are not on the memory run path. The clean-room test should confirm this.

## 2. Hugging Face additions (`memory/` on the public dataset)

| Data | Local source | Size |
|---|---|---|
| Visual maps: `merged/`, `merged_tour/`, `episodes/<id>/{calibration.json, keyframes.npz}`, `memory_index_dev2/`, `route_memory/` | `runs/all_open_fixed_v4/memory/visual/` | ~0.9 GB |
| History actions: 500 × `memory.json` + `commands.npz` + `DONE.json` + manifest | `/nobackup/le/cross-episode-qwen-capx/runs/history_language_action_v4/actions` | 0.9 GB |
| History streams: `plan.json`, `episodes/` (1.6 GB), `streams/` (13 GB: `sensors.h5` with depth, robot mask, RGB, time, retained frame indices; `actions.json`) | `/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4` | ~15 GB |
| Text memory: trees + language | `/nobackup/le/cross-episode-qwen-capx/runs/task_language_memory_v4` | ~4 MB |
| **New** history-log pack: `config.json`, `skill_calls.json`, `instruction.txt` of the 500 history episodes | Today these exist only inside the 183 GB fine-tuning shards | a few MB |

Total: about 17 GB.

**Symlinks:** `visual_tour_root/` holds two symlinks (`episodes → ../visual/episodes`, `merged → ../visual/merged_tour`). Hugging Face does not store symlinks, so the download step has to recreate them.

**Keep out of Hugging Face:**
- `memory/visual_tours/` (9.1 GB). It contains `private_start_pose.npy`, which is simulator truth.
- `private_pose_audit.json` in every `visual/episodes/*`.
- `memory/visual/streams/` (817 MB). Nothing reads it.
- The task-language `locations` chunks (2.2 GB).
- From `balanced_v4`: `logs/`, `progress`, `merged`.
- Superseded folders.

**Metadata to update:**
- `memory/README.md`: it still says the visual maps will be "published separately".
- The dataset README: add the new `memory/` layout and sizes.
- `memory/episode_memory.json`: make paths relative to the dataset root.

## 3. GitHub main

Comparing the GPT tree (`cross-episode-gptmem-eval4`) with `origin/main`:

| | Files |
|---|---|
| Identical | 1,271 |
| Differ | 41: 28 changed only in the tree, 13 changed on both sides |
| Only in the tree | 226 |
| Only on main | 5 (the release scripts; keep them) |

Of the 13 changed on both sides, 7 merge cleanly:
- `sensor_set_table`
- `planning_contract`
- `qwen_zero_shot_plan`
- `skill_arguments`
- `setup_eval_env.sh`
- `test_qwen_zero_shot_plan`
- `docs/qwen_zero_shot_open_loop`

The other 6 conflict and need a hand merge:

| File | Conflicts | Rule |
|---|---|---|
| `evaluation/sensor_household.py` | 16 | |
| `scripts/eval_household.sh` | 6 | Keep main's endpoints-file handling; the tree hard-codes ports 51500–51586 |
| `scripts/eval_households.py` | 6 | Add the `--qwen-memory` preset on top of main |
| `evaluation/articulated_scene.py` | 2 | Keep main's per-task grip force |
| `scripts/run_sensor_household.py` | 2 | |
| `scripts/run_target_conditioned_household.py` | 1 | |

**New files the memory pipeline needs:**
- `evaluation/`: `episode_memory`, `text_memory`, `search_places`, `memory_plan`, `plan_fixer`
- `mapping/`: `action_memory`, `history_localization`, `history_routes`, `room_geometry_localization`
- `navigation/`: `memory_navigation`, `memory_object_match`, `visual_relocalization`, `live_route_memory`, `fixture_transfer`
- `scripts/test_nav_replay.py`, `scripts/test_fixture_memory.py`, `scripts/test_manip_replay.py`. Despite the names, the pipeline imports these at run time.
- `scripts/eval_gpt_memory_test.sh`, `scripts/eval_qwen_memory_test.sh`, the check scripts, the memory build scripts, their tests, and the handoff doc.

**Not part of this release:** about 150 other tree-only files are separate work (VLA, pi05, SmolVLA, LeRobot, datagen v4, 59 reports in `docs/reports`). They stay out unless you decide otherwise.

**Secret scan:** clean (no `sk-`/`hf_` tokens, no email addresses). The OpenAI key is read only from `OPENAI_API_KEY` in the environment.

## 4. Gaps that block a collaborator

1. **The memory pipeline has never run in the release environment.**
   - Our episodes run in the layered `cross-episode-sensors/.venv-sift`. Its `robot_dependencies.pth` adds the site-packages of `.venv-slam`, `cross-episode-native-molmospaces/.venv-native` and a home-directory `sim` venv.
   - A running episode loads jax, warp, triton, pyarrow, lmdb, numba, embreex, shapely, manifold3d, quadprog and torch from them.
   - The memory modules' own imports (h5py, scipy, cv2, mujoco, numpy) are in `requirements-eval-sift.txt`, but that has not been proven by a run.
2. **Hard-coded data paths.** Change each to a default relative to the dataset root, with an environment override:

   | Constant | File | Points at |
   |---|---|---|
   | `DEFAULT_ROOT` | `evaluation/text_memory.py` | `/nobackup/le/cross-episode-qwen-capx/runs/task_language_memory_v4` |
   | `VISUAL_ROOT` | `evaluation/episode_memory.py` | `/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual` |
   | `INDEX_DIR` | `navigation/fixture_transfer.py` | `.../memory/visual/memory_index_dev2` |
   | `ACTIONS` | `scripts/test_nav_replay.py` | `/nobackup/le/cross-episode-qwen-capx/runs/history_language_action_v4/actions` |
   | `HISTORY_ROOT` | `scripts/test_nav_replay.py` | `/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4` (`MEMORY_HISTORY_ROOT` overrides it) |
   | `visual_memory`, `memory_index` | `scripts/run_sensor_household.py` | derived from `VISUAL_ROOT` |
   | `--dataset`, output folder | `eval_gpt_memory_test.sh`, `eval_qwen_memory_test.sh` | `/nobackup/le/...all_open_fixed_v4`, `/nobackup3/...` |

3. **M2T2 checkpoint path is already broken on main.** It is hard-coded as `/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/.../m2t2.pth` in four files:
   - `runtime/fixture_controls.py:110`
   - `runtime/appliance_controls.py:83`
   - `control/live_door_open.py:14`
   - `runtime/episode.py:48`

   `sensor_household.py` reaches `LiveFixtureControls`. I have not checked whether fixture tasks load this file or go through the M2T2 service. If they load it, the already-released baseline also fails on fixture tasks, and the path should come from the checkpoint location `setup_eval_env.sh` writes.
4. **No data download for memory.** `prepare_eval_data.py` fetches only the test splits. It needs a `--memory` option that:
   - downloads `memory/` and the history-log pack;
   - recreates the `visual_tour_root` symlinks;
   - checks that the files are complete.
5. **No single entry point.** Add a README section that an agent can follow:

   ```
   setup_eval_env.sh → prepare_eval_data.py --memory → source eval_env.sh
     → start_eval_services.sh → one memory episode
   ```

   It should also cover the GPT variant (`OPENAI_API_KEY`), the limits (45 wall min, 20 sim min, 100 calls) and the GPU caveats (run the EGL check on every episode GPU; one of our GPUs failed it).

## 5. Clean-room test before anything goes public

1. Build the branch: merge the tree onto `origin/main` (section 3), fix the paths (section 4.2, 4.3), and add `prepare_eval_data.py --memory`.
2. Fresh clone into a new folder on `/nobackup3` (460 GB free; `/nobackup` has 78 GB and must keep 40 GB).
3. Run `setup_eval_env.sh`, then `prepare_eval_data.py --memory`. Read the memory files from a local staging copy laid out exactly as they would be on Hugging Face, so nothing is uploaded yet.
4. Run two memory episodes: set table, and prepare meal (it covers fixtures and the microwave). Run them under `strace -f -e trace=open,openat`, using the GPT planner and the Qwen planner.
5. Every file opened outside the fresh clone and its data folder (our `/nobackup*` disks or the home directory) is something a collaborator would be missing. Fix it and rerun until the list is empty.
6. Only then: upload `memory/` to Hugging Face and push the branch to main, both after maintainer approval (the dataset and the repo are public).

A baseline-only clean-room check passed on 2026-11-21: environment install and episode reload under strace, with no files opened outside `/nobackup3`. That check did not run any episode with memory.

Expect several hours, mostly the environment build and the ~36 GB model download.

## 6. Open decisions (maintainers)

- Should the past-plan fallback (`MEMORY_PLAN_FALLBACK`) get a switch to turn it off? The handoff eval uses `TEXT_MEMORY_STYLE=procedure`, not `past_plan`.
- Do the ~150 unrelated tree-only files (VLA/LeRobot/datagen/reports) go to main separately, or stay local?
- Should the 13 GB history streams be one HF folder or per-house shards? Per-house shards would let a collaborator fetch only the test houses they run.

## 7. Interface the README already documents (must exist when the branch lands)

The README's eval section describes the memory setting as it will work after the merge. The
branch has to provide exactly these, and the clean-room test (section 5) has to use them:

| README promise | Implementation needed |
|---|---|
| `prepare_eval_data.py --root $DATA --memory` | download the layout of section 8, recreate the `visual_tour_root/` links, check every file; never fetch the fine-tuning shards |
| `eval_households.py --memory` | today's `--qwen-memory` preset (text memory auto, procedure style, memory navigation + skip explore) without forcing the planner: works with `--planner gpt_replan` and `qwen_replan` (keep `--qwen-memory` as an alias) |
| `--memory-limits` | 45 min wall, 20 simulated min, 100 calls |
| `eval_household.sh <episode> --memory` | the same options for one episode, using main's endpoints file (no fixed ports) |
| sbatch `MEMORY` (default `1`), `PER_TASK=N` | `MEMORY=1` passes `--memory`; `MEMORY=0` runs zero-shot (+ `--memory-limits` unless `LIMITS` is set) and a first-N-per-task selection |
| every memory path relative to `--dataset`, `MEMORY_ROOT` override | replaces `DEFAULT_ROOT`, `VISUAL_ROOT`, `INDEX_DIR`, `ACTIONS`, `HISTORY_ROOT` (section 4.2) |
| launcher stops with "memory not found" | check each same-layout episode's history list and the memory folders before starting |
| batch remembers its setting | already true (`run_info.json` check in `eval_households.py`) |
| log tags `TEXT_MEMORY`, `MEMORY_HANDOVER`, `MEMORY_MOVE`, `MEMORY_FIXTURE`, `MEMORY_PICK`, `MEMORY_PLACE`, `MEMORY_*_FALLBACK`, `MEMORY_INDEX_MISSING` | already printed by the eval tree |

## 8. Where the memory code reads each file: today vs. after the fix

Checked 2026-11-27 in the GPT tree. Today the memory is spread over **three places**: the dataset's
`memory/` folder, the dataset's `fine_tuning/` folder, and two run folders of another worktree
(`cross-episode-qwen-capx/runs/...`). Only the first is found from `--dataset`.

| Data | Read by | How the code finds it today | Collaborator today | After the fix (README layout) |
|---|---|---|---|---|
| `episode_memory.json`, `lists/` | `episode_memory.py`, `search_places.py`, `memory_plan.py` | walks up from the episode folder to `<root>/memory/` | works, if `memory/` is downloaded (already on HF; `prepare_eval_data.py` does not fetch it yet) | `$DATA/memory/` |
| History `skill_calls.json`, `instruction.txt` (500 episodes; 180 are `round_v3_import`, 141 of their folders are links to v3) | `search_places.history_starts` (every memory episode: this house's history, or all houses' history on a new layout), `memory_plan.py` | `<root>/` + the relative path in the list (`fine_tuning/batches/...`) | **crash** (`FileNotFoundError`, no fallback): only inside the 183 GB fine-tuning shards | `$DATA/fine_tuning/batches/...` from a small pack (links dereferenced); no code change |
| Visual map | `run_sensor_household.py` → `episode_memory.resolve` | the path named in `episode_memory.json` (`memory/visual/merged/<house>_<task>`) **does not exist even here**; falls back to absolute `VISUAL_ROOT/../visual_tour_root` | **broken** | `$DATA/memory/visual_tour_root/merged/<house12>`; fix the names in `episode_memory.json` |
| Memory index | `run_sensor_household.py` | absolute `VISUAL_ROOT/memory_index_dev2` | **broken** | `$DATA/memory/visual/memory_index_dev2` |
| Fixture-transfer index | `fixture_transfer.py` | absolute `INDEX_DIR` | **broken** | same folder, derived from the root |
| Recorded commands | `memory_navigation.py`, `memory_object_match.py`, `fixture_transfer.py` via `scripts/test_nav_replay.ACTIONS` | absolute, `cross-episode-qwen-capx/runs/history_language_action_v4/actions` | **broken** | `$DATA/memory/actions` |
| History streams | `memory_navigation.py` via `test_nav_replay.HISTORY_ROOT` | absolute, `cross-episode-qwen-capx/runs/visual_memory_balanced_v4` (`MEMORY_HISTORY_ROOT` overrides) | **broken** unless the variable is set | `$DATA/memory/history_streams` |
| Text memory (`trees/` 1.7 MB, `language/` 2.0 MB) | `text_memory.py`, `episode_memory.resolve` | absolute `DEFAULT_ROOT`, `cross-episode-qwen-capx/runs/task_language_memory_v4` | **broken** | `$DATA/memory/text` |
| `test_nav_replay.MEM` | `test_nav_replay.py` (`visual_tour_root`) | absolute | **broken** | `$DATA/memory` |

Already fine: the HF test shards store the 83 v3-linked test episodes as real folders under
`test_same_map/batches/round_v3_import/`, so `episode_memory.json` matches them (all 200 test
entries are in the manifest).

Fix: one function, `memory_root(source)` = the `memory/` folder found from the episode (or
`MEMORY_ROOT`), and every constant above becomes a subfolder of it. Then the clean-room test
(section 5) must show no file opened outside the clone and `$DATA`.
