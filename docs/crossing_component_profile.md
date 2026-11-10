# Crossing component cost isolation

November 7, 2026. GPU 0, `.venv-sift`, same saved successful run 18 states, 60 measured frames at 10 Hz (six simulated seconds), ten warm-up frames excluded. CUDA synchronized around timing blocks. All 60 tracking updates were accepted in each mode.

| Component | Previous total | Reduced fusion total | Previous mean per sensing cycle | Reduced mean per sensing cycle |
|---|---:|---:|---:|---:|
| RGB-D rendering/readback (two cameras) | 1.22s | 1.26s | 20.3 ms | 20.9 ms |
| Robot masks (two cameras) | 6.48s | 6.29s | 108.0 ms | 104.9 ms |
| RGB-D localization | 2.15s | 2.11s | 35.9 ms | 35.2 ms |
| Navigation-map fusion | 17.46s | 3.67s | 290.9 ms | 61.2 ms |
| Wrist-map fusion | 7.53s | 0.00s | 125.5 ms | 0.0 ms |
| Physics for 100 ms simulated time | 1.66s | 1.65s | 27.7 ms | 27.6 ms |

Totals: 36.50s versus 14.98s (2.44× component-throughput improvement, 59% reduction). Navigation fusion ran 60 versus 11 times; wrist fusion 60 versus zero. Means include skipped cycles, not per-fusion-call latency.

Localization is approximately 6% of previous measured cost. Both map fusions are approximately 68%; robot masks approximately 18%. Robot masking is now the largest measured component after reduced fusion.

Separate route-planning probe: 5.41 seconds, planned a route. This used the saved final observed map and original estimated starting pose, not the exact original pre-crossing map, so it is not a reconstruction of original planning latency.

## Scope and limitations

- This is offline profiling, not a new physical-success demonstration.
- The populated-map run starts with saved final occupancy, aligned with recorded SLAM estimates for diagnostic timing only. No simulator scene pose is used by the planner. This alignment must not be substituted for live localization.
- Camera timing excludes separately measured masks. Physics timing advances recorded actuator controls for 100 ms; it excludes controller command preparation.
- These totals do not account for all original wall time: topology/coverage updates, Python orchestration, logging/recording, controller overhead, and model RPC are outside this microbenchmark. They must not be presented as a complete attribution of the original nine-minute crossing.
- No video encoding occurs in this benchmark. Full live runtime and outcome with reduced fusion remain unverified.

## Reproduce

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-sift/bin/python scripts/profile_crossing_components.py \
  --recording runs/set_table_pipeline_development/closed_two_room_18_no_runtime_veto/merged_recording \
  --map-run runs/set_table_pipeline_development/closed_two_room_18_no_runtime_veto/door_exploration \
  --output runs/crossing_component_profile_populated
```

[Raw component timings](../runs/crossing_component_profile_populated/components.json)
[Route planning diagnostic](../runs/crossing_component_profile_v1/planning.json)
