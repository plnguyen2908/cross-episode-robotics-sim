# Two onboard RGB-D cameras, four demo panels, and variation controls

User decision: October 21, 2026. New composite/atomic demonstration rendering
use two physical cameras:

1. `robot_0/navigation_cam`: fixed to the mobile base through a rigid mast,
   looking forward with 30-degree downward pitch and 75-degree vertical FOV.
2. `robot_0/wrist_cam`: the existing authored Franka wrist camera. Its original
   placement and optics are retained.

The navigation mast is mounted at local base XY (-0.40, 0.00) m, directly behind the arm, with the
camera approximately 1.70 m above the base body origin. A solid 245 × 80 × 6 mm rear extension plate connects it to the base. A 12 mm radius pole
and camera housing have physical collision geometry; these are not invisible
collision-free props. They participate in native robot contact checks. New
full composite executions still need to qualify the additional mast clearance.
An embodiment without an authored wrist camera gets a hand-fixed camera at its
TCP site; that fallback is not a calibrated physical-camera model.

Composite video is a synchronized 2×2 grid: navigation RGB/depth above wrist RGB/depth. Depth uses a fixed 0–8 m visualization (purple near, yellow far; invalid black). The shared renderer exposes raw depth in metres separately; automatic per-frame depth persistence is not yet enabled for composites. Evaluation camera selection remains two RGB cameras. Raw observation
arrays contain no labels; video panels carry camera labels only. The legacy
`head_camera.mp4` recorder retains its filename and stores the last (wrist)
panel. Room cameras can remain in models for explicitly requested diagnostic
renders, but are not used by the default robot videos or evaluation observations.
Evaluation camera options reject world/room camera names.

Old videos are unchanged. Old saved models without the navigation camera must
be rebuilt before using the new renderer; a room camera is not silently used
as a substitute. Existing history videos must also be regenerated to build a
current four-panel dataset. Changing observations does not remove privileged
state from the oracle skill controller; learned policies receive the onboard
views through the observation interface.

## Additional optional YAML control groups

The original six groups remain supported. These three groups can now be added
under `controls` in any task's sampling YAML:

```yaml
controls:
  furniture_instances:
    scene.furniture_set:
      choices: [baseline, compact_chairs]
  wall_color:
    scene.appearance.wall_rgb:
      choices: [[0.9, 0.9, 0.85], [0.65, 0.75, 0.85]]
  object_color:
    scene.appearance.object_rgb:
      choices: [[0.8, 0.3, 0.2], [0.2, 0.5, 0.8]]
```

Set `randomize_controls.max_active` to the number of groups you want eligible
for simultaneous selection (up to nine when all six original and three new
ones are defined). RGB values are in [0,1]. Each color group currently applies
one sampled color to its eligible geometries, not an independent color per
object. Wall/object color changes set solid RGBA; floor/furniture tints retain texture maps through cloned materials. Hidden geometry remains hidden. Shape, mass, friction, and collision properties stay unchanged.
Changed geometry names are recorded in `task_manifest.json`.

Wall colors target structural north/south/east/west wall bodies. Object colors
target task objects, excluding baskets and bindings with explicit semantic
color metadata. This avoids silently breaking color-conditioned instructions.
The compact-chair set changes the living-room chair widths, depths and backs;
existing room-bound, overlap and doorway-corridor checks remain active. It is
not an arbitrary furniture asset library.

## Room-layout extension

An optional room-layout control now extends the three-room core to 3–8 rooms,
with chain/branch arrangements and physical doors. See
[room-layout variations](room_layout_variations.md) for configuration, previews,
and validation limits. Existing task recipes still use their original supports.

## Validation

See `tests/test_onboard_cameras.py`, `tests/test_appearance.py`,
`tests/test_living_furniture.py`, and `tests/test_variation_sampling.py`.
Static compilation of the actual Franka model found no mast contacts at its
initial posture. This does not certify the mast across every manipulation.


Static default-tuck previews with the raised mast (the external view is diagnostic only):

- [Two onboard views](../runs/onboard_camera_centered_mast_preview/onboard_views.png)
- [Navigation mast on the robot](../runs/onboard_camera_centered_mast_preview/mast_debug.png)

The actual initial scene and standard travel posture had no mast contacts.
The focused camera, appearance, sampler, furniture and embodiment checks passed
(28 tests). Full composite executions with the mast remain unqualified.

The default Franka tucked arm has a conservative geometry top of 1.532 m.
The raised navigation camera is approximately 1.705 m above the floor,
about 17 cm higher. Preview verification uses the configured travel posture.

Current expanded controls and 35-demo schedule: [pipeline diagram](composite_variation_pipeline.md#seven-case--35-demo-generation-default).

### Camera mount regression (October 21, 2026)

The original -0.20 m mount cleared the initial tuck but intersected the elbow
during previously successful composite motions. It is superseded by the -0.40 m
rear mount. The planner also used an overly coarse AABB cover for the thin pole:
89 mm spheres around a 12 mm pole. Capsule export now uses a conservative axial
sphere cover (under 14 mm radius for this pole), retaining all physical geometry
and collision checks. Static initial-pose checks alone are insufficient.

`python scripts/check_camera_mast_clearance.py --source RUN --model MODEL.mjb
--output CHECK.json` checks every saved pose against the physical mount. This is
a geometry regression test, not a claim of successful dynamic execution.
See [the regression record](camera_mast_regression_2026_10_21.md) for physical rerun results.

## Wrist RGB-D and re-rendering saved episodes

The existing wrist camera now exposes `wrist_depth_m` as well as `wrist_rgb`.
Navigation still exposes `navigation_rgb` and `navigation_depth_m`. Both depth
arrays are metric MuJoCo depth, rendered from the same state as the RGB images.
This adds no geometry, mass, or policy change; it models ideal depth, not a noisy
real sensor. Model-specific VLA adapters must explicitly select these modalities.

New videos use navigation RGB/depth on top and wrist RGB/depth below. Existing
models containing both cameras and saved `trace.json` can be re-rendered without
planning or executing physics again:

```bash
python scripts/render_onboard_rgbd.py runs/EPISODE --save-depth
```

This creates a new `onboard_rgbd/` folder with `demo.mp4`, `preview.png`, timestamps,
metadata, and (with `--save-depth`) float32 `navigation_depth_m.npy` and
`wrist_depth_m.npy`. Raw depth arrays can be large; omit the flag for video only.
Use `--output` for another new destination; existing output is never overwritten.
Rendering uses saved poses at original simulation timing, without stepping physics.
An MP4 alone is insufficient to recover metric depth. Successful episodes skipped
by batch resume retain their old videos until explicitly re-rendered.
