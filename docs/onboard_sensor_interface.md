# Onboard sensor interface — step 1

Implemented on `feature/onboard-sensor-interface` in
`/nobackup/le/cross-episode-sensors`. This is the sensor boundary only, not SLAM,
DAAAM, navigation, ROS integration or VLA execution.

## Use

```python
from cross_episode_sim.sensors import OnboardSensorInterface

# model/data belong to the simulator bridge, not the downstream policy.
with OnboardSensorInterface(model, data, width=640, height=480) as sensors:
    packet = sensors.capture()
    nav = packet.cameras['navigation']
    wrist = packet.cameras['wrist']
    rgb, depth_m, K = nav.rgb, nav.depth_m, nav.K
    T_base_optical = nav.T_base_optical
    arm_q = packet.arm.position
    gripper_q = packet.gripper.position
    camera_info_fields = nav.camera_info()
```

Pass only `SensorPacket` to downstream consumers. Python private attributes are
not a security sandbox: giving a policy the bridge/model/data would bypass this
boundary. A ROS/process bridge should serialize only the packet fields.

## Output contract

- Existing `robot_0/navigation_cam` and `robot_0/wrist_cam`; no change to mounting,
  orientation, FOV, collision geometry or physical scene.
- Two uint8 RGB arrays and two float32 metric axial-depth arrays, with identical
  timestamps. Invalid or beyond-configured-range depth is NaN with a validity mask.
  Default usable range is 0–8 m. Colorized depth is diagnostic visualization only.
- Intrinsic K computed from existing vertical FOV and output resolution; square
  pixels, centered pixel coordinates, ideal pinhole/no distortion. Unsupported
  physical-sensor camera configuration fails explicitly rather than approximating.
- `T_base_optical` maps optical-frame points into `robot_0/base`; optical axes are
  x right, y down, z forward. Mount geometry and allowed arm encoder values are
  composed along the kinematic chain. No world-frame transform is used.
- Seven measured Franka arm joint positions/velocities and the two measured gripper
  driver-joint positions/velocities. Gripper values are radians, not aperture in
  metres or a fabricated fraction-closed sensor. Passive linkage states are omitted.
- No base x/y/yaw, global velocity, object IDs/poses, scene graph, segmentation
  labels, collision map, oracle actions or task-success fields.

Intrinsics and robot link geometry are calibration, not online localization.
Joint readback models ideal encoders; sensor noise and real hardware calibration
are not modeled here. Estimated base localization must be added by SLAM later.

Capture refreshes only MuJoCo kinematic/camera caches before rendering, because
`mj_step` can leave those caches one step behind qpos. It does not advance time
or update controls. World-coordinate render buffers remain private to rendering.
No ROS installation is required; `camera_info()` exposes ROS-compatible fields,
but publishing `/joint_states`, `CameraInfo` and TF is future integration work.

## Demonstration and verification

`scripts/demo_sensor_interface.py` replays a 20-second prefix of an existing
successful recording with collisions intact and records the interface output.
The oracle commands are used only by the demonstration harness to move sensors;
there is no navigation/planning policy in this test. No successful composite or
SLAM claim is implied. Videos are 10 fps at actual simulation duration, not sped up.

```bash
PYTHONPATH=. MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 \
/nobackup/le/cross-episode-native-molmospaces/.venv-native/bin/python \
 scripts/demo_sensor_interface.py \
 --source /nobackup/le/cross-episode-robotics-sim/runs/train_test_v1/test_two_rooms/batches/round_continuous/set_table_000000 \
 --output runs/sensor_interface_new
```

Artifacts: four-panel `sensor_demo.mp4`, PNG samples, `observation.npz` (raw metric
RGB-D, calibration, joint readings), `observation.json`, `report.json`.
Private qualification compares FK to simulator transforms while the robot moves,
checks invariance to global base pose, and verifies camera mount/FOV preservation.
Simulator ground truth is used only by these checks, never added to SensorPacket.

Tests: `tests/test_onboard_sensors.py` checks field allowlists, dimensions, metric
depth validity, intrinsics, FK at multiple arm postures, invariance under base
translation/rotation, and absence of stepping or physics/config mutations. It uses
a saved qualification scene and skips when that local scene is unavailable.

### Verified result

The `runs/sensor_interface_v2` qualification passed: 200 synchronized captures
across 10,000 physics steps (20 simulated seconds). Maximum camera-transform
matrix discrepancy was 1.44e-15. Base-pose invariance and unchanged camera
mount/FOV checks passed. All four sensor tests passed (7.73 seconds).

[Watch the demonstration](../runs/sensor_interface_v2/sensor_demo.mp4) ·
[Sample frame](../runs/sensor_interface_v2/frame_0151.png) ·
[Machine-readable report](../runs/sensor_interface_v2/report.json)

Next: [RGB-D tracking and reconstruction](rgbd_slam.md).
