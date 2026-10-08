"""Export recorded episodes as a robomimic-style HDF5 dataset for policy training.

Each run directory written by a task (report.json + trace.json) becomes one
demo. Images are rendered by replaying the recorded simulator states in the
same scene the episode ran in, so no rendering is needed while generating
demonstrations.

Layout (compatible with robomimic / RoboCasa loaders):

    data/
      demo_0/
        actions            (T, 11) float32  absolute targets, see env.task_env
        obs/state          (T, 11) float32  base x, y, yaw, 7 arm joints, gripper
        obs/<camera>_image (T, H, W, 3) uint8
        states             (T, nq) float64  full MuJoCo qpos, for exact replay
      demo_0.attrs: num_samples, run, task, success, instruction
    data.attrs: env_args (JSON), total

Actions are the commands the controller sent when the trace recorded them
(`ctrl`); for older traces without commands, the next recorded robot state is
used as the target, which is what a position-controlled robot tracks.

    python -m cross_episode_sim.data.export runs/breakfast_* --output datasets/breakfast.hdf5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from cross_episode_sim.controller.house_scene import make_house
from cross_episode_sim.robot.embodiment import embodiment_for

DEFAULT_CAMERAS = ("robot_0/wrist_cam",)


def load_run(run):
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    trace_path = run / "trace_trimmed.json"
    if not trace_path.is_file():
        trace_path = run / "trace.json"
    trace = json.loads(trace_path.read_text())
    return report, trace


def build_model(report):
    arguments = report["arguments"]
    spawn = arguments["spawn"]
    model, data, _ = make_house(arguments["scene_xml"], arguments["dynamic_objects"], spawn[:2], spawn[2],
                                robot=report.get("robot", "franka_tidybot"))
    return model, data


class Layout:
    """Index bookkeeping for the policy-facing state and action vectors."""

    def __init__(self, model, robot):
        profile = embodiment_for(robot).profile
        ns = profile.namespace
        self.base_qpos = [model.jnt_qposadr[model.joint(ns + j).id] for j in profile.base_joints]
        self.arm_qpos = [model.jnt_qposadr[model.joint(ns + j).id] for j in profile.arm_joints]
        self.base_act = [model.actuator(ns + a).id for a in profile.base_actuators]
        self.arm_act = [model.actuator(ns + a).id for a in profile.arm_actuators]
        self.grip_act = model.actuator(ns + profile.gripper_actuator).id
        self.grip_range = (profile.gripper_open, profile.gripper_close)

    def gripper_fraction(self, command):
        opening, closing = self.grip_range
        return float(np.clip((command - opening) / (closing - opening), 0.0, 1.0))

    def state(self, qpos, grip_command):
        qpos = np.asarray(qpos)
        return np.concatenate([qpos[self.base_qpos], qpos[self.arm_qpos], [self.gripper_fraction(grip_command)]])

    def action(self, row, following):
        if "ctrl" in row:
            ctrl = np.asarray(row["ctrl"])
            return np.concatenate([ctrl[self.base_act], ctrl[self.arm_act], [self.gripper_fraction(ctrl[self.grip_act])]])
        target = np.asarray(following["qpos"])
        grip = following.get("gripper_command", row.get("gripper_command", self.grip_range[0]))
        return np.concatenate([target[self.base_qpos], target[self.arm_qpos], [self.gripper_fraction(grip)]])


def export(runs, output, cameras=DEFAULT_CAMERAS, image_size=(224, 224), stride=1):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    with h5py.File(output, "w") as file:
        group = file.create_group("data")
        total = 0
        env_args = None
        for index, run in enumerate(runs):
            report, trace = load_run(run)
            trace = trace[::stride]
            model, data = build_model(report)
            if len(trace[0]["qpos"]) != model.nq:
                raise ValueError(f"{run}: trace has {len(trace[0]['qpos'])} positions, model has {model.nq}")
            robot = report.get("robot", "franka_tidybot")
            layout = Layout(model, robot)
            if renderer is not None:
                renderer.close()
            renderer = mujoco.Renderer(model, height=image_size[0], width=image_size[1])
            steps = len(trace)
            demo = group.create_group(f"demo_{index}")
            states = demo.create_dataset("states", (steps, model.nq), np.float64)
            actions = demo.create_dataset("actions", (steps, 11), np.float32)
            obs = demo.create_group("obs")
            state_ds = obs.create_dataset("state", (steps, 11), np.float32)
            images = {c: obs.create_dataset(f"{c.split('/')[-1]}_image", (steps, *image_size, 3), np.uint8,
                                            chunks=(1, *image_size, 3), compression="gzip", compression_opts=4)
                      for c in cameras}
            for t, row in enumerate(trace):
                following = trace[min(t + 1, steps - 1)]
                data.qpos[:] = row["qpos"]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                grip = row["ctrl"][layout.grip_act] if "ctrl" in row else row.get("gripper_command", layout.grip_range[0])
                states[t] = data.qpos
                state_ds[t] = layout.state(row["qpos"], grip)
                actions[t] = layout.action(row, following)
                for camera, dataset in images.items():
                    renderer.update_scene(data, camera=camera)
                    dataset[t] = renderer.render()
            run_dir = Path(run)
            instruction = (run_dir / "instruction.txt").read_text().strip() if (run_dir / "instruction.txt").is_file() else ""
            demo.attrs.update(num_samples=steps, run=str(run_dir.resolve()), success=bool(report.get("success")),
                              task=report.get("composite_task", ""), instruction=instruction)
            total += steps
            env_args = env_args or dict(env_name=report.get("composite_task", ""), robot=robot,
                                        cameras=list(cameras), control_dt=0.04 * stride, image_size=list(image_size),
                                        action="absolute base pose (3), arm joints (7), gripper [0 open, 1 closed]")
        group.attrs["total"] = total
        group.attrs["env_args"] = json.dumps(env_args)
    if renderer is not None:
        renderer.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Episode output directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append", dest="cameras", help="Camera name (repeatable)")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth recorded frame (25 Hz / N)")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()
    runs = []
    for run in args.runs:
        report = json.loads((run / "report.json").read_text())
        if report.get("diagnostic_resume_from"):
            # Starts mid-task from a checkpoint: not a complete demonstration.
            print(f"skipping checkpoint-resumed run {run}")
        elif report.get("success") or args.include_failures:
            runs.append(run)
        else:
            print(f"skipping unsuccessful run {run}")
    path = export(runs, args.output, tuple(args.cameras or DEFAULT_CAMERAS), tuple(args.image_size), args.stride)
    print(f"wrote {len(runs)} demos to {path}")


if __name__ == "__main__":
    main()
