#!/usr/bin/env python3
"""Observe two iTHOR locations, swap their natural objects, then revisit both."""

import argparse
import json
import traceback
from pathlib import Path

import mujoco
import numpy as np

from cross_episode_sim.controller.navigation import (
    NavigationTransfer,
)
from cross_episode_sim.controller.navigation import (
    parse_args as parse_navigation_args,
)

EGG = "egg_45a3d68915c9f19164541ddf4da76856_1_0_0"
TOMATO = "tomato_8e23319fdf6cc27314eb709e9af2350a_1_0_0"
POTATO = "Irishpotato_60af36d8a721448d9e7a8608955f375b_1_0_0"
EGG_STANCE = np.array([-0.89, -1.02])
FRIDGE_STANCE = np.array([0.01, 2.08])


class DynamicRevisitCheck(NavigationTransfer):
    """Physical navigation around a logged, scripted between-visit intervention."""

    def __init__(self, args):
        self.extra_dynamic_prefixes = tuple(
            name.rsplit("_1_0_0", 1)[0] for name in (TOMATO, POTATO)
        )
        self.observation_object = EGG
        super().__init__(args)
        self.report.update(
            scope="visit two natural-object receptacles, change multiple objects, revisit both",
            protocol=(
                "explore A, explore B, logged unobserved multi-object intervention, "
                "revisit A, revisit B"
            ),
            dynamic_objects=[EGG, POTATO, TOMATO],
            observation_passes=[],
            interventions=[],
        )

    def navigation_undock(self):
        # The initial spawn is open floor; later departures are real receptacle undocks.
        if not self.report["observation_passes"]:
            return 0.0
        return super().navigation_undock()

    def gaze_target(self):
        names = (
            self.observation_object
            if isinstance(self.observation_object, tuple)
            else (self.observation_object,)
        )
        return np.mean([self.data.body(name).xpos for name in names], axis=0)

    def update_recording_cameras(self):
        super().update_recording_cameras()
        self.cameras[1].lookat[:] = self.gaze_target()

    def visit(self, pass_index, location, stance, expected_objects):
        expected_objects = tuple(expected_objects)
        self.observation_object = expected_objects
        positions = {
            name: np.asarray(self.data.body(name).xpos, dtype=float)
            for name in expected_objects
        }
        target = np.mean([position[:2] for position in positions.values()], axis=0)
        face = 0.0 if location == "fridge" else float(np.arctan2(*(target - stance)[::-1]))
        self.navigate(stance, carrying=False, face=face)
        labels = ", ".join(name.split("_")[0] for name in expected_objects)
        self.stage = f"observe pass {pass_index}: {location} / {labels}"
        self.tick(2.0)
        distances = {
            name: float(np.linalg.norm(self.base_xy() - position[:2]))
            for name, position in positions.items()
        }
        item = {
            "pass": pass_index,
            "location": location,
            "expected_objects": list(expected_objects),
            "object_positions": {name: position.tolist() for name, position in positions.items()},
            "base_position": self.base_xy().tolist(),
            "base_to_object_m": distances,
            "gaze_error_deg": self.gaze_error_deg(),
        }
        self.report["observation_passes"].append(item)
        self.record(**item)
        if max(distances.values()) > 1.5 or item["gaze_error_deg"] > 12.0:
            raise RuntimeError(f"Observation failed at {location}: {item}")

    def swap_objects(self):
        names = (EGG, POTATO, TOMATO)
        before = {name: self.data.body(name).xpos.copy() for name in names}

        from cross_episode_sim.controller.base import collision_mesh

        def mesh_bottom(name):
            prefix = name.rsplit("_1_0_0", 1)[0]
            bids = {
                b
                for b in range(self.model.nbody)
                if self.model.body(b).name.startswith(prefix)
            }
            vertices = collision_mesh(self.model, self.data, lambda b: b in bids)[0]
            return float(vertices[:, 2].min())

        bottoms = {name: mesh_bottom(name) for name in names}
        # Two separated slots on the tomato's original fridge shelf, and the
        # tomato takes the egg's vacated sink-counter slot.
        destinations = {
            EGG: before[TOMATO] + np.array([-0.10, 0.0, 0.0]),
            POTATO: before[TOMATO] + np.array([0.10, 0.0, 0.0]),
            TOMATO: before[EGG].copy(),
        }
        destination_bottoms = {
            EGG: bottoms[TOMATO],
            POTATO: bottoms[TOMATO],
            TOMATO: bottoms[EGG],
        }
        for name in names:
            joint = self.data.joint(name + "_jntfree_0")
            center_to_bottom = float(before[name][2] - bottoms[name])
            joint.qpos[:2] = destinations[name][:2]
            joint.qpos[2] = destination_bottoms[name] + center_to_bottom + 0.003
            joint.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.stage = "scripted between-pass multi-object change"
        self.tick(1.0)
        after = {name: self.data.body(name).xpos.tolist() for name in names}
        item = {
            "observed": False,
            "kind": "scripted between-step multi-object intervention",
            "moves": [
                {"object": EGG, "from": "sink counter", "to": "fridge"},
                {"object": POTATO, "from": "sink counter", "to": "fridge"},
                {"object": TOMATO, "from": "fridge", "to": "sink counter"},
            ],
            "before": {k: v.tolist() for k, v in before.items()},
            "after": after,
        }
        self.report["interventions"].append(item)
        self.record(**item)

    def execute(self):
        try:
            self.tick(0.5)
            self.visit(1, "sink counter", EGG_STANCE, (EGG, POTATO))
            self.visit(1, "fridge", FRIDGE_STANCE, (TOMATO,))
            self.swap_objects()
            self.visit(2, "sink counter", EGG_STANCE, (TOMATO,))
            self.visit(2, "fridge", FRIDGE_STANCE, (EGG, POTATO))
            self.report["success"] = True
            self.stage = "PASS: dynamic two-receptacle revisit"
            self.tick(1.0)
        except Exception as exc:
            self.report["success"] = False
            self.report["error"] = str(exc)
            self.report["traceback"] = traceback.format_exc()
            print(self.report["traceback"], flush=True)
            self.stage = "FAIL: " + self.stage
        finally:
            self.finalize_report()
            (self.output / "report.json").write_text(json.dumps(self.report, indent=2))
            (self.output / "trace.json").write_text(json.dumps(self.trace))
            try:
                if self.report["success"]:
                    self.render_deferred_video()
            finally:
                self.writer.close()
                self.head_writer.close()
                self.renderer.close()
        return 0 if self.report["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--video-speedup", type=float, default=5.0)
    cli = parser.parse_args()
    args = parse_navigation_args(
        [
            "--assets",
            str(cli.assets),
            "--output",
            str(cli.output),
            "--kitchen",
            "--native-object",
            "--soft-finger",
            "--object-name",
            EGG,
            "--start-base-x",
            ".01",
            "--start-base-y",
            ".68",
            "--start-base-yaw",
            "0",
            "--base-x",
            str(FRIDGE_STANCE[0]),
            "--base-y",
            str(FRIDGE_STANCE[1]),
            "--pickup-stance-x",
            str(EGG_STANCE[0]),
            "--pickup-stance-y",
            str(EGG_STANCE[1]),
            "--reverse-undock",
            ".3",
            "--base-servo-scale",
            "2",
            "--video-fps",
            str(cli.video_fps),
            "--video-speedup",
            str(cli.video_speedup),
            "--defer-video",
            "--gaze",
            "hybrid",
        ]
    )
    return DynamicRevisitCheck(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
