"""Stationary native-object pickup with a fixed close-up and measured grasp telemetry."""

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from cross_episode_sim.controller.base import (
    NS,
    FridgeTransfer,
    collision_mesh,
)
from cross_episode_sim.controller.navigation import (
    NavigationTransfer,
    parse_args,
)


class NativeGraspCheck(NavigationTransfer):
    def __init__(self, args):
        if not args.native_object or not (args.pickup_only or args.carry_only) or args.operate_door:
            raise ValueError(
                "Use --native-object with --pickup-only or --carry-only, without --operate-door"
            )
        super().__init__(args)
        self.initial_object_height = float(self.bread_pose()[2, 3])
        self.detail_target = self.bread_pose()[:3, 3].copy() + [0, 0, 0.06]
        self.cameras[1].azimuth = 90
        self.cameras[1].elevation = -12
        self.cameras[1].distance = 0.55
        self.detail_writer = imageio.get_writer(
            str(self.output / "grasp_closeup.mp4"), fps=args.video_fps
        )
        self.finger_meshes = []
        for name in ("ee_finger_r1", "ee_finger_r2"):
            bid = self.model.body(NS + name).id
            vertices, _ = collision_mesh(self.model, self.data, lambda b, target=bid: b == target)
            body = self.data.body(bid)
            local = (vertices - body.xpos) @ body.xmat.reshape(3, 3)
            self.finger_meshes.append((bid, local))
        self.report["scope"] = "stationary native object grasp, lift and three-second hold"

    def prepare_pickup(self):
        # The diagnostic starts at the counter; no departure/return navigation.
        self.record(diagnostic="stationary grasp; object retains its native pose")

    def update_recording_cameras(self):
        self.cameras[0].lookat[:] = [*self.base_xy(), 0.85]
        self.cameras[1].lookat[:] = self.detail_target

    def tick(self, seconds):
        self.update_recording_cameras()
        FridgeTransfer.tick(self, seconds)

    def record_camera_frames(self, frames):
        super().record_camera_frames(frames)
        contact = self.finger_object_contact()
        meshes = []
        centers = []
        for bid, local in self.finger_meshes:
            body = self.data.body(bid)
            meshes.append(local @ body.xmat.reshape(3, 3).T + body.xpos)
            centers.append(body.xpos.copy())
        axis = centers[1] - centers[0]
        axis /= np.linalg.norm(axis)
        gap = float((meshes[1] @ axis).min() - (meshes[0] @ axis).max())
        forces = [
            contact["normal_force_n"].get(NS + name, 0.0)
            for name in ("ee_finger_r1", "ee_finger_r2")
        ]
        rise = float(self.bread_pose()[2, 3] - self.initial_object_height)
        telemetry = {
            "time": float(self.data.time),
            "stage": self.stage,
            "collision_gap_m": gap,
            "finger_forces_n": forces,
            "object_rise_m": rise,
            "penetration_m": contact["depth_m"],
        }
        self.report.setdefault("grasp_telemetry", []).append(telemetry)
        frame = Image.fromarray(frames[1])
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 640, 82), fill="black")
        label = self.object_name.split("_")[0]
        stage = self.stage.replace("bread", label).replace("loaf", label)
        draw.text((10, 6), stage, fill="white")
        draw.text(
            (10, 25),
            f"Finger gap: {gap * 1000:.1f} mm | Force: {forces[0]:.1f} / {forces[1]:.1f} N",
            fill="white",
        )
        draw.text(
            (10, 44),
            f"Object rise: {rise * 1000:.1f} mm | Penetration: {contact['depth_m'] * 1000:.2f} mm",
            fill="white",
        )
        draw.text(
            (10, 63), "PASS requires >100 mm lift, no counter support, and a 3 s hold", fill="white"
        )
        self.detail_writer.append_data(np.asarray(frame))

    def run(self):
        try:
            return super().run()
        finally:
            self.detail_writer.close()


if __name__ == "__main__":
    raise SystemExit(NativeGraspCheck(parse_args()).run())
