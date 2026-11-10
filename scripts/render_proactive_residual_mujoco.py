#!/usr/bin/env python3
"""MuJoCo cross-simulator preview for proactive/gentle adapt students.

Uses the bundled frozen CEER export and a held-out ee_reach trajectory. This is
an adapter diagnostic, not an Isaac Lab evaluation or a sim-to-sim parity test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from mujoco_residual import ResidualSimulation, wrist_interface
from mujoco_whole_body_student import DEFAULT_MODEL, ROOT, Simulation

LOW_BUNDLE = ROOT / "deploy/mujoco_whole_body/ceer"
MANIFEST = ROOT / "artifacts/ee_reach/manifest.json"
FUTURE_STEPS = (0, 2, 4, 8, 16)


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_point = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.project = nn.Linear(64, 64)

    def forward(self, packed):
        cloud = packed.reshape(-1, 128, 4)
        valid = cloud[..., 3] > 0.5
        features = self.per_point(cloud[..., :3]).masked_fill(~valid[..., None], -1e6)
        pooled = features.amax(dim=-2)
        pooled = torch.where(valid.any(-1, keepdim=True), pooled, torch.zeros_like(pooled))
        return self.project(pooled)


def mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.Mish(),
        nn.Linear(512, 256), nn.LayerNorm(256), nn.Mish(), nn.Linear(256, output_dim),
    )


class HighPolicy:
    def __init__(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.task_name = state["cfg"].task.name
        if self.task_name not in ("G1ProactiveResidual", "G1GentleA"):
            raise ValueError(f"Unsupported task: {self.task_name}")
        if state["policy"]["last_phase"] != "adapt":
            raise ValueError("Expected an adapt student checkpoint")
        actor = state["policy"]["actor_student"]
        self.pointnet = PointNet().eval()
        self.pointnet.load_state_dict({
            k.removeprefix("module.0.module.0.module."): v for k, v in actor.items()
            if k.startswith("module.0.module.0.module.")
        }, strict=True)
        self.adapt = nn.Sequential(mlp(250, 128)[:-1], nn.Linear(256, 128)).eval()
        self.adapt.load_state_dict({
            k.removeprefix("module.1.module."): v for k, v in state["policy"]["adapt_module"].items()
            if k.startswith("module.1.module.")
        }, strict=True)
        self.actor = mlp(442, 3).eval()
        actor_features = {
            k.removeprefix("module.0.module.2.module."): v for k, v in actor.items()
            if k.startswith("module.0.module.2.module.")
        }
        actor_features.update({
            "6." + k.removeprefix("module.0.module.3.module.actor_mean."): v for k, v in actor.items()
            if k.startswith("module.0.module.3.module.actor_mean.")
        })
        self.actor.load_state_dict(actor_features, strict=True)
        env = state["env"]
        self.loc = env["transforms.6.loc"].float()
        self.scale = env["transforms.6.scale"].float()
        assert self.loc.shape == self.scale.shape == (250,) and torch.all(self.scale > 0)
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def __call__(self, public, geometry):
        obs = (torch.from_numpy(public.astype(np.float32))[None] - self.loc) / self.scale
        geom = self.pointnet(torch.from_numpy(geometry.astype(np.float32))[None])
        latent = self.adapt(obs)
        action = self.actor(torch.cat((obs, latent, geom), dim=-1))[0].numpy()
        if action.shape != (3,) or not np.isfinite(action).all():
            raise RuntimeError("High-level student produced a nonfinite action")
        return np.clip(action, -10., 10.)


def scene_model(with_wall, wall_x):
    if not with_wall:
        return DEFAULT_MODEL
    dest = ROOT / ".cache/mujoco_proactive_scene"
    dest.mkdir(parents=True, exist_ok=True)
    meshes = dest / "meshes"
    if not meshes.exists():
        meshes.symlink_to((DEFAULT_MODEL.parent / "meshes").resolve(), target_is_directory=True)
    tree = ET.parse(DEFAULT_MODEL)
    world = tree.find("worldbody")
    ET.SubElement(world, "geom", name="reach_wall", type="box",
                  pos=f"{wall_x + 0.025} -0.15 1.0", size="0.025 0.5 0.5",
                  friction="0.6 0.005 0.0001", rgba="0.30 0.50 0.75 0.32")
    path = dest / f"g1_wall_{wall_x:.3f}.xml"
    tree.write(path, encoding="unicode")
    return path


class HighEESimulation(ResidualSimulation):
    def __init__(self, high_policy, motion, manifest, model_path, wall_x, wall_present):
        self.high_policy = high_policy
        self.motion = motion
        self.manifest = manifest
        self.wall_x = wall_x
        self.wall_present = wall_present
        self.rng_cloud = np.random.default_rng(42)
        super().__init__(LOW_BUNDLE, model_path, physics_dt=.005)
        names = list(manifest["joint_names"])
        self.to_low = np.array([names.index(n) for n in self.meta["joint_names"]])
        self.hand_index = manifest["body_names"].index("left_hand_mimic")
        self.data.qpos[:3] = motion["root_pos"][0]
        root_xyzw = motion["root_rot"][0]
        self.data.qpos[3:7] = root_xyzw[[3, 0, 1, 2]]
        self.data.qpos[self.qadr] = motion["dof_pos"][0, self.to_low]
        mujoco.mj_forward(self.model, self.data)
        self.history[:] = self.data.qpos[self.qadr]
        self.high_actions = np.zeros((3, 3), dtype=np.float32)
        self.cloud_world = None
        self.cloud_valid = None
        self.frame_index = 0
        self.nominal_target = np.zeros(3)
        self.commanded_target = np.zeros(3)
        self.last_high_action = np.zeros(3)
        self.max_wall_force = 0.
        self.hand_wall_contact_count = 0
        self.wall_geom = self.model.geom("reach_wall").id if wall_present else -1
        self.hand_body = self.model.body("left_hand_mimic").id

    def _geometry(self):
        if not self.wall_present:
            return np.zeros(512, dtype=np.float32)
        if self.cloud_world is None or self.age % 2 == 0:
            points = self.rng_cloud.random((128, 3))
            points[:, 0] = self.wall_x
            points[:, 1] -= .35
            points[:, 2] += .5
            points += self.rng_cloud.normal(0., .003, size=points.shape)
            self.cloud_world = points
            self.cloud_valid = self.rng_cloud.random(128) > .1
        rotation = self.data.xmat[self.root].reshape(3, 3)
        cloud = np.zeros((128, 4), dtype=np.float32)
        cloud[:, :3] = (self.cloud_world - self.data.qpos[:3]) @ rotation
        cloud[:, :3] *= self.cloud_valid[:, None]
        cloud[:, 3] = self.cloud_valid
        return cloud.ravel()

    def _update_policy(self):
        t = max(0., self.data.time - .5)
        index = min(int(t / .02), len(self.motion["dof_pos"]) - 1)
        self.frame_index = index
        self.reference = self.motion["dof_pos"][index, self.to_low].astype(np.float64)
        target_world = self.motion["root_pos"][index] + self.motion["local_body_pos"][index, self.hand_index]
        rotation = self.data.xmat[self.root].reshape(3, 3)
        self.nominal_target = target_world
        target_b = rotation.T @ (target_world - self.data.qpos[:3])
        angular = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.root, angular, 1)
        public = Simulation.observation(self)
        high_obs = np.r_[target_b, angular[:3], public[3:6], self.history.ravel(),
                         self.low_actions.ravel(), self.high_actions.ravel()].astype(np.float32)
        assert high_obs.shape == (250,), high_obs.shape
        raw = self.high_policy(high_obs, self._geometry())
        self.last_high_action = raw
        self.high_actions[1:] = self.high_actions[:-1].copy()
        self.high_actions[0] = raw
        command_b = target_b - .15 * np.tanh(raw)
        self.commanded_target = self.data.qpos[:3] + rotation @ command_b
        wrists = wrist_interface(self.reference, self.meta["fk_edges"])
        wrists[:3] = command_b
        low_command = np.array([self.motion["root_pos"][index, 2], 0., 0., 1., 0., 30.])
        low_obs = np.r_[max(0., 1 - self.age / 25), low_command, wrists,
                        public[:151], self.low_actions.ravel()].astype(np.float32)
        assert low_obs.shape == (257,)
        order = self.meta["target_order"]
        future = np.stack([
            self.motion["dof_pos"][min(index + offset, len(self.motion["dof_pos"]) - 1), self.to_low][order]
            for offset in FUTURE_STEPS
        ]).astype(np.float32)
        with torch.inference_mode():
            self.low_action = self.low_policy(torch.from_numpy(low_obs[None]),
                                              torch.from_numpy(future[None]))[0].numpy()
        if not np.isfinite(self.low_action).all():
            raise RuntimeError("Frozen CEER produced a nonfinite action")
        self.low_actions[1:] = self.low_actions[:-1].copy()
        self.low_actions[0] = np.clip(self.low_action, -10., 10.)
        cfg = self.meta["low_action_config"]
        jitter = cfg.get("alpha_jit_scale")
        if jitter is not None:
            self.alpha = float(np.clip(self.alpha + self.rng.uniform(-jitter, jitter), *cfg["alpha_wide"]))
        self.age += 1

    def measure_contact(self):
        if not self.wall_present:
            return
        for j in range(self.data.ncon):
            contact = self.data.contact[j]
            if self.wall_geom not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, j, force)
            self.max_wall_force = max(self.max_wall_force, abs(float(force[0])))
            other = contact.geom2 if contact.geom1 == self.wall_geom else contact.geom1
            body_name = self.model.body(self.model.geom_bodyid[other]).name
            if body_name and ("left_wrist" in body_name or "left_hand" in body_name):
                self.hand_wall_contact_count += 1


def marker(scene, center, color, radius=.025):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius] * 3),
                        np.asarray(center, dtype=float), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", default="motions/val/ee_060_1.0.npz")
    parser.add_argument("--wall-x", type=float, default=.4)
    parser.add_argument("--no-wall", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=6.)
    args = parser.parse_args()
    high = HighPolicy(args.checkpoint)
    manifest = json.loads(MANIFEST.read_text())
    with np.load(ROOT / "artifacts/ee_reach" / args.clip) as source:
        motion = {key: source[key] for key in ("root_pos", "root_rot", "dof_pos", "local_body_pos")}
    sim = HighEESimulation(high, motion, manifest, scene_model(not args.no_wall, args.wall_x),
                           args.wall_x, not args.no_wall)
    sim.model.vis.global_.offwidth = args.width
    sim.model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(sim.model, width=args.width, height=args.height, max_geom=5000)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.azimuth, camera.elevation = 2.35, 125., -12.
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", macro_block_size=2)
    steps = round(args.seconds / sim.model.opt.timestep)
    frames = 0
    try:
        for step in range(steps):
            sim.step(np.zeros_like(sim.forces))
            sim.measure_contact()
            if step != round(frames / (args.fps * sim.model.opt.timestep)):
                continue
            camera.lookat[:] = sim.data.qpos[:3] + np.array([0., 0., .08])
            renderer.update_scene(sim.data, camera)
            marker(renderer.scene, sim.nominal_target, [1., .15, .15, 1.])
            marker(renderer.scene, sim.commanded_target, [1., .72, .1, 1.], .017)
            marker(renderer.scene, sim.data.xpos[sim.hand_body], [.15, 1., .4, 1.])
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, args.width, 61), fill=(8, 13, 20, 220))
            draw.text((10, 5), f"{high.task_name} | MuJoCo preview | {'wall' if sim.wall_present else 'no wall'}", font=font, fill="white")
            draw.text((10, 31), f"t={sim.data.time:.2f}s  target(red)  command(orange)  hand(green)", font=font, fill="white")
            writer.append_data(np.asarray(image))
            frames += 1
    finally:
        writer.close()
        renderer.close()
    error = float(np.linalg.norm(sim.data.xpos[sim.hand_body] - sim.nominal_target))
    result = dict(task=high.task_name, checkpoint=str(args.checkpoint), checkpoint_sha256=high.checkpoint_sha256,
                  low_level_checkpoint_sha256=sim.meta.get("low_level_sha256"),
                  simulator="MuJoCo", adapter="cross-simulator preview, not parity validated",
                  clip=args.clip, wall=sim.wall_present, wall_x=args.wall_x, seconds=float(sim.data.time),
                  frames=frames, first_fall_time=sim.first_fall, final_root_xyz=sim.data.qpos[:3].tolist(),
                  final_hand_target_error_m=error, max_wall_force_N=sim.max_wall_force,
                  hand_wall_contact_samples=sim.hand_wall_contact_count)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
