"""Visualize saved Isaac physics states in MuJoCo EGL (no MuJoCo dynamics)."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml"


def model_with_scene(scene):
    xml = ET.parse(MODEL)
    xml.find("compiler").set("meshdir", str((MODEL.parent / "meshes").resolve()))
    world = xml.find("worldbody")
    ET.SubElement(world, "geom", name="support_rail", type="box", pos="0 .275 1.17",
                  size=".5 .025 .17", rgba=".13 .17 .21 1", contype="0", conaffinity="0")
    objects = (("pin_base", ".42 0 .4", ".1 .15 .4", ".15 .24 .28 1"),
               ("extract_part", ".42 0 .851", ".08 .1 .05", ".95 .38 .08 1"))
    if scene == "v1":
        objects = (("pin_base", ".48 0 .33", ".125 .15 .33", ".15 .24 .28 1"),
                   ("extract_part", ".42 0 .711", ".08 .1 .05", ".95 .38 .08 1"))
    for name, pos, size, color in objects:
        body = ET.SubElement(world, "body", name=name, pos=pos)
        ET.SubElement(body, "freejoint", name=name + "_free")
        ET.SubElement(body, "geom", name=name + "_geom", type="box", size=size,
                      mass="1", rgba=color, contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--still-at", type=float, default=None)
    parser.add_argument("--scene", choices=("v0", "v1"), default="v1")
    args = parser.parse_args()
    rollout = np.load(args.rollout, allow_pickle=False)
    model = model_with_scene(args.scene)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    names = rollout["joint_names"].tolist()
    joint_addresses = np.asarray([model.jnt_qposadr[model.joint(name).id] for name in names])
    base_address = model.jnt_qposadr[model.joint("pin_base_free").id]
    part_address = model.jnt_qposadr[model.joint("extract_part_free").id]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.17, 0., .7]
    camera.distance = 2.4
    camera.azimuth = 300
    camera.elevation = -16
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dt = float(np.median(np.diff(rollout["time"])))
    duration = float(rollout["time"][-1])
    times = ([min(args.still_at, duration)] if args.still_at is not None else
             np.arange(0, duration + 1e-8, 1 / args.fps))

    def frame_at(t):
        index = min(int(round(t / dt)), len(rollout["time"]) - 1)
        data.qpos[:3] = rollout["root_pos"][index]
        data.qpos[3:7] = rollout["root_quat"][index]
        data.qpos[joint_addresses] = rollout["joint_pos"][index]
        for address, key in ((base_address, "base"), (part_address, "part")):
            data.qpos[address:address + 3] = rollout[key + "_pos"][index]
            data.qpos[address + 3:address + 7] = rollout[key + "_quat"][index]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, args.width, 40), fill=(15, 23, 42))
        initial = rollout["part_pos"][0, 0] - rollout["base_pos"][0, 0]
        now = rollout["part_pos"][index, 0] - rollout["base_pos"][index, 0]
        draw.text((12, 11), f"Trained policy | Isaac physics | t={t:5.2f}s | "
                  f"part extraction {(initial - now)*100:.1f} cm", fill="white")
        return image

    if args.still_at is not None:
        frame_at(times[0]).save(args.output)
        print(f"STILL_READY {args.output}", flush=True)
    else:
        with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8) as writer:
            for index, t in enumerate(times):
                writer.append_data(np.asarray(frame_at(t)))
                if index % 150 == 0:
                    print(f"RENDER frame={index}/{len(times)}", flush=True)
        print(f"VIDEO_READY {args.output}", flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
