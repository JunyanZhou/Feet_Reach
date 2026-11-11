"""Render a recorded *Isaac physics policy rollout* with MuJoCo's EGL renderer.

MuJoCo reads the saved robot/object poses only for visualization. Task metrics
must come from eval_policy.py's original Isaac rollout, not this renderer.
"""
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


def make_model(scene="original", object_height=None):
    xml = ET.parse(MODEL)
    xml.find("compiler").set("meshdir", str((MODEL.parent / "meshes").resolve()))
    world = xml.find("worldbody")
    def box(name, pos, size, color):
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                      rgba=color, contype="0", conaffinity="0")
    if scene == "trained_motion":
        box("table_top", (.3,.45,1.34), (.4,.25,.025), ".58 .38 .21 1")
        box("table_top_extension", (.3,.10,1.34), (.4,.10,.025), ".58 .38 .21 1")
        box("table_side_apron", (0,.275,1.17), (.5,.025,.17), ".13 .17 .21 1")
        box("table_front_support", (0,.30,.5), (.035,.035,.5), ".13 .17 .21 1")
        box("table_back_support", (.69,.30,.62), (.025,.05,.62), ".13 .17 .21 1")
        box("table_far_support", (.69,.65,.62), (.025,.025,.62), ".13 .17 .21 1")
        box("retrieval_zone", (-.21,.09,.001), (.1,.1,.001), ".15 .62 .51 .55")
        height = object_height if object_height is not None else .11
        obj = ET.SubElement(world, "body", name="target_object", pos=f".08 .09 {height/2:.6g}")
        object_size = f".03 .03 {height/2:.6g}"
    else:
        box("table_top", (.78, 0, .745), (.5, .53, .025), ".58 .38 .21 1")
        for x, label_x in ((.35, "front"), (1.21, "back")):
            for y, label_y in ((-.46, "right"), (.46, "left")):
                box(f"table_leg_{label_x}_{label_y}", (x,y,.36), (.025,.025,.36), ".13 .17 .21 1")
        box("retrieval_zone", (.135,.12,.001), (.105,.12,.001), ".15 .62 .51 .55")
        height = object_height if object_height is not None else .08
        obj = ET.SubElement(world, "body", name="target_object", pos=f".445 .12 {height/2:.6g}")
        object_size = f".045 .045 {height/2:.6g}"
    ET.SubElement(obj, "freejoint", name="target_object_free")
    ET.SubElement(obj, "geom", name="target_object_geom", type="box",
                  size=object_size, mass=".22", rgba="1 .38 .055 1",
                  contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rollout", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--scene", choices=("original", "trained_motion"), default="original")
    p.add_argument("--object-height", type=float, default=None)
    p.add_argument("--still-at", type=float, default=None, help="Save a PNG at one time instead of a video")
    p.add_argument("--azimuth", type=float, default=315)
    p.add_argument("--elevation", type=float, default=-18)
    p.add_argument("--distance", type=float, default=None)
    args = p.parse_args()
    rollout = np.load(args.rollout, allow_pickle=False)
    model = make_model(args.scene, args.object_height)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    names = rollout["joint_names"].tolist()
    qadr = np.array([model.jnt_qposadr[model.joint(name).id] for name in names])
    obj_addr = model.jnt_qposadr[model.joint("target_object_free").id]
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [.04, .10, .68] if args.scene == "trained_motion" else [.38, .03, .55]
    cam.distance = args.distance or (2.3 if args.scene == "trained_motion" else 2.7)
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dt = float(np.median(np.diff(rollout["time"]))) if len(rollout["time"])>1 else .02
    duration = float(rollout["time"][-1])
    frame_times = [args.still_at] if args.still_at is not None else np.arange(0, duration + 1e-7, 1 / args.fps)
    if args.still_at is not None:
        t = min(args.still_at, duration)
        idx = min(int(round(t / dt)), len(rollout["time"]) - 1)
        data.qpos[:3] = rollout["root_pos"][idx]
        data.qpos[3:7] = rollout["root_quat"][idx]
        data.qpos[qadr] = rollout["joint_pos"][idx]
        data.qpos[obj_addr:obj_addr+3] = rollout["object_pos"][idx]
        data.qpos[obj_addr+3:obj_addr+7] = rollout["object_quat"][idx]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        Image.fromarray(renderer.render()).save(args.output)
        renderer.close()
        print(f"STILL_READY {args.output}", flush=True)
        return
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8) as writer:
        for frame, t in enumerate(frame_times):
            idx = min(int(round(t / dt)), len(rollout["time"]) - 1)
            data.qpos[:3] = rollout["root_pos"][idx]
            data.qpos[3:7] = rollout["root_quat"][idx]
            data.qpos[qadr] = rollout["joint_pos"][idx]
            data.qpos[obj_addr:obj_addr+3] = rollout["object_pos"][idx]
            data.qpos[obj_addr+3:obj_addr+7] = rollout["object_quat"][idx]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, args.width, 40), fill=(15, 23, 42))
            moved = float(rollout["object_pos"][0,0] - rollout["object_pos"][idx,0])
            draw.text((12, 11), f"Policy-controlled Isaac physics rollout | t={t:5.2f}s | object moved {moved*100:.1f} cm", fill="white")
            writer.append_data(np.asarray(image))
            if frame % 150 == 0:
                print(f"RENDER frame={frame}/{len(frame_times)}", flush=True)
    renderer.close()
    print(f"VIDEO_READY {args.output}", flush=True)


if __name__ == "__main__":
    main()
