"""Render recorded MuJoCo physics states with EGL, including the sliding joint."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scripts.foot_pin_bimanual.mujoco_task import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rollout', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--without-brace', action='store_true')
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--azimuth', type=float, default=305.)
    p.add_argument('--elevation', type=float, default=-17.)
    p.add_argument('--distance', type=float, default=2.25)
    p.add_argument('--lookat-z', type=float, default=.70)
    p.add_argument('--caption', type=str, default='')
    a = p.parse_args()
    params = json.loads(a.params.read_text())
    if 'params' in params:
        params = params['params']
    model_path = build_model(not a.without_brace, params)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = a.width
    model.vis.global_.offheight = a.height
    data = mujoco.MjData(model)
    trace = np.load(a.rollout)
    if 'qpos' not in trace:
        raise ValueError('Rollout has no full MuJoCo qpos; record with the current mujoco_task.py')
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [.34, 0, a.lookat_z]
    cam.distance = a.distance
    cam.azimuth = a.azimuth
    cam.elevation = a.elevation
    renderer = mujoco.Renderer(model, width=a.width, height=a.height)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    duration = float(trace['time'][-1])
    times = np.arange(0, duration + 1e-9, 1 / a.fps)
    with imageio.get_writer(a.output, fps=a.fps, codec='libx264', quality=8) as writer:
        for n, t in enumerate(times):
            i = int(np.argmin(np.abs(trace['time'] - t)))
            data.qpos[:] = trace['qpos'][i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, a.width, 50), fill=(15, 23, 42))
            support_name = 'foot plate' if params.get('support_mode') == 'plate' else 'foot brace'
            label = support_name.capitalize() if not a.without_brace else f'No {support_name}'
            part = 'drawer' if params.get('task_mode') == 'pull_drawer' else 'lid'
            suffix = f' | {a.caption}' if a.caption else ''
            drift = (f' | case drift={np.linalg.norm(trace["base_pos"][i]-trace["base_pos"][0])*100:.1f} cm'
                     if 'base_pos' in trace else '')
            d.text((14, 12), f'{label} | MuJoCo physics | t={t:.2f}s | {part}={trace["lid_open"][i]*100:.1f} cm{drift}{suffix}',
                   fill='white')
            writer.append_data(np.asarray(img))
            if n % 100 == 0:
                print(f'RENDER {n}/{len(times)}', flush=True)
    renderer.close()
    print(f'VIDEO_READY {a.output}', flush=True)


if __name__ == '__main__':
    main()
