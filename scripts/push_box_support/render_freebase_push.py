"""Render matched free-base MuJoCo push/empty-space rollouts with EGL."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.push_box_support.mujoco_freebase_push import scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', type=Path, required=True)
    ap.add_argument('--params', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--azimuth', type=float, default=90.)
    ap.add_argument('--elevation', type=float, default=-12.)
    a = ap.parse_args()
    p = json.loads(a.params.read_text()); p = p.get('params', p)
    traces = [np.load(a.folder / file) for file in ('with_box_replay.npz', 'without_box_replay.npz')]
    models = []
    for flag in (True, False):
        model_path = scene(flag, p)
        try:
            models.append(mujoco.MjModel.from_xml_path(str(model_path)))
        finally:
            model_path.unlink(missing_ok=True)
    for model in models:
        model.vis.global_.offwidth = a.width
        model.vis.global_.offheight = a.height
    data = [mujoco.MjData(model) for model in models]
    renderers = [mujoco.Renderer(model, width=a.width, height=a.height) for model in models]
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [.54, 0, .62]; cam.distance = 3.15
    cam.azimuth = a.azimuth; cam.elevation = a.elevation
    duration = max(float(tr['time'][-1]) for tr in traces)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(a.output, fps=a.fps, codec='libx264', quality=8) as writer:
        for frame, t in enumerate(np.arange(0, duration + 1e-9, 1/a.fps)):
            image = Image.new('RGB', (2*a.width, a.height), '#111827')
            for j, (tr, model, d, renderer) in enumerate(zip(traces, models, data, renderers)):
                i = int(np.argmin(np.abs(tr['time'] - min(t, tr['time'][-1]))))
                d.qpos[:] = tr['qpos'][i]
                mujoco.mj_forward(model, d)
                renderer.update_scene(d, cam)
                view = Image.fromarray(renderer.render())
                image.paste(view, (j*a.width, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 2*a.width, 52), fill='#111827')
            bx = float(traces[0]['box_x'][int(np.argmin(np.abs(traces[0]['time'] - min(t, traces[0]['time'][-1]))))])
            initial = float(traces[0]['box_x'][0])
            draw.text((18, 15), f'BOX SUPPORT | shift {(bx-initial)*100:.1f} cm | t {t:.2f}s', font=font, fill='white')
            draw.text((a.width+18, 15), 'NO BOX | identical 29 joint targets', font=font, fill='white')
            writer.append_data(np.asarray(image))
            if frame % 60 == 0:
                print(f'RENDER {frame}', flush=True)
    for renderer in renderers:
        renderer.close()
    print(f'VIDEO_READY {a.output}', flush=True)


if __name__ == '__main__':
    main()
