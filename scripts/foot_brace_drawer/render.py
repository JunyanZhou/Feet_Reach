#!/usr/bin/env python3
"""Render the saved foot-braced drawer replay as a clean MP4."""
import argparse
import os
import sys
from pathlib import Path
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image
from make_demo import OUT, set_robot


def render(out, view='overview', width=1920, height=1080, fps=60, stills=False):
    motion = dict(np.load(out / 'motion.npz'))
    sim = dict(np.load(out / 'simulation.npz'))
    model = mujoco.MjModel.from_xml_path(str(out / 'scene.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.13, 0, .58]
    if view == 'detail':
        camera.lookat[:] = [.13, 0, .63]
        camera.distance = 2.40
        camera.azimuth = 90
        camera.elevation = -8
    else:
        camera.distance = 2.20
        camera.azimuth = 315
        camera.elevation = -25
    path = out / (view + '_clean.mp4')
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=9,
                                macro_block_size=1, ffmpeg_params=['-threads', '4'])
    nframes = round(float(motion['time'][-1]) * fps) + 1
    shots = [0, 1.7, 2.6, 3.7, 4.8, 5.8, 7.8]
    try:
        for i in range(nframes):
            t = i / fps
            set_robot(model, data, motion, t)
            for k in range(3):
                data.qpos[k] = np.interp(t, sim['time'], sim['base_qpos'][:, k])
            qi = min(np.searchsorted(sim['time'], t), len(sim['time'])-1)
            data.qpos[3:7] = sim['base_qpos'][qi, 3:7]
            data.qpos[7] = np.interp(t, sim['time'], sim['drawer_slide'])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera, scene_option=option)
            frame = renderer.render()
            writer.append_data(frame)
            if stills and any(abs(t-s) < .5/fps for s in shots):
                Image.fromarray(frame).save(out / f'{view}_{t:.2f}.png')
            if i % 120 == 0:
                print(f'{view} frame {i}/{nframes}', flush=True)
    finally:
        writer.close()
        renderer.close()
    print(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--view', choices=['overview', 'detail'], default='overview')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--stills', action='store_true')
    a = parser.parse_args()
    render(a.output, a.view, a.width, a.height, a.fps, a.stills)
