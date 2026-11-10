#!/usr/bin/env python3
"""Render clean 1080p movies from saved MuJoCo door/carry states."""
import argparse
import os
import sys
from pathlib import Path
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from make_demo import OUT, set_robot, smooth


def render(out=OUT, view='overview', width=1920, height=1080, fps=60, stills=False):
    motion=dict(np.load(out/'motion.npz'))
    sim=dict(np.load(out/'simulation.npz'))
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height,width)
    option=mujoco.MjvOption()
    option.geomgroup[3]=0
    camera=mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[.35,-.02,.65]
    if view=='side':
        camera.distance=2.75
        camera.azimuth=90
        camera.elevation=-13
    else:
        camera.distance=2.75
        camera.azimuth=45
        camera.elevation=-21
    path=out/(view+'_clean.mp4')
    writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=9,
            macro_block_size=1,ffmpeg_params=['-threads','4'])
    nframes=round(float(motion['time'][-1])*fps)+1
    shots=[0,1.0,1.8,3.0,4.3,5.4,6.5,7.8]
    try:
        for i in range(nframes):
            t=i/fps
            if view=='cinematic':
                u=float(smooth((t-4.0)/2.5))
                v=float(smooth((t-6.5)/3.5))
                camera.azimuth=45+90*u
                camera.lookat[:]=[.35+.23*u+.35*v,-.02-.03*u,.65]
            set_robot(model,data,motion,t)
            data.qpos[model.joint('door_hinge').qposadr[0]]=(
                np.interp(t,sim['time'],sim['door_angle']))
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera,scene_option=option)
            frame=renderer.render()
            writer.append_data(frame)
            if stills and any(abs(t-s)<.5/fps for s in shots):
                Image.fromarray(frame).save(out/f'{view}_{t:.2f}.png')
            if i%120==0:print(f'{view} frame {i}/{nframes}',flush=True)
    finally:
        writer.close()
        renderer.close()
    print(path)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--view',choices=['overview','side','cinematic'],default='overview')
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--fps',type=int,default=60)
    p.add_argument('--stills',action='store_true')
    a=p.parse_args()
    render(a.output,a.view,a.width,a.height,a.fps,a.stills)
