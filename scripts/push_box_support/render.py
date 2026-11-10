#!/usr/bin/env python3
"""Render clean MP4 views of the G1 box-push replay."""
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


def render(out=OUT,view='follow',width=1920,height=1080,fps=60,stills=False):
    motion=dict(np.load(out/'motion.npz'))
    sim=dict(np.load(out/'simulation.npz'))
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height,width)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[.65,0,.52]
    if view=='side':
        camera.distance=3.05;camera.azimuth=90;camera.elevation=-12
    else:
        camera.distance=2.95;camera.azimuth=45;camera.elevation=-18
    path=out/(view+'_clean.mp4')
    writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=9,
            macro_block_size=1,ffmpeg_params=['-threads','4'])
    nframes=round(float(motion['time'][-1])*fps)+1
    shots=[0,1.5,2.5,3.8,5.0,6.4,7.8,9.0]
    try:
        for i in range(nframes):
            t=i/fps
            if view=='follow':
                u=float(smooth((t-1.0)/8.0))
                camera.lookat[:]=[.30+.95*u,0,.52]
            set_robot(model,data,motion,t)
            idx=int(round(t/model.opt.timestep))
            data.qpos[:7]=sim['box_qpos'][min(idx,len(sim['time'])-1)]
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera,scene_option=option)
            frame=renderer.render()
            writer.append_data(frame)
            if stills and any(abs(t-s)<.5/fps for s in shots):
                Image.fromarray(frame).save(out/f'{view}_{t:.2f}.png')
            if i%120==0:print(f'{view} frame {i}/{nframes}',flush=True)
    finally:
        writer.close();renderer.close()
    print(path)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--view',choices=['follow','side'],default='follow')
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1080)
    p.add_argument('--fps',type=int,default=60)
    p.add_argument('--stills',action='store_true')
    a=p.parse_args();render(a.output,a.view,a.width,a.height,a.fps,a.stills)
