#!/usr/bin/env python3
"""Render saved states without solving IK or re-running contact dynamics."""
import argparse
from pathlib import Path
from demo import OUT,render
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--scene',type=Path,default=None);p.add_argument('--view',choices=['overview','detail','side'],default='overview');p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080);p.add_argument('--fps',type=int,default=30);p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=None);p.add_argument('--stills',action='store_true');p.add_argument('--clean-only',action='store_true');a=p.parse_args()
motion=dict(np.load(a.output/'motion.npz'));sim=dict(np.load(a.output/'simulation.npz'));model=mujoco.MjModel.from_xml_path(str(a.scene or a.output/'scene.xml'))
render(model,motion,sim,a.output,a.width,a.height,a.fps,a.view,a.stills,a.start,a.end,a.clean_only)
