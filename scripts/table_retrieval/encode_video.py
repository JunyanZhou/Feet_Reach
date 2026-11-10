#!/usr/bin/env python3
import argparse
from pathlib import Path
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('frames',type=Path);p.add_argument('--fps',type=int,default=30);p.add_argument('--output',type=Path,default=Path('isaac_render.mp4'));a=p.parse_args()
files=sorted(a.frames.glob('rgb*.png'))
if not files:files=sorted(a.frames.rglob('rgb*.png'))
if not files:raise SystemExit('No RGB PNG frames found')
with imageio.get_writer(a.output,fps=a.fps,codec='libx264',quality=9,macro_block_size=1) as w:
 for path in files:w.append_data(imageio.imread(path))
print(a.output,len(files),'frames')
