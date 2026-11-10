#!/usr/bin/env python3
"""Render the box-pushing USD using Isaac Lab or Isaac Sim.

Run on the rendering machine, e.g.:
  ./isaaclab.sh -p /path/to/package/scripts/push_box_support/render_isaac.py --headless --enable_cameras

This script is supplied for migration; it has not been executed on the source
headless host. The self-contained USD is also directly openable in Isaac Sim.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--usd',type=Path,default=ROOT/'artifacts/push_box_support/push_box_support.usdc')
p.add_argument('--output',type=Path,default=ROOT/'artifacts/push_box_support/isaac_frames')
p.add_argument('--camera',default='/World/Cameras/Overview')
p.add_argument('--width',type=int,default=1920);p.add_argument('--height',type=int,default=1080)
p.add_argument('--fps',type=int,default=60);p.add_argument('--subframes',type=int,default=8)
p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=None)
try:
 from isaaclab.app import AppLauncher
except ImportError:
 AppLauncher=None
if AppLauncher is not None:
 AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=True
 launcher=AppLauncher(args);app=launcher.app
else:
 p.add_argument('--headless',action='store_true');p.add_argument('--enable_cameras',action='store_true')
 args=p.parse_args()
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':args.headless,'width':args.width,'height':args.height})
# Isaac imports must follow application creation.
import omni.usd,omni.timeline,omni.replicator.core as rep
try:
 if not args.usd.is_file():raise FileNotFoundError(args.usd)
 omni.usd.get_context().open_stage(str(args.usd.resolve()))
 for _ in range(60):app.update()
 stage=omni.usd.get_context().get_stage();timeline=omni.timeline.get_timeline_interface();timeline.pause()
 if not stage.GetPrimAtPath(args.camera).IsValid():raise ValueError('Unknown camera '+args.camera)
 rep.orchestrator.set_capture_on_play(False)
 product=rep.create.render_product(args.camera,(args.width,args.height))
 writer=rep.WriterRegistry.get('BasicWriter');args.output.mkdir(parents=True,exist_ok=True)
 writer.initialize(output_dir=str(args.output.resolve()),rgb=True,frame_padding=6);writer.attach([product])
 end=stage.GetEndTimeCode()/stage.GetTimeCodesPerSecond() if args.end is None else args.end
 n=round((end-args.start)*args.fps)+1;times=[]
 for i in range(n):
  t=args.start+i/args.fps;timeline.set_current_time(t)
  app.update()
  rep.orchestrator.step(delta_time=0.0,pause_timeline=True,rt_subframes=args.subframes)
  times.append(t)
  if i%30==0:print(f'Isaac render {i+1}/{n}, t={t:.3f}s',flush=True)
 rep.orchestrator.wait_until_complete();writer.detach()
 (args.output/'render_manifest.json').write_text(json.dumps({'usd':str(args.usd),'camera':args.camera,'fps':args.fps,'width':args.width,'height':args.height,'frames':n,'time_s':times},indent=2))
 print('RGB PNG frames saved to',args.output)
finally:app.close()
