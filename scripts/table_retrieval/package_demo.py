#!/usr/bin/env python3
"""Build an explicit, dependency-complete transfer package (no credentials/envs)."""
import hashlib,json,shutil,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
from demo import ROOT,OUT
SCRIPTS=['demo.py','audit.py','evaluate.py','export_usd.py','validate_usd.py','replay_mujoco.py','render_isaac.py','encode_video.py','package_demo.py','requirements.txt','requirements-lock.txt','validated_recovery.npz','recovery_reference_info.json','README.md']
ARTIFACTS=['overview.mp4','overview_clean.mp4','detail.mp4','detail_clean.mp4','storyboard.jpg','contact_report.png','scene.xml','motion.npz','simulation.npz','trajectory_portable.npz','table_retrieval.usdc','scene_manifest.json','evaluation.json','validation.json','usd_validation.json','geometry_audit.json','robustness_summary.json','PROGRESS.md','README.md','video_validation.json','transfer_validation.json']
def package():
 bundle=OUT/'portable_bundle'/'table_retrieval_demo';bundle.mkdir(parents=True,exist_ok=True)
 for name in SCRIPTS:
  src=ROOT/'scripts/table_retrieval'/name;dst=bundle/'scripts/table_retrieval'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 for name in ARTIFACTS:
  src=OUT/name;dst=bundle/'artifacts/table_retrieval'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 control_src=OUT/'controls/evaluation_no_foot_collision.json';control_dst=bundle/'artifacts/table_retrieval/controls/evaluation_no_foot_collision.json';control_dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(control_src,control_dst)
 shutil.copytree(ROOT/'deploy/mujoco_whole_body/g1',bundle/'deploy/mujoco_whole_body/g1',dirs_exist_ok=True)
 shutil.copy2(ROOT/'deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt',bundle/'deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt')
 shutil.copy2(OUT/'README.md',bundle/'README.md')
 scene=bundle/'artifacts/table_retrieval/scene.xml';xml=ET.parse(scene);xml.getroot().find('compiler').set('meshdir','../../deploy/mujoco_whole_body/g1/meshes');xml.write(scene,encoding='unicode')
 checksum={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
 (bundle/'SHA256SUMS.json').write_text(json.dumps(checksum,indent=2)+'\n')
 archive=OUT/'table_retrieval_demo.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(bundle.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(bundle.parent)))
 print(json.dumps({'bundle':str(bundle),'archive':str(archive),'archive_size_MB':archive.stat().st_size/1e6,'files':len(checksum)+1,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
 return bundle,archive
if __name__=='__main__':package()
