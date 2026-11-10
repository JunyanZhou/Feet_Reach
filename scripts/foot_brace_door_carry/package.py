#!/usr/bin/env python3
"""Create transfer archive and MP4 archive with explicit files and checksums."""
import hashlib
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from make_demo import ROOT, OUT

SCRIPT_NAMES = ['make_demo.py', 'render.py', 'evaluate.py',
                'collision_audit.py', 'export_usd.py', 'validate_usd.py', 'render_isaac.py',
                'encode_video.py', 'package.py']
ARTIFACT_NAMES = ['README.md', 'PROGRESS.md', 'scene.xml', 'motion.npz',
                  'simulation.npz', 'simulation_no_foot.npz',
                  'evaluation.json', 'evaluation_no_foot.json',
                  'validation.json', 'scene_manifest.json',
                  'collision_audit.json',
                  'trajectory_portable.npz', 'foot_brace_door_carry.usdc',
                  'usd_validation.json', 'overview_clean.mp4', 'side_clean.mp4',
                  'cinematic_clean.mp4']


def package():
    bundle = OUT / 'portable_bundle/foot_brace_door_carry_demo'
    bundle.mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_NAMES:
        dst = bundle / 'scripts/foot_brace_door_carry' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts/foot_brace_door_carry' / name, dst)
    for name in ['demo.py', 'requirements.txt', 'requirements-lock.txt']:
        dst = bundle / 'scripts/table_retrieval' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts/table_retrieval' / name, dst)
    for name in ARTIFACT_NAMES:
        dst = bundle / 'artifacts/foot_brace_door_carry' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / name, dst)
    shutil.copytree(ROOT / 'deploy/mujoco_whole_body/g1',
                    bundle / 'deploy/mujoco_whole_body/g1', dirs_exist_ok=True)
    shutil.copy2(ROOT / 'deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt',
                 bundle / 'deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt')
    shutil.copy2(OUT / 'README.md', bundle / 'README.md')
    scene = bundle / 'artifacts/foot_brace_door_carry/scene.xml'
    xml = ET.parse(scene)
    xml.getroot().find('compiler').set(
        'meshdir', '../../deploy/mujoco_whole_body/g1/meshes')
    xml.write(scene, encoding='unicode')
    checksums = {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(bundle.rglob('*')) if p.is_file()}
    (bundle / 'SHA256SUMS.json').write_text(json.dumps(checksums, indent=2) + '\n')
    archive = OUT / 'foot_brace_door_carry_demo.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as writer:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                writer.write(p, str(p.relative_to(bundle.parent)))
    movies = OUT / 'foot_brace_door_carry_mp4.zip'
    with zipfile.ZipFile(movies, 'w', compression=zipfile.ZIP_STORED) as writer:
        writer.write(OUT / 'cinematic_clean.mp4', 'foot_brace_door_carry_cinematic.mp4')
        writer.write(OUT / 'overview_clean.mp4', 'foot_brace_door_carry_overview.mp4')
        writer.write(OUT / 'side_clean.mp4', 'foot_brace_door_carry_side.mp4')
    result = {'full_bundle': str(archive), 'mp4_zip': str(movies),
              'file_count': len(checksums) + 1,
              'bundle_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (OUT / 'package_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    package()
