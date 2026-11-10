#!/usr/bin/env python3
"""Package the validated smooth replay and a separate ZIP for the MP4."""
import hashlib
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from demo import ROOT

SOURCE = ROOT / 'artifacts/table_retrieval_smooth'
SCRIPTS = [
    'demo.py', 'audit.py', 'evaluate.py', 'export_usd.py',
    'validate_usd.py', 'replay_mujoco.py', 'render_isaac.py',
    'encode_video.py', 'smooth_motion.py', 'retime_motion.py',
    'package_smooth.py', 'requirements.txt',
    'requirements-lock.txt', 'validated_recovery.npz',
    'recovery_reference_info.json',
]
ARTIFACTS = [
    'README.md', 'overview_clean.mp4', 'scene.xml', 'motion.npz',
    'simulation.npz', 'trajectory_portable.npz', 'table_retrieval.usdc',
    'scene_manifest.json', 'evaluation.json', 'validation.json',
    'usd_validation.json', 'geometry_audit.json',
    'robustness_summary.json', 'smooth_metrics.json',
    'timeline.json',
]


def package(source=SOURCE, name='table_retrieval_smooth'):
    bundle = source / 'portable_bundle' / (name + '_demo')
    bundle.mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        dst = bundle / 'scripts/table_retrieval' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts/table_retrieval' / name, dst)
    for name in ARTIFACTS:
        if not (source / name).is_file():
            continue
        dst = bundle / 'artifacts/table_retrieval' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, dst)
    original = bundle / 'artifacts/table_retrieval_original/motion.npz'
    original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'artifacts/table_retrieval/motion.npz', original)
    if (ROOT / 'artifacts/table_retrieval_smooth/motion.npz').is_file():
        smoothed = bundle / 'artifacts/table_retrieval_smooth_source/motion.npz'
        smoothed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'artifacts/table_retrieval_smooth/motion.npz', smoothed)
    shutil.copytree(ROOT / 'deploy/mujoco_whole_body/g1',
                    bundle / 'deploy/mujoco_whole_body/g1', dirs_exist_ok=True)
    shutil.copy2(ROOT / 'deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt',
                 bundle / 'deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt')
    shutil.copy2(source / 'README.md', bundle / 'README.md')
    scene = bundle / 'artifacts/table_retrieval/scene.xml'
    xml = ET.parse(scene)
    xml.getroot().find('compiler').set(
        'meshdir', '../../deploy/mujoco_whole_body/g1/meshes')
    xml.write(scene, encoding='unicode')
    checksums = {
        str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bundle.rglob('*')) if p.is_file()
    }
    (bundle / 'SHA256SUMS.json').write_text(json.dumps(checksums, indent=2) + '\n')
    archive = source / (bundle.name.removesuffix('_demo') + '_bundle.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as out:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                out.write(p, str(p.relative_to(bundle.parent)))
    movie_archive = source / (bundle.name.removesuffix('_demo') + '_mp4.zip')
    with zipfile.ZipFile(movie_archive, 'w', compression=zipfile.ZIP_STORED) as out:
        out.write(source / 'overview_clean.mp4',
                  bundle.name.removesuffix('_demo') + '.mp4')
    print(json.dumps({
        'full_bundle': str(archive), 'movie_zip': str(movie_archive),
        'files': len(checksums) + 1,
        'bundle_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--name', default='table_retrieval_smooth')
    args = parser.parse_args()
    package(args.source, args.name)
