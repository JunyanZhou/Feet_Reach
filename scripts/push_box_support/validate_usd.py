#!/usr/bin/env python3
"""Verify baked USD transforms and self containment."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from pxr import Usd, UsdGeom, UsdUtils
from make_demo import OUT


def validate(out):
    stage = Usd.Stage.Open(str(out / 'push_box_support.usdc'))
    motion = dict(np.load(out / 'motion.npz'))
    sim = dict(np.load(out / 'simulation.npz'))
    model = mujoco.MjModel.from_xml_path(str(out / 'scene.xml'))
    data = mujoco.MjData(model)
    frames = [0, 100, 200, 300, len(motion['time']) - 1]
    worst = 0.0
    count = 0
    for i in frames:
        data.qpos[:7] = sim['box_qpos'][int(round(
            motion['time'][i]/model.opt.timestep))]
        mujoco.mj_forward(model, data)
        for j, name in enumerate(motion['body_names']):
            path = '/World/G1/' + str(name)
            usd = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            position = np.array(usd.ComputeLocalToWorldTransform(i).ExtractTranslation())
            worst = max(worst, float(np.linalg.norm(position - motion['body_pos'][i, j])))
            count += 1
        for name in ['Box']:
            path = '/World/' + name
            usd = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            position = np.array(usd.ComputeLocalToWorldTransform(i).ExtractTranslation())
            expected = data.xpos[model.body('box').id]
            worst = max(worst, float(np.linalg.norm(position - expected)))
            count += 1
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(out / 'push_box_support.usdc'))
    result = {'frame_count': int(stage.GetEndTimeCode() + 1),
              'world_transform_checks': count,
              'max_translation_error_m': worst,
              'external_assets': list(map(str, assets)),
              'unresolved_dependencies': list(map(str, unresolved)),
              'layer_count': len(layers),
              'isaac_runtime_tested': False}
    result['passed'] = (result['frame_count'] == len(motion['time'])
                        and worst < 1e-6 and not assets and not unresolved)
    (out / 'usd_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('USD validation failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    validate(parser.parse_args().output)
