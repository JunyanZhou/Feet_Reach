"""Paired random-reset test for dynamic free-base box push."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.push_box_support.mujoco_freebase_push import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--seed-start', type=int, default=20000)
    ap.add_argument('--duration', type=float, default=5.5)
    ap.add_argument('--root-noise-m', type=float, default=.01)
    ap.add_argument('--joint-noise-rad', type=float, default=.01)
    ap.add_argument('--box-noise-m', type=float, default=.015)
    a = ap.parse_args()
    p = json.loads(a.params.read_text()); p = p.get('params', p).copy()
    p.update(root_noise_m=a.root_noise_m, joint_noise_rad=a.joint_noise_rad,
             box_noise_m=a.box_noise_m)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (a.output / 'episodes.jsonl').open('w') as out:
        for i in range(a.episodes):
            seed = a.seed_start + i
            _, braced, braced_trace, targets = run(p, True, a.duration, seed=seed)
            _, no_box, no_box_trace, no_box_targets = run(p, False, a.duration, replay=targets, seed=seed)
            row = dict(seed=seed, with_box=braced, no_box_same_targets=no_box)
            rows.append(row)
            out.write(json.dumps(row) + '\n'); out.flush()
            if braced['successful_push']:
                np.savez_compressed(a.output / f'success_{seed}_with_box.npz', **braced_trace, target=targets)
                np.savez_compressed(a.output / f'success_{seed}_no_box.npz', **no_box_trace, target=no_box_targets)
            if (i+1) % 8 == 0:
                print('EVAL',i+1,'/',a.episodes,
                      'success',sum(r['with_box']['successful_push'] for r in rows),flush=True)
    report = dict(params=p, episodes=a.episodes, seed_start=a.seed_start,
                  success_with_box=sum(r['with_box']['successful_push'] for r in rows),
                  no_box_fall=sum(r['no_box_same_targets']['first_fall_s'] is not None for r in rows),
                  box_shift_mean_m=float(np.mean([r['with_box']['box_displacement_x_m'] for r in rows])),
                  box_shift_median_m=float(np.median([r['with_box']['box_displacement_x_m'] for r in rows])),
                  box_tilt_mean_rad=float(np.mean([r['with_box']['max_box_tilt_rad'] for r in rows])))
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
