"""Search scripted grasp and pull references in the physical MuJoCo drawer task."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.foot_pin_bimanual.mujoco_task import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--foot-gain-min', type=float, default=.9)
    parser.add_argument('--foot-gain-max', type=float, default=2.1)
    parser.add_argument('--base-friction-min', type=float)
    parser.add_argument('--base-friction-max', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    base = json.loads(args.params.read_text())['params']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = -1e9
    with args.output.open('w') as out:
        for i in range(args.trials):
            p = dict(base)
            if args.base_friction_min is not None:
                if args.base_friction_max is None:
                    raise ValueError('--base-friction-max is required with --base-friction-min')
                p['base_friction'] = rng.uniform(args.base_friction_min,
                                                  args.base_friction_max)
            pad_x = rng.uniform(-.31, -.11)
            p['left_pad_x'] = pad_x + rng.uniform(-.025, .025)
            p['right_pad_x'] = pad_x + rng.uniform(-.025, .025)
            for side in ('left', 'right'):
                p[f'{side}_pad_width'] = rng.uniform(.038, .10)
                p[f'{side}_pad_height'] = rng.uniform(.045, .11)
                p[f'{side}_pad_depth'] = rng.uniform(.01, .028)
            p['lid_friction'] = rng.uniform(2., 10.)
            p['approach_start'] = rng.uniform(1.0, 1.8)
            p['approach_duration'] = rng.uniform(.45, 1.15)
            p['grasp_start'] = rng.uniform(2.0, 2.8)
            p['grasp_duration'] = rng.uniform(.25, .8)
            p['pull_start'] = p['grasp_start'] + p['grasp_duration'] + rng.uniform(0., .4)
            p['pull_duration'] = rng.uniform(.7, 1.4)
            p['pull_depth'] = rng.uniform(.15, .34)
            p['cartesian_gain'] = rng.uniform(.55, 1.8)
            p['side_offset'] = rng.uniform(.045, .16)
            p['grasp_gap'] = rng.uniform(-.012, .025)
            p['foot_gain'] = rng.uniform(args.foot_gain_min, args.foot_gain_max)
            p['foot_target_y'] = rng.uniform(-.015, .055)
            m = run(True, p, 4.8)
            score = (4 * m['lid_open_m'] - 2 * m['base_max_drift_m'] +
                     .9 * m['contact_s']['both_hands'] +
                     1.5 * m['contact_s']['three_way'] +
                     4 * m['three_way_motion_m'] -
                     2 * m['fell'])
            row = dict(index=i, params=p, metrics=m, score=score)
            out.write(json.dumps(row) + '\n')
            out.flush()
            if score > best:
                best = score
                print('BEST', i, round(best, 3), round(m['lid_open_m'], 3),
                      round(m['base_max_drift_m'], 3),
                      round(m['contact_s']['both_hands'], 3),
                      round(m['contact_s']['three_way'], 3), flush=True)
            if i % 50 == 49:
                print('PROGRESS', i + 1, '/', args.trials, flush=True)


if __name__ == '__main__':
    main()
