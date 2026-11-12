"""Local search selected by multiple full-task randomized MuJoCo rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.foot_pin_bimanual.mujoco_task import run


def candidate(base, rng):
    p = dict(base)
    groups = rng.choice(5, size=rng.integers(1, 4), replace=False)
    for group in groups:
        if group == 0:  # foot contact
            p['foot_gain'] = np.clip(p['foot_gain'] + rng.normal(0, .15), .35, 2.)
            p['foot_start'] = np.clip(p['foot_start'] + rng.normal(0, .1), .4, 1.3)
            p['foot_ramp'] = np.clip(p['foot_ramp'] + rng.normal(0, .1), .35, 1.2)
            p['foot_target_y'] = np.clip(p['foot_target_y'] + rng.normal(0, .01), -.04, .06)
        elif group == 1:  # arm timing
            for key, sigma, lo, hi in (
                ('approach_start', .09, 1., 1.9),
                ('approach_duration', .1, .4, 1.3),
                ('grasp_start', .1, 2., 3.),
                ('grasp_duration', .1, .2, 1.),
                ('pull_start', .1, 3., 4.),
                ('pull_duration', .12, .6, 1.6)):
                p[key] = float(np.clip(p[key] + rng.normal(0, sigma), lo, hi))
        elif group == 2:  # handle layout
            for side in ('left', 'right'):
                for key, sigma, lo, hi in (
                    ('pad_x', .008, -.32, -.07),
                    ('pad_width', .01, .03, .11),
                    ('pad_height', .01, .035, .12)):
                    name = f'{side}_{key}'
                    p[name] = float(np.clip(p[name] + rng.normal(0, sigma), lo, hi))
        elif group == 3:  # physical friction/compliance
            p['lid_friction'] = float(np.clip(p['lid_friction'] *
                                               np.exp(rng.normal(0, .05)), 1., 10.))
            p['brace_solref'] = float(np.clip(p.get('brace_solref', .02) +
                                                rng.normal(0, .002), .02, .04))
        else:  # tracking gains
            p['cartesian_gain'] = float(np.clip(p['cartesian_gain'] +
                                                 rng.normal(0, .12), .5, 2.))
            p['pull_depth'] = float(np.clip(p['pull_depth'] +
                                            rng.normal(0, .025), .16, .4))
            p['side_offset'] = float(np.clip(p['side_offset'] +
                                             rng.normal(0, .015), .04, .18))
    return p


def score(m):
    return (12 * m['success'] + 4 * max(0., m['lid_open_m']) / .19 +
            2 * m['contact_s']['three_way'] +
            5 * m['three_way_motion_m'] -
            15 * max(0., m['base_max_drift_m'] - .05) -
            4 * m['fell'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--trials', type=int, default=60)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--eval-seed-start', type=int, default=28000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.params.read_text())['params']
    rng = np.random.default_rng(a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    best = -1e9
    with a.output.open('w') as out:
        for i in range(a.trials):
            q = dict(base) if i == 0 else candidate(base, rng)
            metrics = [run(True, q, 4.6)]
            for j in range(4):
                seed = a.eval_seed_start + j
                frng = np.random.default_rng(seed + 90315)
                physical = dict(q)
                physical['base_friction'] *= frng.uniform(.9, 1.1)
                physical['lid_friction'] *= frng.uniform(.9, 1.1)
                metrics.append(run(True, physical, 4.6, seed=seed, randomize=True))
            value = sum(score(m) for m in metrics) / len(metrics)
            row = dict(index=i, params=q, score=value,
                       successes=sum(m['success'] for m in metrics), metrics=metrics)
            out.write(json.dumps(row) + '\n');out.flush()
            if value > best:
                best = value
                print('BEST', i, round(value, 3), row['successes'],
                      [round(m['lid_open_m'], 2) for m in metrics], flush=True)
            if i % 20 == 19:
                print('PROGRESS', i + 1, '/', a.trials, flush=True)


if __name__ == '__main__':
    main()
