"""Paired support ablation with matched robot/object perturbations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.foot_pin_bimanual.mujoco_task import run


def mechanical(m):
    return (m['lid_open_m'] >= m.get('open_threshold_m', .15) and
            m['base_max_drift_m'] < .05 and
            m['root_min_height_m'] > .55)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=64)
    p.add_argument('--seed-start', type=int, default=1000)
    p.add_argument('--duration', type=float, default=4.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    teacher = json.loads(a.params.read_text())
    params = teacher.get('params', teacher)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (a.output / 'episodes.jsonl').open('w') as f:
        for i in range(a.episodes):
            seed = a.seed_start + i
            rng = np.random.default_rng(seed + 90315)
            q = dict(params)
            q['base_friction'] *= rng.uniform(.9, 1.1)
            q['lid_friction'] *= rng.uniform(.9, 1.1)
            with_support = run(True, q, a.duration, seed=seed, randomize=True)
            without = run(False, q, a.duration, seed=seed, randomize=True)
            row = dict(seed=seed, with_support=with_support, without_support=without,
                       base_friction=q['base_friction'], lid_friction=q['lid_friction'])
            rows.append(row)
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 8 == 0:
                print(f'PAIRS {i+1}/{a.episodes} success='
                      f'{sum(r["with_support"]["success"] for r in rows)}', flush=True)
    summary = dict(episodes=a.episodes, seed_start=a.seed_start,
                   duration_s=a.duration, open_threshold_m=params.get('open_threshold_m', .15),
                   success_with=sum(r['with_support']['success'] for r in rows),
                   success_without=sum(r['without_support']['success'] for r in rows),
                   mechanical_with=sum(mechanical(r['with_support']) for r in rows),
                   mechanical_without=sum(mechanical(r['without_support']) for r in rows),
                   supported_only_mechanical=sum(mechanical(r['with_support']) and
                                                 not mechanical(r['without_support']) for r in rows),
                   coordinated_with=sum(mechanical(r['with_support']) and
                                        r['with_support']['contact_s']['three_way'] >= .15 and
                                        r['with_support']['three_way_active_contact_s'] >= .08 and
                                        r['with_support']['three_way_motion_m'] >= .003 for r in rows),
                   lid_with_mean_m=float(np.mean([r['with_support']['lid_open_m'] for r in rows])),
                   lid_without_mean_m=float(np.mean([r['without_support']['lid_open_m'] for r in rows])),
                   perturbations=dict(root_xy_uniform_m=.01, joint_normal_std_rad=.01,
                                      base_xy_uniform_m=.015, base_friction_multiplier=[.9, 1.1],
                                      lid_friction_multiplier=[.9, 1.1]))
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
