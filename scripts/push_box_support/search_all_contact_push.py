"""Reproduce the all-contact random search used to select a box-push reference.

Candidate 112 is the nominal reference in freebase_all_contact_candidate112.json.
Held-out reset evaluation is separate: evaluate_freebase_push.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.push_box_support.mujoco_freebase_push import run


def sample(rng: np.random.Generator) -> dict:
    return {
        'contact_mode': 'all',
        'box_mass': float(rng.uniform(9, 21)),
        'box_half_x': float(rng.uniform(.19, .28)),
        'box_friction': float(rng.uniform(.16, .50)),
        'box_com_z_offset': float(rng.uniform(-.15, -.01)),
        'hand_z': float(rng.uniform(.40, .64)),
        'hand_y': float(rng.uniform(.14, .23)),
        'palm_radius': float(rng.uniform(.033, .058)),
        'push_depth': float(rng.uniform(.04, .33)),
        'waist_pitch': float(rng.uniform(.10, .42)),
        'approach_gap': float(rng.uniform(-.01, .035)),
        'left_step_dx': float(rng.uniform(0, .22)),
        'left_step_start': float(rng.uniform(1.6, 2.4)),
        'left_step_duration': float(rng.uniform(.85, 1.35)),
        'step_height': float(rng.uniform(.07, .13)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=180)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--duration', type=float, default=5.5)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(args.trials):
        params = sample(rng)
        _, metrics, _, _ = run(params, True, args.duration)
        rows.append(dict(index=index, params=params, metrics=metrics))
        if metrics['successful_push'] or (index + 1) % 30 == 0:
            print(index, metrics['successful_push'],
                  round(metrics['box_displacement_x_m'], 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
