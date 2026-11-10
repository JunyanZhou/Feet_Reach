#!/usr/bin/env python3
"""Remove staged dwell time from the table retrieval replay.

Time anchors shorten the ready, heel placement, pull, recovery, and hand-reach
stages while preserving their order. Source poses are resampled at 60 Hz,
then all FK poses are rebuilt. Optional extra smoothing can be explored, but
zero is the validated default because contact and collision geometry matter.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from demo import IK, ROOT, smooth

OLD = np.array([0, 1, 3, 4.5, 6.5, 7.7, 8.4, 9, 12.5, 13.3,
                14.8, 15.8, 17, 18.5, 19.5, 20.7, 23.7, 24.7, 27.7, 29.0])
NEW = np.array([0, .16, 1.50, 2.45, 4.00, 4.70, 5.15, 5.57, 8.28, 8.80,
                9.85, 10.45, 11.15, 12.00, 12.57, 13.22, 15.07, 15.44,
                17.55, 17.85])


def build(source: Path, output: Path, sigma_s: float = 0):
    output.mkdir(parents=True, exist_ok=True)
    src = dict(np.load(source / 'motion.npz'))
    fps = float(src['fps'])
    t = np.arange(round(NEW[-1] * fps) + 1) / fps
    old_t = np.interp(t, NEW, OLD)
    q = np.stack([np.interp(old_t, src['time'], src['qpos'][:, j])
                  for j in range(src['qpos'].shape[1])], axis=1)
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    # Optional temporal blending. The active foot remains unfiltered during
    # actual object pull for contact fidelity. The validated release uses 0.
    filtered = (gaussian_filter1d(q[:, 7:], sigma=sigma_s * fps,
                                   axis=0, mode='nearest')
                if sigma_s > 0 else q[:, 7:].copy())
    contact_start = float(np.interp(8.45, OLD, NEW))
    contact_end = float(np.interp(12.8, OLD, NEW))
    preserve = smooth((t - (contact_start - .25)) / .25) * \
               (1 - smooth((t - contact_end) / .25))
    filtered[:, :6] = preserve[:, None] * q[:, 7:13] + \
                      (1 - preserve[:, None]) * filtered[:, :6]
    q[:, 7:] = filtered
    ik = IK()
    body_pos = np.empty((len(t), len(src['body_names']), 3))
    body_quat = np.empty((len(t), len(src['body_names']), 4))
    site_pos = np.empty((len(t), len(src['site_names']), 3))
    site_names = list(map(str, src['site_names']))
    for i in range(len(t)):
        ik.data.qpos[:] = q[i]
        mujoco.mj_forward(ik.model, ik.data)
        body_pos[i] = ik.data.xpos[1:]
        body_quat[i] = ik.data.xquat[1:]
        site_pos[i] = [ik.data.site_xpos[ik.sites[n]] for n in site_names]
    old_i = np.clip(np.rint(old_t * fps).astype(int), 0, len(src['time']) - 1)
    motion = dict(time=t, qpos=q, body_pos=body_pos,
                  body_quat_wxyz=body_quat, joint_names=src['joint_names'],
                  body_names=src['body_names'], site_names=src['site_names'],
                  site_pos=site_pos, ik_error=src['ik_error'][old_i],
                  phase=src['phase'][old_i], fps=src['fps'])
    np.savez_compressed(output / 'motion.npz', **motion)
    events = dict(zip(map(str, OLD), map(float, NEW)))
    timeline = {
        'source': str(source / 'motion.npz'),
        'old_anchor_s': OLD.tolist(), 'new_anchor_s': NEW.tolist(),
        'old_to_new_s': events,
        'support_interval_s': [float(np.interp(3, OLD, NEW)),
                               float(np.interp(19.5, OLD, NEW))],
        'before_contact_end_s': float(np.interp(8, OLD, NEW)),
        'hand_access_interval_s': [float(np.interp(23.7, OLD, NEW)),
                                   float(np.interp(24.7, OLD, NEW))],
        'support_release_end_s': float(np.interp(19.6, OLD, NEW)),
        'joint_speed_limit_rad_s': 5.0,
        'smoothing_sigma_s': sigma_s,
        'mode': 'retimed scripted kinematics; dynamic object must be resimulated',
    }
    (output / 'timeline.json').write_text(json.dumps(timeline, indent=2) + '\n')
    v = np.diff(q[:, 7:], axis=0) * fps
    still = np.linalg.norm(v, axis=1) < .03
    print(json.dumps({'duration_old_s': float(src['time'][-1]),
                      'duration_new_s': float(t[-1]),
                      'max_joint_speed_rad_s': float(abs(v).max()),
                      'nearly_still_fraction': float(still.mean()),
                      'timeline': str(output / 'timeline.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'artifacts/table_retrieval_smooth')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/table_retrieval_flow')
    parser.add_argument('--sigma', type=float, default=0)
    args = parser.parse_args()
    build(args.source, args.output, args.sigma)
