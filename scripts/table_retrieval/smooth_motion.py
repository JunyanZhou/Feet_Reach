#!/usr/bin/env python3
"""Smooth the saved kinematic demo while preserving the active foot contact path.

This is an offline edit to a scripted trajectory. It does not make the robot a
dynamic controller. The original motion is left intact for comparison.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from demo import IK, OUT, smooth, target_at


def make_smooth_motion(source: Path, output: Path, sigma_s: float = .1):
    output.mkdir(parents=True, exist_ok=True)
    original = dict(np.load(source / 'motion.npz'))
    t = original['time']
    fps = float(original['fps'])
    q0 = original['qpos']
    q = q0.copy()
    filtered = gaussian_filter1d(q0[:, 7:], sigma=sigma_s * fps,
                                 axis=0, mode='nearest')
    # Left leg joint order is the first six joints of G1's named 29-DoF model.
    ik = IK()
    assert all(name.startswith('left_') for name in ik.joints[:6])
    # Keep the sole pose identical throughout the contact sequence, including
    # approach and release. Blend back to filtered motion with quintic fades.
    preserve = smooth((t - 5.8) / .5) * (1 - smooth((t - 13.2) / .5))
    filtered[:, :6] = (preserve[:, None] * q0[:, 7:13]
                       + (1 - preserve[:, None]) * filtered[:, :6])
    q[:, 7:] = filtered

    body_pos = np.empty_like(original['body_pos'])
    body_quat = np.empty_like(original['body_quat_wxyz'])
    site_pos = np.empty_like(original['site_pos'])
    ik_error = np.empty_like(original['ik_error'])
    site_names = list(map(str, original['site_names']))
    for i, ti in enumerate(t):
        ik.data.qpos[:] = q[i]
        mujoco.mj_forward(ik.model, ik.data)
        body_pos[i] = ik.data.xpos[1:]
        body_quat[i] = ik.data.xquat[1:]
        site_pos[i] = [ik.data.site_xpos[ik.sites[n]] for n in site_names]
        targets = target_at(float(ti))[2]
        ik_error[i] = [np.linalg.norm(site_pos[i, j] - targets[n])
                       for j, n in enumerate(site_names)]

    updated = dict(original, qpos=q, body_pos=body_pos,
                   body_quat_wxyz=body_quat, site_pos=site_pos,
                   ik_error=ik_error)
    np.savez_compressed(output / 'motion.npz', **updated)
    rate = lambda x: np.diff(x, axis=0) * fps
    v0, v1 = rate(q0[:, 7:]), rate(q[:, 7:])
    a0, a1 = rate(v0), rate(v1)
    contact = (t >= 8.4) & (t <= 13.2)
    support = (t >= 3) & (t <= 19.5)
    metrics = {
        'source': str(source / 'motion.npz'),
        'smoothing_sigma_s': sigma_s,
        'active_foot_preserved_s': [6.3, 13.2],
        'max_joint_speed_before_rad_s': float(abs(v0).max()),
        'max_joint_speed_after_rad_s': float(abs(v1).max()),
        'max_joint_acceleration_before_rad_s2': float(abs(a0).max()),
        'max_joint_acceleration_after_rad_s2': float(abs(a1).max()),
        'max_support_palm_shift_m': float(np.linalg.norm(
            site_pos[support, 2] - original['site_pos'][support, 2], axis=1).max()),
        'max_active_foot_contact_shift_m': float(np.linalg.norm(
            site_pos[contact, 0] - original['site_pos'][contact, 0], axis=1).max()),
        'mode': 'offline-smoothed scripted kinematics',
    }
    (output / 'smooth_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=OUT)
    parser.add_argument('--output', type=Path,
                        default=OUT.parent / 'table_retrieval_smooth')
    parser.add_argument('--sigma', type=float, default=.1)
    args = parser.parse_args()
    make_smooth_motion(args.source, args.output, args.sigma)
