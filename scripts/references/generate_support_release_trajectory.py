"""Build a collision-checked single-foot hand-withdrawal teaching trajectory.

This is an offline candidate for a wall-contact transition. It verifies the
right foot alone can carry the computed wrench at every frame. Isaac closed
loop is required before calling the release dynamically successful.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from generate_foot_reach_dataset import Workspace
from probe_multi_target_support import solve_case


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/support_release.npz"))
    p.add_argument("--distance", type=float, default=.06)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not .02 <= args.distance <= .12:
        raise ValueError("Distance must be between 2 and 12 cm")
    w = Workspace(json.loads(args.reference.read_text()))
    q0 = w.q0.copy()
    pin.framesForwardKinematics(w.model, w.data, q0)
    initial_right = w.data.oMf[w.support].copy()
    initial_left = w.data.oMf[w.foot].copy()
    initial_hand = w.data.oMf[w.hand].copy()
    knots = np.linspace(0., 1., 21)
    poses = [q0]
    guess = q0
    for s in knots[1:]:
        result = solve_case(w, "left_hand", np.array([0., -args.distance*s, 0.]),
                            initial_q=guess, check_dynamics=False)
        if not result["solver_success"] or result["hand_target_error_m"] > 3e-4:
            raise RuntimeError(f"IK failed at release fraction {s}: {result}")
        guess = np.array(result["q_pinocchio"])
        poses.append(guess)
    poses = np.stack(poses)
    times = np.arange(251) / 50.
    fraction = np.where(times < 2., smoothstep(times/2.),
                        np.where(times < 3., 1., 1.-smoothstep((times-3.)/2.)))
    q = np.stack([np.interp(fraction, knots, poses[:, i])
                  for i in range(w.model.nq)], axis=1)
    q[:, 3:7] = q0[3:7]
    v = np.zeros((len(times), w.model.nv))
    a = np.zeros_like(v)
    for col in list(range(3)) + list(range(6, w.model.nv)):
        source = col if col < 3 else col+1
        v[:, col] = np.gradient(q[:, source], .02)
        a[:, col] = np.gradient(v[:, col], .02)
    max_hand_error = max_right_error = max_left_error = 0.
    max_torque_ratio = max_friction_ratio = max_cop_ratio = 0.
    peak_joint_speed = 0.
    min_normal_N = np.inf
    collisions = []
    contact_poses = w.contact_poses
    for i, (qi, vi, ai) in enumerate(zip(q, v, a)):
        pin.framesForwardKinematics(w.model, w.data, qi)
        right = w.data.oMf[w.support].copy()
        left = w.data.oMf[w.foot].copy()
        hand = w.data.oMf[w.hand].copy()
        max_right_error = max(max_right_error,
                              np.linalg.norm(pin.log6(initial_right.inverse()*right).vector))
        max_left_error = max(max_left_error,
                             np.linalg.norm(pin.log6(initial_left.inverse()*left).vector))
        max_hand_error = max(max_hand_error,
                             abs(hand.translation[1] -
                                 (initial_hand.translation[1]-args.distance*fraction[i])))
        # Reuse the conservative self/environment collision screen, with the
        # hand deliberately free and both foot pose errors checked above.
        w.contact_poses = [right, hand]
        reason = w.geometry(qi)
        if reason:
            collisions.append((i, reason))
        h = pin.rnea(w.model, w.data, qi, vi, ai)
        pin.computeJointJacobians(w.model, w.data, qi)
        J = pin.getFrameJacobian(w.model, w.data, w.support,
                                 pin.LOCAL_WORLD_ALIGNED)
        wrench = np.linalg.solve(J[:, :6].T, h[:6])
        tau = h[6:] - J[:, 6:].T @ wrench
        effort = .85*w.model.effortLimit[6:]
        max_torque_ratio = max(max_torque_ratio,
                               np.max(np.abs(tau)/np.maximum(effort, 1e-6)))
        normal = wrench[2]
        min_normal_N = min(min_normal_N, normal)
        max_friction_ratio = max(max_friction_ratio,
                                 np.linalg.norm(wrench[:2])/normal)
        # Same conservative foot moment bounds as Workspace.dynamics.
        max_cop_ratio = max(max_cop_ratio,
                            abs(wrench[3]/normal)/.028,
                            abs(wrench[5]/normal)/.018,
                            (-wrench[4]/normal)/.11,
                            (wrench[4]/normal)/.045)
        peak_joint_speed = max(peak_joint_speed, np.max(np.abs(vi[6:])))
    w.contact_poses = contact_poses
    report = {
        "distance_m": args.distance, "frames": len(times), "frequency_Hz": 50,
        "max_right_foot_pose_error_m": max_right_error,
        "max_left_foot_pose_error_m": max_left_error,
        "max_hand_y_error_m": max_hand_error,
        "max_torque_ratio_85pct_limit": max_torque_ratio,
        "max_friction_ratio": max_friction_ratio,
        "max_foot_moment_ratio": max_cop_ratio,
        "min_foot_normal_N": min_normal_N,
        "peak_joint_speed_rad_s": peak_joint_speed,
        "collision_count": len(collisions), "first_collision": collisions[0] if collisions else None,
        "note": "Offline inverse dynamics under right-foot-only support; no Isaac tracking claim.",
    }
    if (max(max_right_error, max_left_error, max_hand_error) > 1e-3
            or max_torque_ratio > 1. or max_friction_ratio > .7
            or max_cop_ratio > 1. or min_normal_N <= 0 or collisions):
        raise RuntimeError(f"Release trajectory failed offline checks: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, time_s=times, q_pinocchio=q,
                        joint_names=np.array(w.names), dof_pos=q[:, 7:],
                        hand_release_fraction=fraction, report_json=json.dumps(report))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
