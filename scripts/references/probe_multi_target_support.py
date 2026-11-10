"""Probe small supported G1 root and hand targets with full-body Pinocchio IK.

This is an offline feasibility screen, not a physics rollout or teaching dataset.
It holds the right sole and left swing sole at the reference and keeps the left
hand on the wall plane while moving one commanded body target at a time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from generate_foot_reach_dataset import Workspace


def solve_case(workspace, goal, delta, initial_q=None, check_dynamics=True):
    model, data = workspace.model, workspace.data
    q0 = workspace.q0
    pin.framesForwardKinematics(model, data, q0)
    support0 = data.oMf[workspace.support].copy()
    hand0 = data.oMf[workspace.hand].copy()
    foot0 = data.oMf[workspace.foot].copy()
    root_target = q0[:3].copy()
    hand_target = hand0.translation.copy()
    if goal == "root":
        root_target += delta
    elif goal == "left_hand":
        hand_target += delta
    else:
        raise KeyError(goal)

    lower = np.r_[q0[:3] - .05, model.lowerPositionLimit[7:] + 1e-4]
    upper = np.r_[q0[:3] + .05, model.upperPositionLimit[7:] - 1e-4]
    # A few imported joints have effectively unbounded limits. Keep the
    # local probe close to the measured supported configuration.
    lower[3:] = np.maximum(lower[3:], q0[7:] - .4)
    upper[3:] = np.minimum(upper[3:], q0[7:] + .4)

    def unpack(x):
        q = q0.copy()
        q[:3] = x[:3]
        q[7:] = x[3:]
        return q

    def residual(x):
        q = unpack(x)
        pin.framesForwardKinematics(model, data, q)
        stance = data.oMf[workspace.support]
        hand = data.oMf[workspace.hand]
        swing = data.oMf[workspace.foot]
        return np.r_[
            120. * (stance.translation - support0.translation),
            20. * pin.log3(support0.rotation.T @ stance.rotation),
            120. * (hand.translation - hand_target),
            8. * pin.log3(hand0.rotation.T @ hand.rotation),
            120. * (swing.translation - foot0.translation),
            15. * pin.log3(foot0.rotation.T @ swing.rotation),
            120. * (q[:3] - root_target),
            .2 * (q[7:] - q0[7:]),
        ]

    guess = q0 if initial_q is None else initial_q
    result = least_squares(residual, np.r_[guess[:3], guess[7:]],
                           bounds=(lower, upper), max_nfev=100,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10)
    q = unpack(result.x)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    stance = data.oMf[workspace.support]
    hand = data.oMf[workspace.hand]
    swing = data.oMf[workspace.foot]
    max_contact_error = max(
        np.linalg.norm(pin.log6(support0.inverse() * stance).vector),
        np.linalg.norm(pin.log6(foot0.inverse() * swing).vector))
    hand_error = np.linalg.norm(hand.translation - hand_target)
    root_error = np.linalg.norm(q[:3] - root_target)
    workspace.jt = np.column_stack((
        pin.getFrameJacobian(model, data, workspace.support,
                             pin.LOCAL_WORLD_ALIGNED).T,
        pin.getFrameJacobian(model, data, workspace.hand,
                             pin.LOCAL_WORLD_ALIGNED)[:3].T))
    static_dynamics = workspace.dynamics(q) if check_dynamics else None
    return {
        "goal": goal, "offset_m": delta.tolist(), "solver_success": bool(result.success),
        "max_fixed_contact_pose_error": float(max_contact_error),
        "hand_target_error_m": float(hand_error),
        "root_target_error_m": float(root_error),
        "max_joint_delta_rad": float(np.max(np.abs(q[7:] - q0[7:]))),
        "static_contact_force_feasible": (static_dynamics is not None
                                           if check_dynamics else None),
        "q_pinocchio": q.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/ik_probe.json"))
    args = p.parse_args()
    workspace = Workspace(json.loads(args.reference.read_text()))
    cases = [("root", np.array(v)) for v in ((.015, 0., 0.),
                                               (0., .015, 0.))]
    cases += [("left_hand", np.array(v)) for v in ((.015, 0., 0.),
                                                    (0., 0., .015))]
    results = [solve_case(workspace, goal, delta) for goal, delta in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print({k: v for k, v in result.items() if k != "q_pinocchio"}, flush=True)


if __name__ == "__main__":
    main()
