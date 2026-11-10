"""Center new public goal observations on a verified supported G1 reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path,
                        default=Path("artifacts/foot_reach/motions/train/reach_0074_v1.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/unified_support_reach/goal_norm.json"))
    parser.add_argument("--position-std", type=float, default=.025)
    args = parser.parse_args()
    if not 0 < args.position_std <= .2:
        raise ValueError("position-std must be in (0, 0.2] m")
    data = np.load(args.clip)
    names = list(data["body_names"])
    hand = data["local_body_pos"][:, names.index("left_hand_mimic")]
    stance = names.index("right_ankle_roll_link")
    sole = (data["local_body_pos"][:, stance]
            + Rotation.from_quat(data["local_body_rot"][:, stance]).apply(
                np.array([0., 0., -.03540915])))
    mean = np.r_[np.zeros(3), hand.mean(axis=0), sole.mean(axis=0), np.zeros(3)]
    std = np.r_[np.full(9, args.position_std), np.ones(3)]
    report = {"mean": mean.tolist(), "std": std.tolist(),
              "field_order": ["root_xyz", "left_hand_xyz", "right_sole_xyz",
                              "hand_rightfoot_leftfoot_load"],
              "reference_clip": str(args.clip),
              "reference_sha256": hashlib.sha256(args.clip.read_bytes()).hexdigest(),
              "position_std_m": args.position_std}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
