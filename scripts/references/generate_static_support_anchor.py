"""Repeat a validated wall-supported pose as a fixed 50 Hz training reference."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path,
                   default=Path("artifacts/foot_reach/motions/val/reach_0005_v0.npz"))
    p.add_argument("--source-manifest", type=Path,
                   default=Path("artifacts/foot_reach/manifest.json"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/static_anchor"))
    p.add_argument("--frames", type=int, default=500)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.frames < 100:
        raise ValueError("Need at least 100 frames for sustained support training")
    source = np.load(args.source)
    manifest = json.loads(args.source_manifest.read_text())
    clip = {}
    for key in source.files:
        value = source[key]
        if key in ("fps", "joint_names", "body_names"):
            clip[key] = value
        elif value.shape[0] == source["dof_pos"].shape[0]:
            clip[key] = np.repeat(value[:1], args.frames, axis=0)
        else:
            raise ValueError(f"Unexpected motion field {key}: {value.shape}")
    for key in ("joint_vel", "joint_acc"):
        clip[key] = np.zeros_like(clip[key])
    for split in ("train", "val"):
        dest = args.output/"motions"/split
        dest.mkdir(parents=True)
        np.savez_compressed(dest/"static_support.npz", **clip)
    manifest = {key: manifest[key] for key in
                ("seed", "reference", "joint_names", "body_names", "sole_offset")}
    manifest.update(fps=50, source_static_pose=str(args.source.resolve()),
                    static_frames=args.frames, frames=2*args.frames,
                    clips=[{"file": f"motions/{split}/static_support.npz",
                            "frames": args.frames, "split": split}
                           for split in ("train", "val")])
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"STATIC_SUPPORT_ANCHOR {args.frames} frames; {len(clip['joint_names'])} joints; "
          f"{len(clip['body_names'])} bodies", flush=True)


if __name__ == "__main__":
    main()
