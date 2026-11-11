"""Extract the low-table motion after hand support is established.

The saved trajectory supplies training targets and an initial state.  Evaluation
still runs the free-base robot and a dynamic object in Isaac physics.
"""
import argparse
import json
from pathlib import Path

from active_adaptation.utils.motion import MotionData

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/table_retrieval_easy/train")
    parser.add_argument("--dest", type=Path, default=ROOT / "dataset/table_retrieval_supported/train")
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--end", type=int, default=750)
    args = parser.parse_args()
    meta = json.loads((args.source / "meta_motion.json").read_text())
    if len(meta["starts"]) != 1 or not 0 <= args.start < args.end <= meta["ends"][0]:
        raise ValueError("Expected a single motion and a valid crop")
    data = MotionData.load_memmap(str(args.source))
    clip = data[args.start:args.end].clone()
    clip.motion_id.zero_()
    clip.step[:] = clip.step - clip.step[0].clone()
    args.dest.mkdir(parents=True, exist_ok=True)
    clip.memmap(str(args.dest))
    meta["starts"] = [0]
    meta["ends"] = [args.end - args.start]
    meta["info"] = {}
    (args.dest / "meta_motion.json").write_text(json.dumps(meta))
    (args.dest / "source_files.json").write_text(json.dumps([f"table_retrieval_easy/train [{args.start}:{args.end}]"]))
    print(f"SUPPORTED_CLIP_READY {args.dest} {args.end - args.start} frames")


if __name__ == "__main__":
    main()
