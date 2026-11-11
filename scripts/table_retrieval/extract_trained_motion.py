"""Extract one original wall-reach motion as an exact policy curriculum.

The original student was trained on these motions. This helper avoids
introducing a new out-of-distribution reference while testing table contacts.
"""
import argparse
import json
from pathlib import Path
from active_adaptation.utils.motion import MotionData

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=117)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/wall_foot_reach/train")
    parser.add_argument("--dest", type=Path, default=ROOT / "dataset/table_retrieval_trained_motion/train")
    args = parser.parse_args()
    meta = json.loads((args.source / "meta_motion.json").read_text())
    start, end = meta["starts"][args.index], meta["ends"][args.index]
    source = MotionData.load_memmap(str(args.source))
    segment = source[start:end].clone()
    segment.motion_id.zero_()
    segment.step[:] = segment.step - segment.step[0]
    args.dest.mkdir(parents=True, exist_ok=True)
    segment.memmap(str(args.dest))
    selected = dict(meta)
    selected["starts"] = [0]
    selected["ends"] = [end - start]
    selected["info"] = {}
    (args.dest / "meta_motion.json").write_text(json.dumps(selected))
    (args.dest / "source_files.json").write_text(json.dumps([f"original wall_foot_reach motion index {args.index}"]))
    print(f"EXTRACTED {args.index}: {end-start} frames to {args.dest}")

if __name__ == "__main__":
    main()
