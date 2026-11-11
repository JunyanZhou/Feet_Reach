"""Generate standing references with a wider dynamic support margin.

`stable` keeps the original foot-retrieval reach. `easy` shortens the reach so
the object can be placed shallower under the same table for a first policy demo.
"""
import argparse
from pathlib import Path
import demo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("stable", "easy"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for key in demo.KEYS:
        row = list(key)
        if row[0] <= 1.0:
            row[3] = [-0.07, 0.12, 0.0]
        if row[0] <= 15.8:
            stance = [0.02, -0.12, 0.0]
            if len(row) > 7:
                row[7] = stance
            else:
                row.append(stance)
        if args.variant == "easy" and 4.5 <= row[0] <= 14.8:
            foot = list(row[3])
            foot[0] = 0.64 * foot[0] + 0.05
            row[3] = foot
            root = list(row[1])
            if 6.5 <= row[0] <= 12.5:
                root[0] *= 0.75
            row[1] = root
        rows.append(tuple(row))
    demo.KEYS = rows
    output = args.output or ROOT / f"artifacts/table_retrieval_{args.variant}"
    output.mkdir(parents=True, exist_ok=True)
    demo.build_motion(output, fps=60)
    print(f"REFERENCE_VARIANT_READY {args.variant} {output / 'motion.npz'}", flush=True)


if __name__ == "__main__":
    main()
