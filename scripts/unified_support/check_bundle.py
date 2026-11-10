"""Check files needed to evaluate or fine-tune the unified controller.

The default checks use only Python's standard library, so they also work on a
machine before Isaac Lab and PyTorch have been installed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts/unified_support_reach/v2_goal_offset_x10.pt"
WORKSPACE = ROOT / "artifacts/foot_reach/workspace.npz"
VALIDATION = ROOT / "dataset/wall_foot_reach/val"
TRAINING_DATASETS = {
    "wall_foot_reach/train": 36000,
    "unified_support/train": 1812,
    "unified_static_support/train": 500,
    "unified_withdrawn_support/train": 500,
}
CHECKPOINT_SHA256 = "ce04c9f2eb46283698d8dbd258c2f981fde9e40dc593dc546ecbc4847aa21758"
WORKSPACE_SHA256 = "a273bdc3752bf1a0f2b04f9c09e1009953259d09a04c02b152104924020080d4"
DTYPE_BYTES = {"torch.float32": 4, "torch.int32": 4, "torch.int64": 8}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dataset(path, expected_rows):
    for name in ("meta.json", "meta_motion.json", "source_files.json"):
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(f"Missing dataset file: {file}")
        json.loads(file.read_text())

    schema_path = path / "_tensordict/meta.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing dataset schema: {schema_path}")
    schema = json.loads(schema_path.read_text())
    rows = schema["shape"][0]
    if rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} frames in {path}, found {rows}")
    total_bytes = 0
    for name, spec in schema.items():
        if not isinstance(spec, dict) or "dtype" not in spec:
            continue
        dtype = spec["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dataset dtype {dtype} for {name}")
        file = path / "_tensordict" / f"{name}.memmap"
        expected = math.prod(spec["shape"]) * DTYPE_BYTES[dtype]
        if not file.is_file() or file.stat().st_size != expected:
            raise ValueError(f"Missing or truncated dataset file: {file} (expected {expected} bytes)")
        total_bytes += expected
    return {"frames": rows, "memmap_bytes": total_bytes}


def check(checkpoint=CHECKPOINT, training=False):
    checks = ((WORKSPACE, WORKSPACE_SHA256),)
    if checkpoint.resolve() == CHECKPOINT.resolve():
        checks += ((CHECKPOINT, CHECKPOINT_SHA256),)
    elif not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    for path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required bundle file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {path}; expected {expected}, got {actual}")

    with zipfile.ZipFile(WORKSPACE) as archive:
        if "points_root.npy" not in archive.namelist() or archive.testzip():
            raise ValueError(f"Invalid workspace archive: {WORKSPACE}")

    validation = check_dataset(VALIDATION, 8400)
    result = {"checkpoint": str(checkpoint),
            "workspace_sha256": WORKSPACE_SHA256,
            "validation_frames": validation["frames"],
            "validation_memmap_bytes": validation["memmap_bytes"]}
    if checkpoint.resolve() == CHECKPOINT.resolve():
        result["checkpoint_sha256"] = CHECKPOINT_SHA256
    if training:
        result["training_datasets"] = {
            name: check_dataset(ROOT / "dataset" / name, rows)
            for name, rows in TRAINING_DATASETS.items()
        }
    return result


def check_torch(checkpoint):
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for field in ("cfg", "policy", "vecnorm"):
        if field not in state:
            raise ValueError(f"Checkpoint is missing {field}")
    if not state["cfg"].task.command._target_.endswith(".UnifiedSupportReachCommand"):
        raise ValueError("Checkpoint has the wrong command interface")
    if state["policy"].get("last_phase") != "finetune":
        raise ValueError("Checkpoint does not contain the finetuned student")
    actual_dim = state["vecnorm"]["_extra_state"]["policy_sum"].numel()
    if actual_dim != 262:
        raise ValueError(f"Expected 262 policy inputs, got {actual_dim}")
    return {"policy_inputs": actual_dim, "last_phase": "finetune"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true", help="Also load and inspect the PyTorch checkpoint")
    parser.add_argument("--training", action="store_true", help="Also check all bundled curriculum datasets")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="Checkpoint to inspect; the bundled checkpoint is hash-checked")
    args = parser.parse_args()
    result = check(args.checkpoint, args.training)
    if args.torch:
        result.update(check_torch(args.checkpoint))
    print(json.dumps(result, indent=2))
