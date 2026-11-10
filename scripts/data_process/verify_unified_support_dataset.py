"""Verify packed supported root/hand clips against source and split metadata."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.motion import MotionDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("artifacts/unified_support_reach/teaching"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("dataset/unified_support"))
    args = parser.parse_args()
    manifest = json.loads((args.source/"manifest.json").read_text())
    assert not manifest["failures"], manifest["failures"]
    assert manifest["fps"] == 50
    assert len(manifest["clips"]) >= 8
    assert max(c["worst_fixed_contact_error_m"] for c in manifest["clips"]) < .0005
    assert max(c["worst_goal_error_m"] for c in manifest["clips"]) < .0005
    for split in ("train", "val"):
        ds = MotionDataset.create_from_path_lazy(str((args.dataset/split).resolve()))
        clips = {c["name"]: c for c in manifest["clips"] if c["split"] == split}
        sources = json.loads((args.dataset/split/"source_files.json").read_text())
        assert len(ds.starts) == len(sources) == len(clips)
        assert ds.joint_names == manifest["joint_names"]
        assert ds.body_names == manifest["body_names"]
        assert ds.data.joint_pos.shape[0] == sum(c["frames"] for c in clips.values())
        for _, value in ds.data.items():
            if value.dtype.is_floating_point:
                assert torch.isfinite(value).all()
        for i, relative_path in enumerate(sources):
            source = args.source/relative_path
            record = clips[source.name]
            raw = np.load(source)
            start = int(ds.starts[i])
            count = record["frames"]
            packed = ds.data[start:start+count]
            np.testing.assert_allclose(packed.joint_pos.numpy(), raw["dof_pos"], atol=1e-7)
            np.testing.assert_allclose(packed.root_pos_w.numpy(), raw["root_pos"], atol=1e-7)
            np.testing.assert_allclose(packed.body_pos_b.numpy(), raw["local_body_pos"], atol=2e-7)
            # Each generated clip must actually vary its intended target.
            if record["goal"] == "root":
                observed = raw["root_pos"][-1] - raw["root_pos"][0]
            else:
                hand_id = ds.body_names.index("left_hand_mimic")
                observed = (packed.body_pos_w[-1, hand_id]
                            - packed.body_pos_w[0, hand_id]).numpy()
            intended = np.asarray(record["offset_m"])
            assert np.dot(observed, intended) > 0.5*np.dot(intended, intended), (
                source, observed, intended)
        print(split, len(clips), "clips verified")
    print("UNIFIED_SUPPORT_DATASET_OK")


if __name__ == "__main__":
    main()
