"""Split the BTCV training cases into a tuning-train set and a tuning-validation set.

Hyperparameters of the method are tuned on the validation cases only; the 12 official test volumes are
never used for tuning. Validation volumes are re-assembled from the training slices (npz) into the same
h5 layout as the test volumes ('image' [D,H,W] float32, 'label' [D,H,W] uint8), so evaluate_volumes()
works unchanged.

Outputs:
  <lists_dir>/train_tune.txt      slice names of the remaining training cases (original order)
  <lists_dir>/val_tune_vol.txt    validation case names
  <h5_dir>/<case>.npy.h5          validation volumes
  <h5_dir>/split_manifest.json    rule, cases, per-organ voxel counts, sha256 of each volume
"""
import argparse
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Sequence

import h5py
import numpy as np

DEFAULT_VAL_CASES = ("case0028", "case0037", "case0021", "case0007")
NUM_ORGANS = 8
SLICE_RE = re.compile(r"^(case\d+)_slice(\d+)$")


def parse_slices(names: Sequence[str]) -> "OrderedDict[str, List[int]]":
    by_case: "OrderedDict[str, List[int]]" = OrderedDict()
    for n in names:
        m = SLICE_RE.match(n)
        if not m:
            raise ValueError(f"Unexpected slice name: {n!r}")
        by_case.setdefault(m.group(1), []).append(int(m.group(2)))
    return by_case


def check_contiguous(by_case: Dict[str, List[int]]) -> None:
    for case, idx in by_case.items():
        if sorted(idx) != list(range(len(idx))):
            raise ValueError(f"{case}: slice indices are not contiguous 0..{len(idx) - 1}; "
                             "a re-assembled volume would be missing slices")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_split(npz_dir: str, train_list: str, val_cases: Sequence[str], h5_dir: str, lists_dir: str,
                min_organ_vox: int = 10000) -> dict:
    with open(train_list, "r", encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    by_case = parse_slices(names)
    check_contiguous(by_case)

    val_cases = list(val_cases)
    if len(set(val_cases)) != len(val_cases):
        raise ValueError("Duplicate validation cases")
    unknown = [c for c in val_cases if c not in by_case]
    if unknown:
        raise ValueError(f"Validation cases not in the training list: {unknown}")

    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(lists_dir, exist_ok=True)
    manifest = {
        "rule": ("Validation cases chosen by hand (see tuning_plan.md): every one of the 8 organs must have at "
                 f">= {min_organ_vox} labelled voxels; slice counts spread over the short/medium/long range."),
        "source_list": os.path.abspath(train_list),
        "min_organ_vox": min_organ_vox,
        "val_cases": {},
    }
    for case in val_cases:
        imgs, labs = [], []
        for i in range(len(by_case[case])):
            z = np.load(os.path.join(npz_dir, f"{case}_slice{i:03d}.npz"))
            imgs.append(z["image"].astype(np.float32))
            lab = z["label"]
            if not np.array_equal(lab, np.round(lab)) or lab.min() < 0 or lab.max() > NUM_ORGANS:
                raise ValueError(f"{case} slice {i}: label values outside integer range 0..{NUM_ORGANS}")
            labs.append(lab.astype(np.uint8))
        image = np.stack(imgs)
        label = np.stack(labs)
        counts = np.bincount(label.ravel(), minlength=NUM_ORGANS + 1)[: NUM_ORGANS + 1]
        weak = [int(k) for k in range(1, NUM_ORGANS + 1) if counts[k] < min_organ_vox]
        if weak:
            raise ValueError(f"{case}: organs {weak} have fewer than {min_organ_vox} voxels")
        out = os.path.join(h5_dir, f"{case}.npy.h5")
        with h5py.File(out, "w") as f:
            f.create_dataset("image", data=image, compression="gzip", compression_opts=1)
            f.create_dataset("label", data=label, compression="gzip", compression_opts=1)
        manifest["val_cases"][case] = {
            "slices": int(image.shape[0]),
            "organ_voxels": {str(k): int(counts[k]) for k in range(1, NUM_ORGANS + 1)},
            "sha256": sha256_file(out),
        }

    val_set = set(val_cases)
    train_lines = [n for n in names if SLICE_RE.match(n).group(1) not in val_set]
    with open(os.path.join(lists_dir, "train_tune.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(train_lines) + "\n")
    with open(os.path.join(lists_dir, "val_tune_vol.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(val_cases)) + "\n")

    manifest["train_tune_slices"] = len(train_lines)
    manifest["train_tune_cases"] = sorted(c for c in by_case if c not in val_set)
    with open(os.path.join(h5_dir, "split_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz_dir", default="data/btcv/train_npz")
    p.add_argument("--train_list", default="data/lists_Synapse/train.txt")
    p.add_argument("--val_cases", default=",".join(DEFAULT_VAL_CASES))
    p.add_argument("--h5_dir", default="data/btcv/val_vol_h5")
    p.add_argument("--lists_dir", default="data/lists_Synapse")
    p.add_argument("--min_organ_vox", type=int, default=10000)
    a = p.parse_args(argv)
    m = build_split(a.npz_dir, a.train_list, [c.strip() for c in a.val_cases.split(",") if c.strip()],
                    a.h5_dir, a.lists_dir, a.min_organ_vox)
    n_val = sum(v["slices"] for v in m["val_cases"].values())
    print(f"validation: {list(m['val_cases'])} ({n_val} slices); train_tune: {len(m['train_tune_cases'])} cases, "
          f"{m['train_tune_slices']} slices")


if __name__ == "__main__":
    main()
