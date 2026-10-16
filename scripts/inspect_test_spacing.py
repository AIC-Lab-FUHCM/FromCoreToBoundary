#!/usr/bin/env python3
"""Check whether the NIfTI headers can supply real voxel spacing for the BTCV test volumes.

The test .h5 files carry no spacing attributes, so HD95/NSD are computed with the default
(1,1,1) spacing, i.e. in voxels rather than millimetres. This script reads
data/test_nii/<case>_gt.nii.gz, checks that the NIfTI array shape matches the .h5 shape
(up to axis permutation), and reports the spacing found.

Exit code: 0 = every case matched AND at least one spacing differs from (1,1,1) (usable);
           2 = headers matched but are all (1,1,1) (uninformative: metrics stay in voxels);
           1 = some case could not be matched.
With --out, the per-case report is also written as JSON (nothing is written by default).

Usage:
    python scripts/inspect_test_spacing.py [--nii_dir data/test_nii]
        [--h5_dir data/btcv/test_vol_h5] [--out report.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_nii_header(path: str) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """Return (array_shape_zyx, spacing_zyx). Uses SimpleITK if available, else nibabel."""
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(path)
        size_xyz = img.GetSize()
        spacing_xyz = img.GetSpacing()
        return tuple(reversed(size_xyz)), tuple(float(s) for s in reversed(spacing_xyz))
    except ImportError:
        import nibabel as nib

        img = nib.load(path)
        shape_xyz = img.shape[:3]
        zooms_xyz = img.header.get_zooms()[:3]
        return tuple(reversed(shape_xyz)), tuple(float(s) for s in reversed(zooms_xyz))


def match_axes(
    nii_shape_zyx: Tuple[int, ...], nii_spacing_zyx: Tuple[float, ...], h5_shape: Tuple[int, ...]
) -> Tuple[Optional[List[float]], str]:
    """Find the axis permutation that maps NIfTI (z,y,x) onto the h5 array axes.

    Returns (spacing_in_h5_array_order or None, note). If more than one permutation fits and they
    give different spacings, the case is reported as ambiguous and not matched.
    """
    fits = []
    for perm in itertools.permutations(range(3)):
        if tuple(nii_shape_zyx[p] for p in perm) == tuple(h5_shape):
            fits.append(tuple(nii_spacing_zyx[p] for p in perm))
    unique = sorted(set(fits))
    if not unique:
        return None, "shape mismatch"
    if len(unique) > 1:
        return None, f"ambiguous axis mapping: {unique}"
    return list(unique[0]), "ok" if len(fits) == 1 else "ok (equal-length axes, identical spacing)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nii_dir", default=os.path.join(REPO_ROOT, "data", "test_nii"))
    ap.add_argument("--h5_dir", default=os.path.join(REPO_ROOT, "data", "btcv", "test_vol_h5"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result: Dict[str, dict] = {}
    n_ok = 0
    cases = sorted(
        f[: -len(".npy.h5")] for f in os.listdir(args.h5_dir) if f.endswith(".npy.h5")
    )
    for case in cases:
        rec: dict = {"matched": False}
        h5_path = os.path.join(args.h5_dir, f"{case}.npy.h5")
        with h5py.File(h5_path, "r") as f:
            h5_shape = tuple(int(s) for s in f["label"].shape)
        rec["h5_shape"] = list(h5_shape)
        nii_path = os.path.join(args.nii_dir, f"{case}_gt.nii.gz")
        if not os.path.isfile(nii_path):
            rec["note"] = "nii file missing"
        else:
            nii_shape, nii_spacing = read_nii_header(nii_path)
            rec["nii_shape_zyx"] = list(nii_shape)
            rec["nii_spacing_zyx"] = list(nii_spacing)
            spacing, note = match_axes(nii_shape, nii_spacing, h5_shape)
            rec["note"] = note
            if spacing is not None:
                rec["matched"] = True
                rec["spacing_array_order"] = spacing
                n_ok += 1
        result[case] = rec
        print(f"{case}: {rec}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    print(f"\nMatched {n_ok}/{len(cases)} cases")
    if n_ok != len(cases):
        return 1
    informative = any(
        tuple(r["spacing_array_order"]) != (1.0, 1.0, 1.0) for r in result.values() if r["matched"]
    )
    if not informative:
        print("All header spacings are (1,1,1): uninformative. Distance metrics must be labelled voxels.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
