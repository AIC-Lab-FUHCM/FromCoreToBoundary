"""Data discovery and slice dataset. Helper bodies copied verbatim from
experiments/kas_oa_h4/run_kis_cba_pilot.py (lines 138-615)."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# Resolve repository root
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def resolve_existing_path(requested: Optional[str], candidates: List[Optional[str]]) -> Optional[str]:
    cands: List[str] = []
    if requested:
        cands.append(requested)
    for c in candidates:
        if c and c not in cands:
            cands.append(c)
    for cand in cands:
        cand_expanded = os.path.expanduser(cand)
        if os.path.isfile(cand_expanded):
            return os.path.abspath(cand_expanded)
    return requested


def is_valid_slice_dir(path: Optional[str]) -> bool:
    """Check if path is an existing directory containing at least one .npz slice file."""
    if not path or not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.name.endswith(".npz") and entry.is_file():
                    return True
    except OSError:
        return False
    return False


def is_valid_vol_dir(path: Optional[str]) -> bool:
    """Check if path is an existing directory containing at least one .h5 volume file."""
    if not path or not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.name.endswith((".h5", ".npy.h5", ".npz")) and entry.is_file():
                    return True
    except OSError:
        return False
    return False


def is_valid_slice_list(list_file: str, data_dir: str) -> bool:
    """Verify that entries in the list file actually correspond to files in data_dir."""
    try:
        if not os.path.isdir(data_dir):
            return False
        with open(list_file, "r", encoding="utf-8") as f:
            checked = 0
            for line in f:
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                clean = name[:-4] if name.endswith(".npz") else name
                if os.path.isfile(os.path.join(data_dir, f"{clean}.npz")) or os.path.isfile(os.path.join(data_dir, name)):
                    return True
                checked += 1
                if checked >= 50:
                    break
        return False
    except Exception:
        return False


def is_valid_case_list(list_file: str, test_dir: str) -> bool:
    """Verify that case entries in the list file actually correspond to volume files in test_dir."""
    try:
        if not os.path.isdir(test_dir):
            return False
        with open(list_file, "r", encoding="utf-8") as f:
            checked = 0
            for line in f:
                case = line.strip()
                if not case or case.startswith("#"):
                    continue
                cands = [
                    os.path.join(test_dir, f"{case}.npy.h5"),
                    os.path.join(test_dir, f"{case}.h5"),
                    os.path.join(test_dir, f"{case}.npz"),
                ]
                if any(os.path.isfile(p) for p in cands):
                    return True
                checked += 1
                if checked >= 20:
                    break
        return False
    except Exception:
        return False


def find_train_slice_dir(requested: Optional[str] = None, iad_root: Optional[str] = None) -> str:
    """
    Bulletproof candidate search for BTCV / Synapse train_npz directory.
    Checks requested path, nested zip extractions (data/btcv/data/btcv/train_npz),
    repo roots, standard Colab/Drive paths, and dynamic shallow scan.
    """
    if requested and os.path.isabs(requested) and is_valid_slice_dir(requested):
        return os.path.abspath(requested)

    candidates = []
    if iad_root:
        if requested:
            candidates.append(os.path.join(iad_root, requested))
        candidates.extend([
            os.path.join(iad_root, "data", "btcv", "data", "btcv", "train_npz"),
            os.path.join(iad_root, "data", "btcv", "train_npz"),
            os.path.join(iad_root, "data", "train_npz"),
            os.path.join(iad_root, "train_npz"),
            os.path.join(iad_root, "data", "synapse", "train_npz"),
        ])

    if requested:
        candidates.append(requested)
        candidates.append(os.path.join(REPO_ROOT, requested))

    candidates.extend([
        # BTCV nested extraction path (common when btcv_data.zip extracted into data/btcv)
        os.path.join(REPO_ROOT, "data", "btcv", "data", "btcv", "train_npz"),
        "/content/IAD/data/btcv/data/btcv/train_npz",
        "/content/data/btcv/data/btcv/train_npz",
        "/content/drive/MyDrive/IAD/data/btcv/data/btcv/train_npz",
        "/content/drive/My Drive/IAD/data/btcv/data/btcv/train_npz",
        "I:/My Drive/IAD/data/btcv/data/btcv/train_npz",
        "H:/My Drive/IAD/data/btcv/data/btcv/train_npz",
        # BTCV primary candidates
        os.path.join(REPO_ROOT, "data", "btcv", "train_npz"),
        os.path.join(REPO_ROOT, "..", "data", "btcv", "train_npz"),
        os.path.join(REPO_ROOT, "..", "btcv", "train_npz"),
        os.path.join(REPO_ROOT, "btcv", "train_npz"),
        "/content/IAD/data/btcv/train_npz",
        "/content/data/btcv/train_npz",
        "/content/btcv/train_npz",
        "/content/btcv/data/btcv/train_npz",
        "/content/drive/MyDrive/boost/data/btcv/train_npz",
        "/content/drive/MyDrive/IAD/data/btcv/train_npz",
        "/content/drive/MyDrive/IAD/btcv/train_npz",
        "/content/drive/MyDrive/IAD/btcv/data/btcv/train_npz",
        "/content/drive/MyDrive/btcv/train_npz",
        "/content/drive/MyDrive/btcv/data/btcv/train_npz",
        "/content/drive/My Drive/boost/data/btcv/train_npz",
        "/content/drive/My Drive/IAD/data/btcv/train_npz",
        "I:/My Drive/boost/data/btcv/train_npz",
        "I:/My Drive/IAD/data/btcv/train_npz",
        "I:/My Drive/IAD/btcv/train_npz",
        "H:/My Drive/boost/data/btcv/train_npz",
        "H:/My Drive/IAD/data/btcv/train_npz",
        # Flat / sibling candidates
        os.path.join(REPO_ROOT, "data", "train_npz"),
        os.path.join(REPO_ROOT, "train_npz"),
        "/content/IAD/data/train_npz",
        "/content/IAD/train_npz",
        "/content/data/train_npz",
        "/content/train_npz",
        # Legacy Synapse fallbacks
        os.path.join(REPO_ROOT, "data", "synapse", "train_npz"),
        "/content/IAD/data/synapse/train_npz",
    ])

    for cand in candidates:
        if cand and is_valid_slice_dir(cand):
            return os.path.abspath(cand)

    # Dynamic shallow tree search under local roots only (never walk Google Drive to avoid API stalls)
    search_roots = [
        r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content"]
        if r and os.path.isdir(r) and not ("drive" in r.lower() or "my drive" in r.lower() or r.startswith(("I:", "i:", "H:", "h:", "G:", "g:")))
    ]
    for root in search_roots:
        try:
            for cur_root, dirs, _ in os.walk(root):
                depth = len(Path(cur_root).relative_to(Path(root)).parts)
                if depth > 4:
                    dirs.clear()
                    continue
                if os.path.basename(cur_root) == "train_npz" and is_valid_slice_dir(cur_root):
                    return os.path.abspath(cur_root)
        except Exception:
            continue

    for root in search_roots:
        try:
            for cur_root, dirs, files in os.walk(root):
                depth = len(Path(cur_root).relative_to(Path(root)).parts)
                if depth > 4:
                    dirs.clear()
                    continue
                npz_count = sum(1 for f in files if f.endswith(".npz"))
                if npz_count >= 10:
                    return os.path.abspath(cur_root)
        except Exception:
            continue

    return requested or "data/btcv/train_npz"


def find_test_vol_dir(requested: Optional[str] = None, iad_root: Optional[str] = None) -> str:
    """
    Bulletproof candidate search for BTCV / Synapse test_vol_h5 directory.
    Checks requested path, nested zip extractions (data/btcv/data/btcv/test_vol_h5),
    repo roots, standard Colab/Drive paths, and dynamic shallow scan.
    """
    if requested and os.path.isabs(requested) and is_valid_vol_dir(requested):
        return os.path.abspath(requested)

    candidates = []
    if iad_root:
        if requested:
            candidates.append(os.path.join(iad_root, requested))
        candidates.extend([
            os.path.join(iad_root, "data", "btcv", "data", "btcv", "test_vol_h5"),
            os.path.join(iad_root, "data", "btcv", "test_vol_h5"),
            os.path.join(iad_root, "data", "test_vol_h5"),
            os.path.join(iad_root, "test_vol_h5"),
            os.path.join(iad_root, "data", "synapse", "test_vol_h5"),
        ])

    if requested:
        candidates.append(requested)
        candidates.append(os.path.join(REPO_ROOT, requested))

    candidates.extend([
        # BTCV nested extraction path
        os.path.join(REPO_ROOT, "data", "btcv", "data", "btcv", "test_vol_h5"),
        "/content/IAD/data/btcv/data/btcv/test_vol_h5",
        "/content/data/btcv/data/btcv/test_vol_h5",
        "/content/drive/MyDrive/IAD/data/btcv/data/btcv/test_vol_h5",
        "/content/drive/My Drive/IAD/data/btcv/data/btcv/test_vol_h5",
        "I:/My Drive/IAD/data/btcv/data/btcv/test_vol_h5",
        "H:/My Drive/IAD/data/btcv/data/btcv/test_vol_h5",
        # BTCV primary candidates
        os.path.join(REPO_ROOT, "data", "btcv", "test_vol_h5"),
        os.path.join(REPO_ROOT, "..", "data", "btcv", "test_vol_h5"),
        os.path.join(REPO_ROOT, "..", "btcv", "test_vol_h5"),
        os.path.join(REPO_ROOT, "btcv", "test_vol_h5"),
        "/content/IAD/data/btcv/test_vol_h5",
        "/content/data/btcv/test_vol_h5",
        "/content/btcv/test_vol_h5",
        "/content/btcv/data/btcv/test_vol_h5",
        "/content/drive/MyDrive/boost/data/btcv/test_vol_h5",
        "/content/drive/MyDrive/IAD/data/btcv/test_vol_h5",
        "/content/drive/MyDrive/IAD/btcv/test_vol_h5",
        "/content/drive/MyDrive/IAD/btcv/data/btcv/test_vol_h5",
        "/content/drive/MyDrive/btcv/test_vol_h5",
        "/content/drive/MyDrive/btcv/data/btcv/test_vol_h5",
        "/content/drive/My Drive/boost/data/btcv/test_vol_h5",
        "/content/drive/My Drive/IAD/data/btcv/test_vol_h5",
        "I:/My Drive/boost/data/btcv/test_vol_h5",
        "I:/My Drive/IAD/data/btcv/test_vol_h5",
        "I:/My Drive/IAD/btcv/test_vol_h5",
        "H:/My Drive/boost/data/btcv/test_vol_h5",
        "H:/My Drive/IAD/data/btcv/test_vol_h5",
        # Flat / sibling candidates
        os.path.join(REPO_ROOT, "data", "test_vol_h5"),
        os.path.join(REPO_ROOT, "test_vol_h5"),
        "/content/IAD/data/test_vol_h5",
        "/content/IAD/test_vol_h5",
        "/content/data/test_vol_h5",
        "/content/test_vol_h5",
        # Legacy Synapse fallbacks
        os.path.join(REPO_ROOT, "data", "synapse", "test_vol_h5"),
        "/content/IAD/data/synapse/test_vol_h5",
    ])

    for cand in candidates:
        if cand and is_valid_vol_dir(cand):
            return os.path.abspath(cand)

    # Dynamic shallow tree search under local roots only (never walk Google Drive to avoid API stalls)
    search_roots = [
        r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content"]
        if r and os.path.isdir(r) and not ("drive" in r.lower() or "my drive" in r.lower() or r.startswith(("I:", "i:", "H:", "h:", "G:", "g:")))
    ]
    for root in search_roots:
        try:
            for cur_root, dirs, _ in os.walk(root):
                depth = len(Path(cur_root).relative_to(Path(root)).parts)
                if depth > 4:
                    dirs.clear()
                    continue
                if os.path.basename(cur_root) == "test_vol_h5" and is_valid_vol_dir(cur_root):
                    return os.path.abspath(cur_root)
        except Exception:
            continue

    for root in search_roots:
        try:
            for cur_root, dirs, files in os.walk(root):
                depth = len(Path(cur_root).relative_to(Path(root)).parts)
                if depth > 4:
                    dirs.clear()
                    continue
                h5_count = sum(1 for f in files if f.endswith((".h5", ".npy.h5")))
                if h5_count >= 2:
                    return os.path.abspath(cur_root)
        except Exception:
            continue

    return requested or "data/btcv/test_vol_h5"


def resolve_existing_dir(requested: Optional[str], candidates: List[Optional[str]]) -> Optional[str]:
    cands: List[str] = []
    if requested:
        cands.append(requested)
    for c in candidates:
        if c and c not in cands:
            cands.append(c)
    for cand in cands:
        cand_expanded = os.path.expanduser(cand)
        if os.path.isdir(cand_expanded):
            return os.path.abspath(cand_expanded)
    return requested


def resolve_list_file(
    requested: Optional[str],
    list_name: str,
    iad_root: Optional[str] = None,
    list_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[str]:
    target_name = (
        os.path.basename(requested)
        if (requested and not os.path.isdir(requested) and os.path.splitext(requested)[1])
        else list_name
    )
    names_to_check = [target_name]
    if list_name != target_name:
        names_to_check.append(list_name)

    candidates: List[Optional[str]] = []
    if requested:
        candidates.append(requested)
        if os.path.isdir(requested):
            for n in names_to_check:
                candidates.append(os.path.join(requested, n))

    if list_dir:
        for n in names_to_check:
            candidates.append(os.path.join(list_dir, n))
        candidates.append(list_dir)

    roots = [r for r in [iad_root, REPO_ROOT, os.getcwd()] if r]
    for r in roots:
        for n in names_to_check:
            candidates.extend([
                os.path.join(r, "data", "lists_Synapse", n),
                os.path.join(r, "lists_Synapse", n),
                os.path.join(r, "datasets", "lists_Synapse", n),
                os.path.join(r, "data", n),
                os.path.join(r, n),
            ])

    if data_dir:
        d_abs = os.path.abspath(data_dir)
        d_parent = os.path.dirname(d_abs)
        d_grandparent = os.path.dirname(d_parent)
        for n in names_to_check:
            candidates.extend([
                os.path.join(d_abs, n),
                os.path.join(d_parent, "lists_Synapse", n),
                os.path.join(d_grandparent, "lists_Synapse", n),
                os.path.join(d_grandparent, "data", "lists_Synapse", n),
                os.path.join(d_parent, "data", "lists_Synapse", n),
                os.path.join(d_parent, n),
            ])

    for n in names_to_check:
        candidates.extend([
            f"/content/IAD/data/lists_Synapse/{n}",
            f"/content/IAD/lists_Synapse/{n}",
            f"/content/data/lists_Synapse/{n}",
            f"/content/drive/MyDrive/IAD/data/lists_Synapse/{n}",
            f"/content/drive/MyDrive/IAD/lists_Synapse/{n}",
            f"/content/drive/My Drive/IAD/data/lists_Synapse/{n}",
            f"I:/My Drive/IAD/data/lists_Synapse/{n}",
            f"I:/My Drive/IAD/lists_Synapse/{n}",
            f"H:/My Drive/IAD/data/lists_Synapse/{n}",
        ])

    res = resolve_existing_path(requested, candidates)
    if res and os.path.isfile(res):
        return res

    for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content", "I:/My Drive/IAD"]:
        if not r or not os.path.isdir(r):
            continue
        for n in names_to_check:
            cand = os.path.join(r, "data", "lists_Synapse", n)
            if os.path.isfile(cand):
                return os.path.abspath(cand)
            cand = os.path.join(r, "lists_Synapse", n)
            if os.path.isfile(cand):
                return os.path.abspath(cand)

    return requested


# -----------------------------------------------------------------------------
# Dataset & Model Builders
# -----------------------------------------------------------------------------

def full_augment(image: np.ndarray, label: np.ndarray, output_size=(512, 512)):
    """Original IAD training augmentation (datasets.dataset_synapse.RandomGenerator).

    Same operations, same selection chain and probabilities as RandomGenerator.__call__
    (rot/flip, rotate, scale-crop, elastic, else identity), followed by the same resize to
    `output_size` (cubic for the image, nearest for the label). The augmentation functions are
    imported from datasets/dataset_synapse.py (single source of truth; needs cv2, einops and
    icecream, which that module imports at top level).
    """
    from datasets.dataset_synapse import random_elastic, random_rot_flip, random_rotate, random_scale

    binary_label = (label > 0).astype(np.uint8)
    if random.random() > 0.5:
        image, label, binary_label = random_rot_flip(image, label, binary_label)
    elif random.random() > 0.55:
        image, label, binary_label = random_rotate(image, label, binary_label)
    elif random.random() > 0.25:
        image, label, binary_label = random_scale(image, label, binary_label)
    elif random.random() > 0.15:
        image, label, binary_label = random_elastic(
            image, label, binary_label, image.shape[1] * 2, image.shape[1] * 0.08, image.shape[1] * 0.08
        )

    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
        image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
        label = zoom(label, (output_size[0] / x, output_size[1] / y), order=0)
    return image.astype(np.float32), np.rint(label).astype(np.int64)


class SliceDataset(Dataset):
    """Synapse/BTCV 2D CT slice dataset for training with auto-discovery."""

    def __init__(
        self,
        data_dir: str,
        list_path: Optional[str] = None,
        list_dir: Optional[str] = None,
        iad_root: Optional[str] = None,
        augment: bool = True,
        aug: str = "full",
    ):
        if aug not in ("full", "flip"):
            raise ValueError(f"aug must be 'full' or 'flip', got {aug!r}")
        self.aug = aug
        self.data_dir = find_train_slice_dir(data_dir, iad_root)

        resolved_list = resolve_list_file(
            requested=list_path,
            list_name="train.txt",
            iad_root=iad_root,
            list_dir=list_dir,
            data_dir=self.data_dir,
        )

        valid_list = resolved_list and os.path.isfile(resolved_list) and is_valid_slice_list(resolved_list, self.data_dir)

        names = []
        if valid_list:
            with open(resolved_list, "r", encoding="utf-8") as f:
                raw_names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            for n in raw_names:
                clean = n[:-4] if n.endswith(".npz") else n
                if os.path.isfile(os.path.join(self.data_dir, f"{clean}.npz")) or os.path.isfile(os.path.join(self.data_dir, n)):
                    names.append(clean)
            if names:
                print(f"[SliceDataset] Loaded {len(names)} slices (out of {len(raw_names)} in list) from: {resolved_list} in {self.data_dir}", flush=True)

        if not names:
            # DIRECT DIRECTORY SCAN FALLBACK: Scan data_dir directly for all *.npz files
            if os.path.isdir(self.data_dir):
                npz_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npz")]
                names = sorted([os.path.splitext(f)[0] for f in npz_files])
                if names:
                    print(f"[SliceDataset] Auto-discovered {len(names)} slices directly from {self.data_dir}.", flush=True)

        if not names:
            raise FileNotFoundError(
                f"[SliceDataset] No slices found in '{self.data_dir}' (requested: '{data_dir}') using list '{list_path}' (resolved: '{resolved_list}')"
            )
        self.names = names
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        clean_name = self.names[idx]
        file_path = os.path.join(self.data_dir, f"{clean_name}.npz")
        raw = np.load(file_path)
        image = raw["image"].astype(np.float32)
        label = raw["label"].astype(np.int64)

        if self.augment:
            if self.aug == "full":
                image, label = full_augment(image, label)
            else:
                if random.random() > 0.5:
                    image = np.flip(image, 1).copy()
                    label = np.flip(label, 1).copy()
                k = random.randint(0, 3)
                if k:
                    image = np.rot90(image, k).copy()
                    label = np.rot90(label, k).copy()

        image = np.stack([image, image, image], axis=0)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_name": clean_name,
        }


# -----------------------------------------------------------------------------
# ACDC (4-Class Cardiac MRI, 256x256) Discovery & Dataset
# -----------------------------------------------------------------------------

ACDC_CLASS_NAMES = {
    1: "RV",
    2: "Myo",
    3: "LV",
}

POLYP_CLASS_NAMES = {
    1: "polyp",
}

POLYP_TEST_SETS = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
    "CVC-300",
)

POLYP_PRIMARY_4_SETS = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
)


def find_acdc_split_dir(
    split: str = "train",
    requested: Optional[str] = None,
    iad_root: Optional[str] = None,
) -> str:
    """Locate ACDC split directory ('train', 'valid', or 'test') containing .npz files."""
    if requested and os.path.isabs(requested) and is_valid_slice_dir(requested):
        return os.path.abspath(requested)

    candidates: List[Optional[str]] = []
    if requested:
        candidates.append(requested)
        if iad_root:
            candidates.append(os.path.join(iad_root, requested))
        candidates.append(os.path.join(REPO_ROOT, requested))
        # Also check if requested is the parent ACDC root
        for root_cand in [requested, os.path.join(REPO_ROOT, requested)] + ([os.path.join(iad_root, requested)] if iad_root else []):
            candidates.append(os.path.join(root_cand, split))
            candidates.append(os.path.join(root_cand, "ACDC", split))

    roots = [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content", "/content/drive/MyDrive/IAD", "I:/My Drive/IAD", "J:/My Drive/IAD", "H:/My Drive/IAD"] if r]
    for r in roots:
        candidates.extend([
            os.path.join(r, "data", "ACDC", split),
            os.path.join(r, "data", "acdc", split),
            os.path.join(r, "data", "ACDC", "ACDC", split),
            os.path.join(r, "ACDC", split),
            os.path.join(r, "acdc", split),
        ])

    for cand in candidates:
        if cand and is_valid_slice_dir(cand):
            return os.path.abspath(cand)

    return requested or f"data/ACDC/{split}"


def resolve_acdc_list_file(
    requested: Optional[str],
    split: str = "train",
    iad_root: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[str]:
    list_name = f"{split}.txt"
    candidates: List[Optional[str]] = []
    if requested:
        candidates.append(requested)
        if iad_root:
            candidates.append(os.path.join(iad_root, requested))
        candidates.append(os.path.join(REPO_ROOT, requested))
    if data_dir:
        d_parent = os.path.dirname(os.path.abspath(data_dir))
        candidates.extend([
            os.path.join(d_parent, "lists_ACDC", list_name),
            os.path.join(d_parent, list_name),
        ])
    for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content/drive/MyDrive/IAD"]:
        if not r:
            continue
        candidates.extend([
            os.path.join(r, "data", "ACDC", "lists_ACDC", list_name),
            os.path.join(r, "data", "lists_ACDC", list_name),
            os.path.join(r, "lists_ACDC", list_name),
        ])
    res = resolve_existing_path(requested, candidates)
    if res and os.path.isfile(res):
        return res
    return None


def acdc_augment(image: np.ndarray, label: np.ndarray, output_size=(256, 256)):
    """ACDC training augmentation matching datasets/dataset_ACDC.py RandomGenerator."""
    from scipy import ndimage

    if random.random() > 0.5:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
    elif random.random() > 0.5:
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)

    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
        image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
        label = zoom(label, (output_size[0] / x, output_size[1] / y), order=0)
    return image.astype(np.float32), np.rint(label).astype(np.int64)


class ACDCSliceDataset(Dataset):
    """ACDC 2D cardiac MRI slice dataset (4 classes: 0=BG, 1=RV, 2=Myo, 3=LV) at 256x256."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        list_path: Optional[str] = None,
        iad_root: Optional[str] = None,
        split: str = "train",
        augment: bool = True,
        aug: str = "full",
        img_size: int = 256,
    ):
        if aug not in ("full", "flip"):
            raise ValueError(f"aug must be 'full' or 'flip', got {aug!r}")
        self.aug = aug
        self.split = split
        self.img_size = int(img_size)
        self.data_dir = find_acdc_split_dir(split=split, requested=data_dir, iad_root=iad_root)

        resolved_list = resolve_acdc_list_file(
            requested=list_path,
            split=split,
            iad_root=iad_root,
            data_dir=self.data_dir,
        )
        valid_list = resolved_list and os.path.isfile(resolved_list) and is_valid_slice_list(resolved_list, self.data_dir)

        names: List[str] = []
        if valid_list:
            with open(resolved_list, "r", encoding="utf-8") as f:
                raw_names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            for n in raw_names:
                clean = n[:-4] if n.endswith(".npz") else n
                if os.path.isfile(os.path.join(self.data_dir, f"{clean}.npz")):
                    names.append(clean)
            if names:
                print(
                    f"[ACDCSliceDataset:{split}] Loaded {len(names)} slices from list {resolved_list} in {self.data_dir}",
                    flush=True,
                )

        if not names and os.path.isdir(self.data_dir):
            npz_files = [
                f for f in os.listdir(self.data_dir)
                if f.endswith(".npz") and not f.startswith("._")
            ]
            names = sorted([os.path.splitext(f)[0] for f in npz_files])
            if names:
                print(f"[ACDCSliceDataset:{split}] Auto-discovered {len(names)} slices from {self.data_dir}", flush=True)

        if not names:
            raise FileNotFoundError(
                f"[ACDCSliceDataset:{split}] No .npz slices found in '{self.data_dir}' (requested: '{data_dir}')"
            )

        # Dataset Integrity Guard: verify slices belong to ACDC (labels in {0, 1, 2, 3}, 2D slices)
        for sample_name in names[:5]:
            sample_path = os.path.join(self.data_dir, f"{sample_name}.npz")
            with np.load(sample_path) as raw:
                if "img" not in raw and "image" not in raw:
                    raise ValueError(f"[Dataset Guard] ACDC slice '{sample_path}' missing 'img'/'image' key: {list(raw.keys())}")
                lbl = raw["label"]
                max_cls = int(np.max(lbl))
                min_cls = int(np.min(lbl))
                if min_cls < 0 or max_cls > 3:
                    raise ValueError(
                        f"[Dataset Guard] Dataset mismatch in '{sample_path}': ACDC labels must be in {{0..3}}, "
                        f"but found label range [{min_cls}, {max_cls}] (looks like Synapse/BTCV 9-class data!)."
                    )

        self.names = names
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        clean_name = self.names[idx]
        file_path = os.path.join(self.data_dir, f"{clean_name}.npz")
        with np.load(file_path) as raw:
            img_arr = raw["img"] if "img" in raw else raw["image"]
            image = img_arr.astype(np.float32)
            label = raw["label"].astype(np.int64)

        if self.augment:
            if self.aug == "full":
                image, label = acdc_augment(image, label, output_size=(self.img_size, self.img_size))
            else:
                if random.random() > 0.5:
                    image = np.flip(image, 1).copy()
                    label = np.flip(label, 1).copy()
                k = random.randint(0, 3)
                if k:
                    image = np.rot90(image, k).copy()
                    label = np.rot90(label, k).copy()
                x, y = image.shape
                if x != self.img_size or y != self.img_size:
                    image = zoom(image, (self.img_size / x, self.img_size / y), order=3).astype(np.float32)
                    label = np.rint(zoom(label, (self.img_size / x, self.img_size / y), order=0)).astype(np.int64)
        else:
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3).astype(np.float32)
                label = np.rint(zoom(label, (self.img_size / x, self.img_size / y), order=0)).astype(np.int64)

        image = np.stack([image, image, image], axis=0)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_name": clean_name,
        }


# -----------------------------------------------------------------------------
# Polyp (Binary RGB Endoscopy Segmentation, 256x256) Discovery & Dataset
# -----------------------------------------------------------------------------

def _is_valid_polyp_pair_dir(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    img_dir = os.path.join(path, "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(path, "image")
    msk_dir = os.path.join(path, "masks")
    if not os.path.isdir(msk_dir):
        msk_dir = os.path.join(path, "mask")
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        return False
    try:
        with os.scandir(img_dir) as it:
            return any(
                e.is_file() and not e.name.startswith("._") and e.name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
                for e in it
            )
    except OSError:
        return False


def find_polyp_root(requested: Optional[str] = None, iad_root: Optional[str] = None) -> str:
    """Locate the Polyp dataset root containing TrainDataset/ and TestDataset/."""
    candidates: List[Optional[str]] = []
    if requested:
        candidates.append(requested)
        if iad_root:
            candidates.append(os.path.join(iad_root, requested))
        candidates.append(os.path.join(REPO_ROOT, requested))

    roots = [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content", "/content/drive/MyDrive/IAD", "I:/My Drive/IAD", "J:/My Drive/IAD", "H:/My Drive/IAD"] if r]
    for r in roots:
        candidates.extend([
            os.path.join(r, "data", "dataset"),
            os.path.join(r, "data", "Polyp", "dataset"),
            os.path.join(r, "data", "Polyp"),
            os.path.join(r, "data", "polyp", "dataset"),
            os.path.join(r, "data", "polyp"),
            os.path.join(r, "dataset"),
            os.path.join(r, "Polyp"),
        ])

    for cand in candidates:
        if not cand or not os.path.isdir(cand):
            continue
        if _is_valid_polyp_pair_dir(os.path.join(cand, "TrainDataset")) or _is_valid_polyp_pair_dir(cand):
            return os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "TestDataset")):
            return os.path.abspath(cand)
        # If cand is TrainDataset itself, return its parent if parent has TestDataset
        parent = os.path.dirname(os.path.abspath(cand))
        if os.path.basename(os.path.abspath(cand)) == "TrainDataset" and _is_valid_polyp_pair_dir(cand):
            return parent

    return requested or "data/dataset"


def find_polyp_split_dir(
    split: str = "train",
    subdataset: Optional[str] = None,
    requested: Optional[str] = None,
    iad_root: Optional[str] = None,
) -> str:
    """Resolve the specific Polyp pair directory containing images/ and masks/."""
    if requested and os.path.isdir(requested) and _is_valid_polyp_pair_dir(requested):
        return os.path.abspath(requested)

    root = find_polyp_root(requested, iad_root)
    cands: List[str] = []
    if split == "train":
        cands.extend([
            os.path.join(root, "TrainDataset"),
            os.path.join(root, "PolypSegData"),
            root,
        ])
    else:
        sub = subdataset or "Kvasir"
        cands.extend([
            os.path.join(root, "TestDataset", sub),
            os.path.join(root, sub),
            os.path.join(os.path.dirname(root), "TestDataset", sub),
        ])
        if requested:
            cands.insert(0, os.path.join(requested, sub))
            cands.insert(1, os.path.join(requested, "TestDataset", sub))

    for c in cands:
        if _is_valid_polyp_pair_dir(c):
            return os.path.abspath(c)

    return cands[0] if cands else (requested or "data/dataset/TrainDataset")


class PolypDataset(Dataset):
    """Polyp 2D RGB endoscopy segmentation dataset (1 binary foreground channel at 256x256).

    Matches datasets/dataset_polyp.py preprocessing:
      - Resize to (img_size, img_size) via torchvision.transforms.functional.resize
      - Synchronized RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotation([-90, 90], p=0.5)
      - ImageNet RGB normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      - Binary mask thresholding: (mask / 255.0 >= 0.5) -> {0, 1}
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        subdataset: Optional[str] = None,
        iad_root: Optional[str] = None,
        augment: bool = True,
        aug: str = "full",
        img_size: int = 256,
    ):
        if aug not in ("full", "flip"):
            raise ValueError(f"aug must be 'full' or 'flip', got {aug!r}")
        self.aug = aug
        self.split = split
        self.subdataset = subdataset
        self.img_size = int(img_size)
        self.augment = augment

        self.pair_dir = find_polyp_split_dir(
            split=split,
            subdataset=subdataset,
            requested=data_dir,
            iad_root=iad_root,
        )
        img_dir = os.path.join(self.pair_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(self.pair_dir, "image")
        msk_dir = os.path.join(self.pair_dir, "masks")
        if not os.path.isdir(msk_dir):
            msk_dir = os.path.join(self.pair_dir, "mask")

        if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
            raise FileNotFoundError(
                f"[PolypDataset:{split}] Could not find 'images/' and 'masks/' inside '{self.pair_dir}' (requested: '{data_dir}')"
            )

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if not f.startswith("._") and f.lower().endswith(valid_exts)
        ])
        msk_map = {
            os.path.splitext(f)[0]: os.path.join(msk_dir, f)
            for f in os.listdir(msk_dir)
            if not f.startswith("._") and f.lower().endswith(valid_exts)
        }

        pairs: List[Tuple[str, str, str]] = []
        for img_f in img_files:
            stem = os.path.splitext(img_f)[0]
            if stem in msk_map:
                pairs.append((stem, os.path.join(img_dir, img_f), msk_map[stem]))

        if not pairs:
            raise FileNotFoundError(
                f"[PolypDataset:{split}] No matching (image, mask) pairs found in '{self.pair_dir}'"
            )

        # Dataset Integrity Guard: check first pair is valid RGB image + mask
        from PIL import Image
        with Image.open(pairs[0][1]) as im_check:
            if im_check.mode not in ("RGB", "RGBA", "L"):
                raise ValueError(f"[Dataset Guard] Unexpected image mode {im_check.mode} in {pairs[0][1]}")

        self.pairs = pairs
        label_tag = f"{split}:{subdataset}" if subdataset else split
        print(f"[PolypDataset:{label_tag}] Loaded {len(self.pairs)} image-mask pairs from {self.pair_dir}", flush=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        import torchvision.transforms.functional as TF

        stem, img_path, msk_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        img = TF.resize(img, (self.img_size, self.img_size))
        msk = TF.resize(msk, (self.img_size, self.img_size))

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                msk = TF.hflip(msk)
            if random.random() < 0.5:
                img = TF.vflip(img)
                msk = TF.vflip(msk)
            if self.aug == "full" and random.random() < 0.5:
                deg = random.uniform(-90.0, 90.0)
                img = TF.rotate(img, deg)
                msk = TF.rotate(msk, deg)

        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        mean_t = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        img_t = (img_t - mean_t) / std_t

        msk_np = (np.array(msk, dtype=np.float32) / 255.0 >= 0.5).astype(np.int64)
        msk_t = torch.from_numpy(msk_np)

        return {
            "image": img_t,
            "label": msk_t,
            "case_name": stem,
        }

