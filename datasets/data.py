from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from torch.utils.data import Dataset

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ACDC_CLASS_NAMES = {1: "RV", 2: "Myo", 3: "LV"}
POLYP_CLASS_NAMES = {1: "polyp"}
POLYP_TEST_SETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")
POLYP_PRIMARY_4_SETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB")


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
    if not path or not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as it:
            return any(e.name.endswith(".npz") and e.is_file() for e in it)
    except OSError:
        return False


def is_valid_vol_dir(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as it:
            return any(e.name.endswith((".h5", ".npy.h5", ".npz")) and e.is_file() for e in it)
    except OSError:
        return False


def is_valid_slice_list(list_file: str, data_dir: str) -> bool:
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
    if requested and os.path.isabs(requested) and is_valid_slice_dir(requested):
        return os.path.abspath(requested)

    candidates = []
    for root in [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content"] if r]:
        if requested:
            candidates.append(os.path.join(root, requested))
        candidates.extend([
            os.path.join(root, "data", "btcv", "data", "btcv", "train_npz"),
            os.path.join(root, "data", "btcv", "train_npz"),
            os.path.join(root, "data", "train_npz"),
            os.path.join(root, "train_npz"),
            os.path.join(root, "data", "synapse", "train_npz"),
        ])
    if requested:
        candidates.insert(0, requested)

    for cand in candidates:
        if cand and is_valid_slice_dir(cand):
            return os.path.abspath(cand)
    return requested or "data/btcv/train_npz"


def find_test_vol_dir(requested: Optional[str] = None, iad_root: Optional[str] = None) -> str:
    if requested and os.path.isabs(requested) and is_valid_vol_dir(requested):
        return os.path.abspath(requested)

    candidates = []
    for root in [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content"] if r]:
        if requested:
            candidates.append(os.path.join(root, requested))
        candidates.extend([
            os.path.join(root, "data", "btcv", "data", "btcv", "test_vol_h5"),
            os.path.join(root, "data", "btcv", "test_vol_h5"),
            os.path.join(root, "data", "test_vol_h5"),
            os.path.join(root, "test_vol_h5"),
            os.path.join(root, "data", "synapse", "test_vol_h5"),
        ])
    if requested:
        candidates.insert(0, requested)

    for cand in candidates:
        if cand and is_valid_vol_dir(cand):
            return os.path.abspath(cand)
    return requested or "data/btcv/test_vol_h5"


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
    names_to_check = [target_name] + ([list_name] if list_name != target_name else [])
    candidates: List[Optional[str]] = [requested] if requested else []
    if list_dir:
        for n in names_to_check:
            candidates.append(os.path.join(list_dir, n))
    for r in [r for r in [iad_root, REPO_ROOT, os.getcwd()] if r]:
        for n in names_to_check:
            candidates.extend([
                os.path.join(r, "data", "lists_Synapse", n),
                os.path.join(r, "lists_Synapse", n),
                os.path.join(r, "data", n),
            ])
    if data_dir:
        d_parent = os.path.dirname(os.path.abspath(data_dir))
        for n in names_to_check:
            candidates.extend([
                os.path.join(d_parent, "lists_Synapse", n),
                os.path.join(d_parent, n),
            ])
    res = resolve_existing_path(requested, candidates)
    return res if (res and os.path.isfile(res)) else requested


def _random_rot_flip(image: np.ndarray, label: np.ndarray, binary_label: np.ndarray):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    binary_label = np.rot90(binary_label, k)
    axis = np.random.randint(0, 2)
    return (
        np.flip(image, axis=axis).copy(),
        np.flip(label, axis=axis).copy(),
        np.flip(binary_label, axis=axis).copy(),
    )


def _random_rotate(image: np.ndarray, label: np.ndarray, binary_label: np.ndarray):
    angle = np.random.randint(-20, 20)
    return (
        ndimage.rotate(image, angle, order=0, reshape=False),
        ndimage.rotate(label, angle, order=0, reshape=False),
        ndimage.rotate(binary_label, angle, order=0, reshape=False),
    )


def _random_scale(image: np.ndarray, label: np.ndarray, binary_label: np.ndarray):
    w, h = image.shape
    scale = 0.2 + random.random() * 0.6
    new_h, new_w = int(h * scale), int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    return image[x : x + new_w, y : y + new_h], label[x : x + new_w, y : y + new_h], binary_label[x : x + new_w, y : y + new_h]


def _random_elastic(image: np.ndarray, label: np.ndarray, binary_label: np.ndarray, alpha: float, sigma: float, alpha_affine: float):
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]
    image = image.astype(np.float32)
    label = label.astype(np.float32)
    binary_label = binary_label.astype(np.float32)
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size,
    ])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    m_mat = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, m_mat, shape_size[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, m_mat, shape_size[::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    binary_label = cv2.warpAffine(binary_label, m_mat, shape_size[::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    image = map_coordinates(image, indices, order=1, mode="constant").reshape(shape)
    label = map_coordinates(label, indices, order=0, mode="constant").reshape(shape)
    binary_label = map_coordinates(binary_label, indices, order=0, mode="constant").reshape(shape)
    return image, label, binary_label


def full_augment(image: np.ndarray, label: np.ndarray, output_size=(512, 512)):
    binary_label = (label > 0).astype(np.uint8)
    if random.random() > 0.5:
        image, label, binary_label = _random_rot_flip(image, label, binary_label)
    elif random.random() > 0.55:
        image, label, binary_label = _random_rotate(image, label, binary_label)
    elif random.random() > 0.25:
        image, label, binary_label = _random_scale(image, label, binary_label)
    elif random.random() > 0.15:
        image, label, binary_label = _random_elastic(
            image, label, binary_label, image.shape[1] * 2, image.shape[1] * 0.08, image.shape[1] * 0.08
        )

    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
        image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
        label = zoom(label, (output_size[0] / x, output_size[1] / y), order=0)
    return image.astype(np.float32), np.rint(label).astype(np.int64)


class SliceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        list_path: Optional[str] = None,
        list_dir: Optional[str] = None,
        iad_root: Optional[str] = None,
        augment: bool = True,
    ):
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
                if os.path.isfile(os.path.join(self.data_dir, f"{clean}.npz")):
                    names.append(clean)
        if not names and os.path.isdir(self.data_dir):
            names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.data_dir) if f.endswith(".npz")])
        if not names:
            raise FileNotFoundError(f"No slices found in '{self.data_dir}'")
        self.names = names
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        clean_name = self.names[idx]
        raw = np.load(os.path.join(self.data_dir, f"{clean_name}.npz"))
        image = raw["image"].astype(np.float32)
        label = raw["label"].astype(np.int64)
        if self.augment:
            image, label = full_augment(image, label)
        image = np.stack([image, image, image], axis=0)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_name": clean_name,
        }


def find_acdc_split_dir(split: str = "train", requested: Optional[str] = None, iad_root: Optional[str] = None) -> str:
    if requested and os.path.isabs(requested) and is_valid_slice_dir(requested):
        return os.path.abspath(requested)

    candidates: List[Optional[str]] = []
    if requested:
        candidates.extend([requested, os.path.join(REPO_ROOT, requested), os.path.join(requested, split)])
        if iad_root:
            candidates.append(os.path.join(iad_root, requested))
    for r in [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content"] if r]:
        candidates.extend([
            os.path.join(r, "data", "ACDC", split),
            os.path.join(r, "data", "acdc", split),
            os.path.join(r, "ACDC", split),
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
    candidates: List[Optional[str]] = [requested] if requested else []
    if data_dir:
        d_parent = os.path.dirname(os.path.abspath(data_dir))
        candidates.extend([os.path.join(d_parent, "lists_ACDC", list_name), os.path.join(d_parent, list_name)])
    for r in [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD"] if r]:
        candidates.extend([
            os.path.join(r, "data", "ACDC", "lists_ACDC", list_name),
            os.path.join(r, "data", "lists_ACDC", list_name),
        ])
    res = resolve_existing_path(requested, candidates)
    return res if (res and os.path.isfile(res)) else None


def acdc_augment(image: np.ndarray, label: np.ndarray, output_size=(256, 256)):
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
    def __init__(
        self,
        data_dir: Optional[str] = None,
        list_path: Optional[str] = None,
        iad_root: Optional[str] = None,
        split: str = "train",
        augment: bool = True,
        img_size: int = 256,
    ):
        self.split = split
        self.img_size = int(img_size)
        self.data_dir = find_acdc_split_dir(split=split, requested=data_dir, iad_root=iad_root)
        resolved_list = resolve_acdc_list_file(requested=list_path, split=split, iad_root=iad_root, data_dir=self.data_dir)
        valid_list = resolved_list and os.path.isfile(resolved_list) and is_valid_slice_list(resolved_list, self.data_dir)

        names: List[str] = []
        if valid_list:
            with open(resolved_list, "r", encoding="utf-8") as f:
                raw_names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            for n in raw_names:
                clean = n[:-4] if n.endswith(".npz") else n
                if os.path.isfile(os.path.join(self.data_dir, f"{clean}.npz")):
                    names.append(clean)
        if not names and os.path.isdir(self.data_dir):
            names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.data_dir) if f.endswith(".npz") and not f.startswith("._")])
        if not names:
            raise FileNotFoundError(f"No ACDC .npz slices found in '{self.data_dir}'")

        self.names = names
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        clean_name = self.names[idx]
        with np.load(os.path.join(self.data_dir, f"{clean_name}.npz")) as raw:
            img_arr = raw["img"] if "img" in raw else raw["image"]
            image = img_arr.astype(np.float32)
            label = raw["label"].astype(np.int64)

        if self.augment:
            image, label = acdc_augment(image, label, output_size=(self.img_size, self.img_size))
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


def _is_valid_polyp_pair_dir(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    img_dir = os.path.join(path, "images") if os.path.isdir(os.path.join(path, "images")) else os.path.join(path, "image")
    msk_dir = os.path.join(path, "masks") if os.path.isdir(os.path.join(path, "masks")) else os.path.join(path, "mask")
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
    candidates: List[Optional[str]] = []
    if requested:
        candidates.extend([requested, os.path.join(REPO_ROOT, requested)])
        if iad_root:
            candidates.append(os.path.join(iad_root, requested))
    for r in [r for r in [iad_root, REPO_ROOT, os.getcwd(), "/content/IAD", "/content"] if r]:
        candidates.extend([
            os.path.join(r, "data", "dataset"),
            os.path.join(r, "data", "Polyp", "dataset"),
            os.path.join(r, "data", "Polyp"),
            os.path.join(r, "dataset"),
        ])
    for cand in candidates:
        if not cand or not os.path.isdir(cand):
            continue
        if _is_valid_polyp_pair_dir(os.path.join(cand, "TrainDataset")) or _is_valid_polyp_pair_dir(cand):
            return os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "TestDataset")):
            return os.path.abspath(cand)
        if os.path.basename(os.path.abspath(cand)) == "TrainDataset" and _is_valid_polyp_pair_dir(cand):
            return os.path.dirname(os.path.abspath(cand))
    return requested or "data/dataset"


def find_polyp_split_dir(
    split: str = "train",
    subdataset: Optional[str] = None,
    requested: Optional[str] = None,
    iad_root: Optional[str] = None,
) -> str:
    if requested and os.path.isdir(requested) and _is_valid_polyp_pair_dir(requested):
        return os.path.abspath(requested)
    root = find_polyp_root(requested, iad_root)
    if split == "train":
        cands = [os.path.join(root, "TrainDataset"), os.path.join(root, "PolypSegData"), root]
    else:
        sub = subdataset or "Kvasir"
        cands = [os.path.join(root, "TestDataset", sub), os.path.join(root, sub)]
        if requested:
            cands = [os.path.join(requested, sub), os.path.join(requested, "TestDataset", sub)] + cands
    for c in cands:
        if _is_valid_polyp_pair_dir(c):
            return os.path.abspath(c)
    return cands[0] if cands else (requested or "data/dataset/TrainDataset")


class PolypDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        subdataset: Optional[str] = None,
        iad_root: Optional[str] = None,
        augment: bool = True,
        img_size: int = 256,
    ):
        self.split = split
        self.subdataset = subdataset
        self.img_size = int(img_size)
        self.augment = augment
        self.pair_dir = find_polyp_split_dir(split=split, subdataset=subdataset, requested=data_dir, iad_root=iad_root)

        img_dir = os.path.join(self.pair_dir, "images") if os.path.isdir(os.path.join(self.pair_dir, "images")) else os.path.join(self.pair_dir, "image")
        msk_dir = os.path.join(self.pair_dir, "masks") if os.path.isdir(os.path.join(self.pair_dir, "masks")) else os.path.join(self.pair_dir, "mask")
        if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
            raise FileNotFoundError(f"Could not find images/ and masks/ inside '{self.pair_dir}'")

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        img_files = sorted([f for f in os.listdir(img_dir) if not f.startswith("._") and f.lower().endswith(valid_exts)])
        msk_map = {
            os.path.splitext(f)[0]: os.path.join(msk_dir, f)
            for f in os.listdir(msk_dir)
            if not f.startswith("._") and f.lower().endswith(valid_exts)
        }
        self.pairs: List[Tuple[str, str, str]] = [
            (os.path.splitext(f)[0], os.path.join(img_dir, f), msk_map[os.path.splitext(f)[0]])
            for f in img_files
            if os.path.splitext(f)[0] in msk_map
        ]
        if not self.pairs:
            raise FileNotFoundError(f"No matching (image, mask) pairs found in '{self.pair_dir}'")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        import torchvision.transforms.functional as TF

        stem, img_path, msk_path = self.pairs[idx]
        img = TF.resize(Image.open(img_path).convert("RGB"), (self.img_size, self.img_size))
        msk = TF.resize(Image.open(msk_path).convert("L"), (self.img_size, self.img_size))

        if self.augment:
            if random.random() < 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if random.random() < 0.5:
                img, msk = TF.vflip(img), TF.vflip(msk)
            if random.random() < 0.5:
                deg = random.uniform(-90.0, 90.0)
                img, msk = TF.rotate(img, deg), TF.rotate(msk, deg)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        mean_t = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        img_t = (img_t - mean_t) / std_t
        msk_t = torch.from_numpy((np.array(msk, dtype=np.float32) / 255.0 >= 0.5).astype(np.int64))
        return {
            "image": img_t,
            "label": msk_t,
            "case_name": stem,
        }
