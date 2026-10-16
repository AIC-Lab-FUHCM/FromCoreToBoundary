"""Frozen teacher and student builders. Bodies copied verbatim from
experiments/kas_oa_h4/run_kis_cba_pilot.py (lines 616-747)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Resolve repository root and models directory
_REPO_PATH = Path(__file__).resolve().parent.parent
REPO_ROOT = str(_REPO_PATH)
for _p in (_REPO_PATH, _REPO_PATH / "models", _REPO_PATH / "scripts"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets.data import resolve_existing_path


DATASET_MODEL_SPECS = {
    "synapse": {
        "image_size": 512,
        "sam_num_classes": 8,
        "num_classes": 9,
        "mask_tokens_shape": (9, 256),
        "default_teacher_ckpt": "teacher_weights/epoch_299.pth",
    },
    "acdc": {
        "image_size": 256,
        "sam_num_classes": 3,
        "num_classes": 4,
        "mask_tokens_shape": (4, 256),
        "default_teacher_ckpt": "teacher_weights/acdc_tea.pth",
    },
    "polyp": {
        "image_size": 256,
        "sam_num_classes": 0,
        "num_classes": 1,
        "mask_tokens_shape": (1, 256),
        "default_teacher_ckpt": "teacher_weights/polyp_tea.pth",
    },
}

SHAPE_TO_DATASET = {
    (9, 256): "synapse",
    (4, 256): "acdc",
    (1, 256): "polyp",
}


def normalize_dataset_name(dataset: str) -> str:
    ds = str(dataset or "synapse").strip().lower()
    if ds in ("btcv", "synapse"):
        return "synapse"
    if ds in DATASET_MODEL_SPECS:
        return ds
    raise ValueError(f"Unsupported dataset {dataset!r}. Expected one of {sorted(DATASET_MODEL_SPECS)}")


def verify_checkpoint_compatibility(
    state_or_path,
    dataset: str = "synapse",
    role: str = "teacher",
    source_label: Optional[str] = None,
) -> dict:
    """Strictly validate that a checkpoint matches the target dataset and model role (teacher vs student).

    Prevents accidental cross-dataset or cross-role checkpoint loading across Synapse (9 classes),
    ACDC (4 classes), and Polyp (1 binary output channel).
    """
    ds = normalize_dataset_name(dataset)
    expected_spec = DATASET_MODEL_SPECS[ds]
    expected_role = str(role).strip().lower()
    if expected_role not in ("teacher", "student"):
        raise ValueError(f"Invalid role {role!r}; expected 'teacher' or 'student'")

    if isinstance(state_or_path, (str, Path)):
        source_label = source_label or str(state_or_path)
        raw_state = torch.load(str(state_or_path), map_location="cpu", weights_only=False)
    elif isinstance(state_or_path, dict):
        source_label = source_label or "<in-memory state_dict>"
        raw_state = state_or_path
    else:
        raise TypeError(f"Expected checkpoint path or dict, got {type(state_or_path)}")

    state = raw_state
    if isinstance(state, dict):
        for k in ("student", "model", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {source_label} does not contain a valid state_dict dictionary.")

    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    keys = set(cleaned.keys())

    has_lora_ab = any(k.startswith("w_a_") for k in keys) and any(k.startswith("w_b_") for k in keys)
    has_effnet = any("_conv_stem.weight" in k or "_blocks.0." in k for k in keys)

    detected_role = None
    if has_lora_ab and not has_effnet:
        detected_role = "teacher"
    elif has_effnet:
        detected_role = "student"

    if detected_role is not None and detected_role != expected_role:
        raise ValueError(
            f"[Checkpoint Guard] Role mismatch for '{source_label}': "
            f"expected role='{expected_role}', but checkpoint has role='{detected_role}'."
        )

    mask_token_tensor = None
    for mk in ("mask_decoder.mask_tokens.weight", "sam.mask_decoder.mask_tokens.weight"):
        if mk in cleaned:
            mask_token_tensor = cleaned[mk]
            break

    actual_shape = tuple(mask_token_tensor.shape) if mask_token_tensor is not None else None
    detected_dataset = SHAPE_TO_DATASET.get(actual_shape) if actual_shape is not None else None
    expected_shape = expected_spec["mask_tokens_shape"]

    if actual_shape is not None and actual_shape != expected_shape:
        raise ValueError(
            f"[Checkpoint Guard] Dataset mismatch for '{source_label}': "
            f"expected dataset='{ds}' (mask_tokens.weight shape {expected_shape}), "
            f"but checkpoint has mask_tokens.weight shape {actual_shape} "
            f"(detected dataset='{detected_dataset or 'unknown'}')."
        )

    return {
        "dataset": detected_dataset or ds,
        "role": detected_role or expected_role,
        "mask_tokens_shape": actual_shape,
        "num_keys": len(cleaned),
        "path": source_label,
    }


def build_frozen_teacher(
    iad_root: str,
    teacher_ckpt: Optional[str],
    base_ckpt: Optional[str],
    device: torch.device,
    dataset: str = "synapse",
) -> nn.Module:
    """Build and load frozen Teacher (SAM ViT-L with LoRA rank 8) in eval mode."""
    if iad_root not in sys.path:
        sys.path.insert(0, iad_root)

    ds = normalize_dataset_name(dataset)
    spec = DATASET_MODEL_SPECS[ds]

    from segment_anything_Tea import sam_model_registry_tea
    from sam_lora_image_encoder_tea import LoRA_Sam

    base_cands = [
        base_ckpt,
        os.path.join(iad_root, "data", "weights", "sam_vit_l_0b3195.pth"),
        os.path.join(REPO_ROOT, "data", "weights", "sam_vit_l_0b3195.pth"),
        "/content/IAD/data/weights/sam_vit_l_0b3195.pth",
        "/content/drive/MyDrive/IAD/data/weights/sam_vit_l_0b3195.pth",
    ]
    resolved_base = resolve_existing_path(base_ckpt, base_cands)

    sam_tea, _ = sam_model_registry_tea["vit_l"](
        image_size=spec["image_size"],
        num_classes=spec["sam_num_classes"],
        checkpoint=resolved_base,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    net_tea = LoRA_Sam(sam_tea, r=8)

    default_tea_rel = spec["default_teacher_ckpt"]
    default_tea_name = os.path.basename(default_tea_rel)
    req_tea = teacher_ckpt or default_tea_rel

    tea_cands = [
        req_tea,
        os.path.join(iad_root, req_tea),
        os.path.join(REPO_ROOT, req_tea),
        os.path.join(iad_root, default_tea_rel),
        os.path.join(REPO_ROOT, default_tea_rel),
        os.path.join(iad_root, "teacher_weights", default_tea_name),
        os.path.join(REPO_ROOT, "teacher_weights", default_tea_name),
        f"/content/{default_tea_name}",
        f"/content/IAD/{default_tea_rel}",
        f"/content/drive/MyDrive/IAD/{default_tea_rel}",
        f"/content/drive/MyDrive/IAD/{default_tea_name}",
    ]
    if ds == "acdc":
        tea_cands.extend([
            os.path.join(REPO_ROOT, "..", "Pth_s_a_p", "Pth_tea_acdc", "pth.pth"),
            os.path.join(iad_root, "..", "Pth_s_a_p", "Pth_tea_acdc", "pth.pth"),
        ])
    elif ds == "polyp":
        tea_cands.extend([
            os.path.join(REPO_ROOT, "..", "Pth_s_a_p", "Pth_tea_polyp", "PolypSegData", "pth.pth"),
            os.path.join(iad_root, "..", "Pth_s_a_p", "Pth_tea_polyp", "PolypSegData", "pth.pth"),
        ])

    resolved_tea = resolve_existing_path(req_tea, tea_cands)
    if not resolved_tea or not os.path.isfile(resolved_tea):
        raise FileNotFoundError(f"Teacher checkpoint not found for dataset={ds!r} (tried {req_tea})")

    state = torch.load(resolved_tea, map_location="cpu", weights_only=False)
    verify_checkpoint_compatibility(state, dataset=ds, role="teacher", source_label=resolved_tea)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and any(k.startswith("w_a_") for k in state.keys()):
        net_tea.load_lora_parameters(resolved_tea)
    else:
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        net_tea.load_state_dict(cleaned, strict=False)

    net_tea = net_tea.to(device)
    net_tea.eval()
    for p in net_tea.parameters():
        p.requires_grad = False
    print(
        f"[Teacher] Frozen ViT-L Teacher ({ds}, img_size={spec['image_size']}, "
        f"out_ch={spec['num_classes']}) loaded from: {resolved_tea}",
        flush=True,
    )
    return net_tea


def build_student(
    iad_root: str,
    student_ckpt: Optional[str],
    device: torch.device,
    strict_load: bool = True,
    dataset: str = "synapse",
) -> nn.Module:
    """Build and load Student (EfficientNet-B1 backbone + SAM Mask Decoder) with LoRA rank 8."""
    if iad_root not in sys.path:
        sys.path.insert(0, iad_root)

    ds = normalize_dataset_name(dataset)
    spec = DATASET_MODEL_SPECS[ds]

    from segment_anything_Stu import sam_model_registry as sam_model_registry_stu
    from sam_lora_image_encoder_stu import LoRA_Sam as LoRA_Sam_Stu

    sam_stu, _ = sam_model_registry_stu["vit_l"](
        image_size=spec["image_size"],
        num_classes=spec["sam_num_classes"],
        checkpoint=None,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    net_stu = LoRA_Sam_Stu(sam_stu, r=8)

    resolved_stu = None
    if student_ckpt:
        req_name = os.path.basename(student_ckpt)
        stu_cands = [
            student_ckpt,
            os.path.join(iad_root, student_ckpt),
            os.path.join(REPO_ROOT, student_ckpt),
            os.path.join(iad_root, "teacher_weights", req_name),
            os.path.join(REPO_ROOT, "teacher_weights", req_name),
            os.path.join(iad_root, "data", "weights", req_name),
            os.path.join(REPO_ROOT, "data", "weights", req_name),
            f"/content/IAD/{student_ckpt}",
            f"/content/drive/MyDrive/IAD/{student_ckpt}",
            f"/content/IAD/teacher_weights/{req_name}",
            f"/content/drive/MyDrive/IAD/teacher_weights/{req_name}",
            f"/content/IAD/data/weights/{req_name}",
            f"/content/drive/MyDrive/IAD/data/weights/{req_name}",
        ]
        resolved_stu = resolve_existing_path(student_ckpt, stu_cands)
        if not resolved_stu or not os.path.isfile(resolved_stu):
            raise FileNotFoundError(f"[Student] Checkpoint '{student_ckpt}' requested but not found in any search path!")

    if resolved_stu and os.path.isfile(resolved_stu):
        state = torch.load(resolved_stu, map_location="cpu", weights_only=False)
        verify_checkpoint_compatibility(state, dataset=ds, role="student", source_label=resolved_stu)
        if isinstance(state, dict):
            for k in ("student", "model", "state_dict"):
                if k in state:
                    state = state[k]
                    break
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        try:
            missing, unexpected = net_stu.load_state_dict(cleaned, strict=strict_load)
            print(f"[Student Checkpoint Audit] All keys matched successfully with strict={strict_load} from {resolved_stu}!", flush=True)
        except RuntimeError as err:
            missing, unexpected = net_stu.load_state_dict(cleaned, strict=False)
            err_msg = (
                f"[Student Checkpoint Audit ERROR] Key mismatch loading {resolved_stu}!\n"
                f"Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
                f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}\n"
                f"Details: {err}"
            )
            if strict_load:
                raise RuntimeError(err_msg) from err
            else:
                print(f"Warning: {err_msg}", flush=True)
    else:
        print(f"[Student] Starting from scratch for dataset={ds!r} (ImageNet EfficientNet-B1 backbone).", flush=True)

    return net_stu.to(device)
