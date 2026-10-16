from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "models"), os.path.join(REPO_ROOT, "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from losses.kis_loss import KISLoss, lift_binary_logits, structure_loss
from losses.ours_loss import OursLoss
from losses.kd_baselines import BaselineKDLoss, IADKasHead
from models.builder import DATASET_MODEL_SPECS, normalize_dataset_name, verify_checkpoint_compatibility

VALID_METHODS = ("ours", "iad", "dist_plus", "lskd", "sinkd")


def normalize_method_name(method: str) -> str:
    m = str(method or "ours").strip().lower()
    if m in ("ours", "ours_v5", "v5"):
        return "ours"
    if m in ("iad", "iad_200_100"):
        return "iad"
    if m in ("dist_plus", "dist+", "dist"):
        return "dist_plus"
    if m in VALID_METHODS:
        return m
    raise ValueError(f"Unknown method {method!r}. Expected one of {VALID_METHODS}")


def apply_dataset_defaults(args: argparse.Namespace) -> str:
    ds = normalize_dataset_name(getattr(args, "dataset", "synapse"))
    args.dataset = ds
    spec = DATASET_MODEL_SPECS[ds]
    if not getattr(args, "img_size", 0):
        args.img_size = int(spec["image_size"])

    if ds == "acdc":
        args.num_classes = 4
        if getattr(args, "teacher_ckpt", None) in (None, "teacher_weights/epoch_299.pth"):
            args.teacher_ckpt = spec["default_teacher_ckpt"]
        if getattr(args, "data_dir", None) == "data/btcv/train_npz":
            args.data_dir = "data/ACDC/train"
        if getattr(args, "list_path", None) == "data/lists_Synapse/train.txt":
            args.list_path = "data/ACDC/lists_ACDC/train.txt"
        if getattr(args, "test_dir", None) == "data/btcv/test_vol_h5":
            args.test_dir = "data/ACDC/test"
        if getattr(args, "test_list", None) == "data/lists_Synapse/test_vol.txt":
            args.test_list = "data/ACDC/lists_ACDC/test.txt"
    elif ds == "polyp":
        args.num_classes = 1
        if getattr(args, "teacher_ckpt", None) in (None, "teacher_weights/epoch_299.pth"):
            args.teacher_ckpt = spec["default_teacher_ckpt"]
        if getattr(args, "data_dir", None) == "data/btcv/train_npz":
            args.data_dir = "data/dataset/TrainDataset"
        if getattr(args, "test_dir", None) == "data/btcv/test_vol_h5":
            args.test_dir = "data/dataset"
    else:
        args.num_classes = 9
    return ds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def copy_atomic(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = f"{dst}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def resolve_file(path: Optional[str]) -> Optional[str]:
    if not path or str(path).lower() in ("none", "null"):
        return None
    for cand in (path, os.path.join(REPO_ROOT, path)):
        if os.path.isfile(cand):
            return cand
    return None


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def get_curriculum_lambda(epoch_1indexed: int, max_lambda: float = 0.02, warmup_epochs: int = 20, ramp_epochs: int = 30) -> float:
    if warmup_epochs <= 0 and ramp_epochs <= 0:
        return float(max_lambda)
    if epoch_1indexed <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_lambda)
    return float(max_lambda * min(1.0, max(0.0, (epoch_1indexed - warmup_epochs) / float(ramp_epochs))))


def build_lr_fn(total_steps: int, warmup_iters: int):
    def lr_schedule_fn(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return float(step + 1) / float(max(1, warmup_iters))
        decay_steps = max(1, total_steps - warmup_iters)
        prog = min(max(float(step + 1 - warmup_iters) / float(decay_steps), 0.0), 1.0)
        return max(0.0, 1.0 - prog) ** 0.9

    return lr_schedule_fn


def ema_update(student: nn.Module, ema_student: nn.Module, decay: float = 0.99) -> None:
    with torch.no_grad():
        for p_s, p_ema in zip(student.parameters(), ema_student.parameters()):
            p_ema.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)
        for b_s, b_ema in zip(student.buffers(), ema_student.buffers()):
            if b_s.is_floating_point():
                b_ema.data.mul_(decay).add_(b_s.data, alpha=1.0 - decay)
            else:
                b_ema.data.copy_(b_s.data)


def summarize_eval(eval_res: dict) -> dict:
    ov = eval_res["overall"]
    spacings = [tuple(c.get("voxelspacing", (1.0, 1.0, 1.0))) for c in eval_res.get("cases", [])]
    unit = "mm" if any(s != (1.0, 1.0, 1.0) for s in spacings) else "vox"
    res = {
        "eval_n_units": int(ov["primary_n_units"]),
        "eval_dice_raw": float(ov["primary_dice_raw_macro"]),
        "eval_dice_lcc": float(ov["primary_dice_lcc_macro"]),
        "eval_nsd_raw": float(ov["primary_nsd_raw_macro"]),
        "eval_nsd_lcc": float(ov["primary_nsd_lcc_macro"]),
        "eval_hd95_raw": float(ov["primary_hd95_raw_valid_macro"]),
        "eval_hd95_lcc": float(ov["primary_hd95_lcc_valid_macro"]),
        "eval_recall": float(ov["primary_recall_macro"]),
        "eval_precision": float(ov["primary_precision_macro"]),
        "eval_vol_ratio": float(ov["primary_vol_ratio_macro"]),
        "distance_unit": unit,
    }
    if "primary_jc_raw_macro" in ov:
        res["eval_jc_raw"] = float(ov["primary_jc_raw_macro"])
        res["eval_jc_lcc"] = float(ov.get("primary_jc_lcc_macro", ov["primary_jc_raw_macro"]))
    return res


def write_metrics_files(run_dir: str, history: List[dict]) -> None:
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    tmp = f"{jsonl_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for rec in history:
            handle.write(json.dumps(rec, default=str) + "\n")
    os.replace(tmp, jsonl_path)

    columns: List[str] = []
    for rec in history:
        for key, val in rec.items():
            if not isinstance(val, (dict, list)) and key not in columns:
                columns.append(key)
    csv_path = os.path.join(run_dir, "metrics.csv")
    tmp = f"{csv_path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        for rec in history:
            writer.writerow(rec)
    os.replace(tmp, csv_path)


class SyntheticSliceDataset(Dataset):
    def __init__(self, num_samples: int = 8, num_classes: int = 9, img_size: int = 64):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.h = int(img_size) if img_size and img_size > 0 else 64
        self.w = self.h

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        label = np.zeros((self.h, self.w), dtype=np.int64)
        n_fg = max(1, self.num_classes - 1) if self.num_classes > 1 else 1
        sy, sx = max(1, self.h // 8), max(1, self.w // 8)
        if self.num_classes == 1:
            label[sy * 2 : sy * 5, sx * 2 : sx * 5] = 1
        else:
            label[sy : sy * 3, sx : sx * 3] = 1
            if n_fg >= 2:
                label[sy * 2 : sy * 4, sx * 4 : sx * 6] = 2
            label[sy : sy * 2, sx * 5 : sx * 7] = (idx % n_fg) + 1
        image = (label > 0).astype(np.float32) * 0.5 + 0.1
        return {
            "image": torch.from_numpy(np.stack([image, image, image], axis=0)),
            "label": torch.from_numpy(label),
            "case_name": f"synthetic_{idx:03d}",
        }


class SimpleConvStudent(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 9):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x, return_logits: bool = True):
        return (self.backbone(x),)


def run_training(args: argparse.Namespace) -> dict:
    dataset = apply_dataset_defaults(args)
    set_seed(args.seed)
    method = normalize_method_name(getattr(args, "method", "ours"))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    run_dir = os.path.join(args.output_dir, method)
    os.makedirs(run_dir, exist_ok=True)

    teacher = None
    if getattr(args, "smoke_test", False):
        student = SimpleConvStudent(in_channels=3, num_classes=args.num_classes).to(device)
        ema_student = SimpleConvStudent(in_channels=3, num_classes=args.num_classes).to(device)
        ema_student.load_state_dict(student.state_dict())
    else:
        from models.builder import build_frozen_teacher, build_student

        teacher = build_frozen_teacher(
            iad_root=REPO_ROOT,
            teacher_ckpt=args.teacher_ckpt,
            base_ckpt=args.teacher_base_ckpt,
            device=device,
            dataset=dataset,
        )
        stu_ckpt = args.student_ckpt if (args.student_ckpt and str(args.student_ckpt).lower() not in ("none", "null")) else None
        student = build_student(REPO_ROOT, stu_ckpt, device, dataset=dataset)
        ema_student = build_student(REPO_ROOT, stu_ckpt, device, dataset=dataset)
        ema_student.load_state_dict(student.state_dict())

    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False

    if getattr(args, "smoke_test", False):
        train_dataset = SyntheticSliceDataset(
            num_samples=args.batch_size * args.grad_accum_steps * 2,
            num_classes=args.num_classes,
            img_size=args.img_size,
        )
    elif dataset == "acdc":
        from datasets.data import ACDCSliceDataset

        train_dataset = ACDCSliceDataset(data_dir=args.data_dir, list_path=args.list_path, iad_root=REPO_ROOT, split="train", augment=True, img_size=args.img_size)
    elif dataset == "polyp":
        from datasets.data import PolypDataset

        train_dataset = PolypDataset(data_dir=args.data_dir, iad_root=REPO_ROOT, split="train", augment=True, img_size=args.img_size)
    else:
        from datasets.data import SliceDataset

        train_dataset = SliceDataset(data_dir=args.data_dir, list_path=args.list_path, iad_root=REPO_ROOT, augment=True)

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" or getattr(args, "smoke_test", False) else 2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    ours_criterion = None
    if method == "ours":
        c_num = 2 if args.num_classes == 1 else args.num_classes
        ours_criterion = OursLoss(
            m=args.m,
            tau=args.tau,
            r=args.r,
            beta=args.beta,
            k_bg=args.k_bg,
            k_fg=args.k_fg,
            num_classes=c_num,
            feat_dim=c_num,
            cache_momentum=args.cache_momentum,
            feature_space="raw" if args.num_classes == 1 else "centered",
        ).to(device)

    kis_criterion = None
    if method in ("ours", "iad"):
        kis_criterion = KISLoss(
            num_classes=args.num_classes,
            dice_weight=0.9,
            sgs_weight=args.sgs_weight,
            kis_weight=args.kis_weight,
        ).to(device)

    baseline_kd_criterion = None
    if method in ("lskd", "sinkd", "dist_plus"):
        baseline_kd_criterion = BaselineKDLoss(
            method=method,
            num_classes=args.num_classes,
            dice_weight=0.9,
            sgs_weight=args.sgs_weight,
            kd_weight=args.kis_weight,
            lskd_tau=args.lskd_tau,
            sinkd_lam=args.sinkd_lam,
            sinkd_iters=args.sinkd_iters,
            dist_beta=args.dist_beta,
            dist_gamma=args.dist_gamma,
            dist_tau=args.dist_tau,
        ).to(device)

    kas_head = None
    kas_sgs_criterion = None
    kas_optimizer = None
    kis_stage_epochs = int(args.kis_stage_epochs)
    if method == "iad":
        rng_before = capture_rng_state()
        kas_head = IADKasHead(in_channels=256, num_classes=args.num_classes).to(device)
        kas_sgs_criterion = BaselineKDLoss(method="sgs_only", num_classes=args.num_classes, dice_weight=0.9, sgs_weight=1.0, kd_weight=0.0).to(device)
        kas_optimizer = torch.optim.AdamW(kas_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        restore_rng_state(rng_before)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_iters = min(args.warmup_iters, max(1, total_steps // 4)) if total_steps > 0 else 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=build_lr_fn(total_steps, warmup_iters))

    use_amp = (device.type == "cuda" and not args.no_amp)
    is_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if is_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not is_bf16))

    start_epoch = 0
    history: List[dict] = []

    resume_path = args.resume
    if resume_path and str(resume_path).lower() == "auto":
        cands = [os.path.join(run_dir, "latest_checkpoint.pth")]
        if args.drive_sync_dir:
            cands.append(os.path.join(args.drive_sync_dir, "latest_checkpoint.pth"))
        resume_path = next((c for c in cands if os.path.isfile(c)), None)

    if resume_path:
        resolved_resume = resolve_file(resume_path)
        if not resolved_resume:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resolved_resume, map_location=device, weights_only=False)
        if not getattr(args, "smoke_test", False) and "student" in ckpt:
            verify_checkpoint_compatibility(ckpt["student"], dataset=dataset, role="student", source_label=resolved_resume)
        student.load_state_dict(ckpt["student"])
        if ckpt.get("ema_student") is not None:
            ema_student.load_state_dict(ckpt["ema_student"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        if kas_head is not None and ckpt.get("kas_head") is not None:
            kas_head.load_state_dict(ckpt["kas_head"])
        if kas_optimizer is not None and ckpt.get("kas_optimizer") is not None:
            kas_optimizer.load_state_dict(ckpt["kas_optimizer"])
        ours_state = ckpt.get("ours_criterion_state_dict") or ckpt.get("cba_criterion_state_dict")
        if ours_criterion is not None and ours_state is not None:
            ours_criterion.load_state_dict(ours_state)
        restore_rng_state(ckpt.get("rng"))
        start_epoch = int(ckpt.get("epoch", 0))
        history = list(ckpt.get("history", []))

    final_preds_dir = os.path.join(run_dir, "final_preds")
    avg_loss = float("nan")
    final_eval_summary: Optional[dict] = None

    def build_ckpt(epoch_done: int) -> dict:
        return {
            "epoch": epoch_done,
            "student": student.state_dict(),
            "ema_student": ema_student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "ours_criterion_state_dict": ours_criterion.state_dict() if ours_criterion is not None else None,
            "kas_head": kas_head.state_dict() if kas_head is not None else None,
            "kas_optimizer": kas_optimizer.state_dict() if kas_optimizer is not None else None,
            "rng": capture_rng_state(),
            "history": history,
            "method": method,
            "args": vars(args),
        }

    for epoch in range(start_epoch, args.epochs):
        student.train()
        if ours_criterion is not None:
            ours_criterion.train()
        in_kas_stage = (method == "iad") and ((epoch + 1) > kis_stage_epochs)
        if kas_head is not None and in_kas_stage:
            kas_head.train()
            kas_optimizer.zero_grad(set_to_none=True)

        curr_lambda = (
            get_curriculum_lambda(epoch + 1, args.lambda_a, args.curriculum_warmup, args.curriculum_ramp)
            if method == "ours"
            else 0.0
        )
        ours_active = ours_criterion is not None and curr_lambda > 0.0

        running_loss = 0.0
        running_base = 0.0
        running_ours = 0.0
        running_b = 0.0
        running_e = 0.0
        batches = 0

        optimizer.zero_grad(set_to_none=True)
        t_ep_start = time.perf_counter()

        for b_idx, batch in enumerate(loader):
            img = batch["image"].to(device, non_blocking=True)
            lab = batch["label"].to(device, non_blocking=True)

            tea_logits = None
            if teacher is not None and not in_kas_stage:
                with torch.no_grad():
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                            tea_out = teacher(img, multimask_output=8)
                    else:
                        tea_out = teacher(img, multimask_output=8)
                    tea_logits = tea_out[0] if isinstance(tea_out, (tuple, list)) else tea_out
            elif teacher is None and not in_kas_stage:
                with torch.no_grad():
                    ema_out_tea = ema_student(img, return_logits=True) if getattr(args, "smoke_test", False) else ema_student(img, True)
                    tea_logits = (ema_out_tea[0] if isinstance(ema_out_tea, (tuple, list)) else ema_out_tea).detach()

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = student(img, return_logits=True) if getattr(args, "smoke_test", False) else student(img, True)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                if in_kas_stage and kas_head is not None and kas_sgs_criterion is not None:
                    stu_global = out[-1] if isinstance(out, (tuple, list)) else logits
                    loss_sgs, _ = kas_sgs_criterion.forward_sgs(logits, lab)
                    kas_res = kas_head(logits, stu_global, temperature=args.kas_temp)
                    loss_base = loss_sgs + args.kas_weight * kas_res["loss_kas"]
                elif baseline_kd_criterion is not None:
                    loss_base, _ = baseline_kd_criterion(outputs_student=logits, outputs_teacher=tea_logits, labels=lab)
                elif kis_criterion is not None and tea_logits is not None:
                    loss_base, _ = kis_criterion(outputs_student=logits, outputs_teacher=tea_logits, labels=lab)
                else:
                    loss_base = structure_loss(logits, lab) if args.num_classes == 1 else nn.functional.cross_entropy(logits, lab.long())

                loss_ours = torch.tensor(0.0, device=device)
                if ours_active:
                    with torch.no_grad():
                        ema_out = ema_student(img, return_logits=True) if getattr(args, "smoke_test", False) else ema_student(img, True)
                        fplus_ema = ema_out[0] if isinstance(ema_out, (tuple, list)) else ema_out

                    g_base = torch.autograd.grad(loss_base, logits, retain_graph=True)[0].detach()
                    if args.num_classes == 1:
                        ours_fplus = lift_binary_logits(logits)
                        ours_fplus_ema = lift_binary_logits(fplus_ema) if fplus_ema is not None else None
                        ours_g_ref = torch.cat([-g_base, g_base], dim=1)
                    else:
                        ours_fplus = logits
                        ours_fplus_ema = fplus_ema
                        ours_g_ref = g_base

                    loss_ours, diag = ours_criterion(
                        fplus=ours_fplus,
                        y_gt=lab,
                        fplus_ema=ours_fplus_ema,
                        g_ref=ours_g_ref,
                    )
                    running_b += diag["loss_b"]
                    running_e += diag["loss_e"]

                total_loss = loss_base + curr_lambda * loss_ours
                scaled_loss = total_loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (b_idx + 1) % args.grad_accum_steps == 0 or (b_idx + 1) == len(loader):
                if in_kas_stage and kas_optimizer is not None:
                    for g in kas_optimizer.param_groups:
                        g["lr"] = optimizer.param_groups[0]["lr"]
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    if in_kas_stage and kas_optimizer is not None:
                        scaler.step(kas_optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    if in_kas_stage and kas_optimizer is not None:
                        kas_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if in_kas_stage and kas_optimizer is not None:
                    kas_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema_update(student, ema_student, decay=0.99)

            running_loss += float(total_loss.item())
            running_base += float(loss_base.item())
            running_ours += float(loss_ours.item())
            batches += 1

        nb = max(1, batches)
        ep_sec = time.perf_counter() - t_ep_start
        avg_loss = running_loss / nb
        avg_base = running_base / nb
        avg_ours = running_ours / nb
        avg_b = running_b / nb
        avg_e = running_e / nb

        print(
            f"[{method}] Ep {epoch+1:03d}/{args.epochs:03d} | "
            f"Loss: {avg_loss:.4f} (Base: {avg_base:.4f}, Ours: {avg_ours:.4f} [L_B: {avg_b:.4f}, L_E: {avg_e:.4f}], lambda={curr_lambda:.4f}) | "
            f"Time: {ep_sec:.2f}s",
            flush=True,
        )

        ep_record = {
            "epoch": epoch + 1,
            "loss_total": avg_loss,
            "loss_base": avg_base,
            "loss_ours": avg_ours,
            "loss_b": avg_b,
            "loss_e": avg_e,
            "lambda_a": curr_lambda,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": ep_sec,
        }

        is_final = (epoch + 1) == args.epochs
        if is_final and not getattr(args, "smoke_test", False) and not getattr(args, "skip_eval", False):
            from scripts.evaluation import evaluate_volumes

            if ours_criterion is not None:
                ours_criterion.eval()

            eval_res = evaluate_volumes(
                student_net=student,
                test_dir=args.test_dir,
                test_list=args.test_list,
                device=device,
                batch_size=args.batch_size,
                use_amp=use_amp,
                iad_root=REPO_ROOT,
                save_preds_dir=final_preds_dir,
                tag="final",
                dataset=dataset,
                img_size=args.img_size,
            )
            summary = summarize_eval(eval_res)
            ep_record.update(summary)
            atomic_json(os.path.join(run_dir, "eval_final.json"), eval_res)
            final_eval_summary = summary

        history.append(ep_record)
        write_metrics_files(run_dir, history)

        latest_ckpt_path = os.path.join(run_dir, "latest_checkpoint.pth")
        atomic_torch_save(build_ckpt(epoch + 1), latest_ckpt_path)

        if args.drive_sync_dir and ((epoch + 1) % 5 == 0 or is_final):
            try:
                copy_atomic(latest_ckpt_path, os.path.join(args.drive_sync_dir, "latest_checkpoint.pth"))
                for name in ("metrics.jsonl", "metrics.csv", "eval_final.json"):
                    src = os.path.join(run_dir, name)
                    if os.path.isfile(src):
                        copy_atomic(src, os.path.join(args.drive_sync_dir, name))
            except Exception:
                pass

    final_path = os.path.join(run_dir, "final_student.pth")
    atomic_torch_save(
        {
            "student": student.state_dict(),
            "ours_criterion_state_dict": ours_criterion.state_dict() if ours_criterion is not None else None,
            "kas_head": kas_head.state_dict() if kas_head is not None else None,
            "method": method,
            "epoch": args.epochs,
            "args": vars(args),
        },
        final_path,
    )
    if args.drive_sync_dir:
        try:
            copy_atomic(final_path, os.path.join(args.drive_sync_dir, "final_student.pth"))
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "method": method,
        "dataset": dataset,
        "final_loss": avg_loss,
        "final_eval": final_eval_summary,
        "history": history,
    }
