"""Calibrate lambda_A (and the Error-stream weight) from gradient-norm ratios.

For K training microbatches (no optimizer step) with the student/EMA weights of a checkpoint, measure at the
logits the norms of
    g_base = grad(L_base)           (0.8*SGS + 0.2*KIS, the loss CBA is added to)
    g_B    = grad(L_B)              (Boundary stream, after the directional gate)
    g_E    = grad(L_E)              (Error stream)
and report the ratio per unit coefficient:  rho_b = |g_B| / |g_base|,  rho_e = |g_E| / |g_base|.
The lambda that gives a target ratio t is  lambda = t / rho  (Boundary) and  lambda*beta = t / rho_e (Error).

The ratios depend on the training stage, so run it on the shared-warmup checkpoint (epoch 20, where the
lambda ramp starts) and, if available, on a later checkpoint.

Example:
  python scripts/calibrate_lambda.py --arm v2 --ckpt runs/tune/warmup/Arm_V1_KIS/checkpoint_epoch_020.pth \
      --data_dir data/btcv/train_npz --list_path data/lists_Synapse/train_tune.txt --batches 40 \
      --out runs/tune/calib_v2_ep20.json
"""
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.cba_v2.train import (  # noqa: E402
    SimpleConvStudent, SyntheticSliceDataset, build_cba_v2_loss, build_parser, resolve_arm_name, set_seed,
)


def quantiles(x: List[float]) -> dict:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def lambdas_for_targets(rho_median: float, targets: List[float]) -> dict:
    """lambda (per unit rho) reaching each target ratio; None if the stream has no gradient."""
    return {f"{t:g}": (float(t / rho_median) if rho_median > 0 else None) for t in targets}


def calibrate(args) -> dict:
    set_seed(args.seed)
    arm_name = resolve_arm_name(args.arm)
    if arm_name == "Arm_V1_KIS":
        raise ValueError("Arm_V1_KIS has no CBA term to calibrate")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    kis_criterion = None
    teacher = None
    if args.smoke_test:
        student = SimpleConvStudent(in_channels=3, num_classes=args.num_classes).to(device)
        ema = SimpleConvStudent(in_channels=3, num_classes=args.num_classes).to(device)
        dataset = SyntheticSliceDataset(num_samples=args.batch_size * args.batches, num_classes=args.num_classes)
    else:
        from experiments.cba_v2.data import SliceDataset
        from experiments.cba_v2.models import build_frozen_teacher, build_student
        from experiments.cba_v2.losses.kis_loss import KISLoss

        teacher = build_frozen_teacher(iad_root=REPO_ROOT, teacher_ckpt=args.teacher_ckpt,
                                       base_ckpt=args.teacher_base_ckpt, device=device)
        student = build_student(REPO_ROOT, None, device)
        ema = build_student(REPO_ROOT, None, device)
        dataset = SliceDataset(data_dir=args.data_dir, list_path=args.list_path, iad_root=REPO_ROOT,
                               augment=True, aug=args.aug)
        kis_criterion = KISLoss(num_classes=args.num_classes, dice_weight=0.9, cpd_weight_mode=args.cpd_weight_mode,
                                sgs_weight=args.sgs_weight, kis_weight=args.kis_weight).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student"])
    ema.load_state_dict(ckpt["ema_student"] if ckpt.get("ema_student") is not None else ckpt["student"])
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    student.train()

    cba = build_cba_v2_loss(arm_name, args).to(device)
    cba.train()
    ce = torch.nn.CrossEntropyLoss()
    gates_on = (cba.gating_type != "none") or (
        cba.enable_error and cba.error_mode == "contrastive" and cba.error_gating_type != "none")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0 if (os.name == "nt" or args.smoke_test) else 2)
    rho_b, rho_e, gate_b, conflict_b = [], [], [], []
    for b_idx, batch in enumerate(loader):
        if b_idx >= args.batches:
            break
        img = batch["image"].to(device)
        lab = batch["label"].to(device)
        tea_logits = None
        if teacher is not None:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                t_out = teacher(img, multimask_output=8)
            tea_logits = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = student(img, return_logits=True) if args.smoke_test else student(img, True)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if kis_criterion is not None and tea_logits is not None:
                loss_base, _ = kis_criterion(outputs_student=logits, outputs_teacher=tea_logits, labels=lab)
            else:
                loss_base = ce(logits, lab)
            g_base = torch.autograd.grad(loss_base, logits, retain_graph=True)[0].detach()
            with torch.no_grad():
                e_out = ema(img, return_logits=True) if args.smoke_test else ema(img, True)
                fplus_ema = e_out[0] if isinstance(e_out, (tuple, list)) else e_out
            _, _, diag = cba(fplus=logits, y_gt=lab, fplus_ema=fplus_ema, return_diagnostics=True,
                             return_stream_losses=True, compute_autograd_norms=True,
                             g_ref=g_base if (args.gate_ref == "base" and gates_on) else None)
        nb = float(g_base.norm().item())
        if nb <= 0:
            continue
        rho_b.append(float(diag.get("grad_norm_b_autograd", 0.0)) / nb)
        rho_e.append(float(diag.get("grad_norm_e_autograd", 0.0)) / nb)
        gate_b.append(float(diag.get("boundary_gate_mean", 0.0)))
        cb = diag.get("boundary_conflict_rate", {}) or {}
        conflict_b.append(float(sum(cb.values()) / len(cb)) if cb else 0.0)

    targets = [float(t) for t in str(args.targets).split(",") if t.strip()]
    qb, qe = quantiles(rho_b), quantiles(rho_e)
    result = {
        "arm": arm_name,
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "batches": len(rho_b),
        "cba_feature": args.cba_feature,
        "gate_ref": args.gate_ref,
        "gating_type": args.gating_type,
        "rho_b": qb,
        "rho_e": qe,
        "boundary_gate_mean": quantiles(gate_b),
        "boundary_conflict_macro": quantiles(conflict_b),
        "ratio_b_at_current_lambda": float(args.lambda_a * qb["median"]),
        "ratio_e_at_current_lambda_beta": float(args.lambda_a * args.beta * qe["median"]),
        "current": {"lambda_a": args.lambda_a, "beta": args.beta},
        "targets": targets,
        "lambda_for_target_ratio_b": lambdas_for_targets(qb["median"], targets),
        "lambda_beta_for_target_ratio_e": lambdas_for_targets(qe["median"], targets),
    }
    return result


def main(argv=None):
    parser = build_parser()
    parser.description = "Calibrate lambda_A from gradient-norm ratios"
    parser.add_argument("--ckpt", required=True, help="Checkpoint providing student / EMA weights")
    parser.add_argument("--batches", type=int, default=40, help="Number of microbatches to measure")
    parser.add_argument("--targets", type=str, default="0.05,0.1,0.2", help="Target ratios lambda*|g_stream|/|g_base|")
    parser.add_argument("--out", type=str, default=None, help="Write the result as JSON")
    args = parser.parse_args(argv)
    res = calibrate(args)
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
