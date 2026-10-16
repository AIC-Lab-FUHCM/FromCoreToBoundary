#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import copy
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
for _p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

atexit.register(
    lambda: [
        shutil.rmtree(r / "__pycache__", ignore_errors=True)
        for r in (REPO_ROOT, REPO_ROOT.parent)
        if (r / "__pycache__").is_dir()
    ]
)

from models.builder import build_frozen_teacher, build_student, normalize_dataset_name  # noqa: E402
from scripts.evaluation import evaluate_volumes  # noqa: E402
from scripts.train import (  # noqa: E402
    SimpleConvStudent,
    apply_dataset_defaults,
    atomic_json,
    normalize_method_name,
    resolve_file,
    run_training,
    set_seed,
    summarize_eval,
)

CANONICAL_DATASETS: Tuple[str, ...] = ("synapse", "acdc", "polyp")
DEFAULT_SEEDS: Tuple[int, ...] = (2345, 42, 1234)


def parse_seeds(seed: int, seeds_arg: Optional[Sequence[str]]) -> List[int]:
    if not seeds_arg:
        return [int(seed)]
    tokens: List[str] = []
    for item in seeds_arg:
        for sub in str(item).replace(",", " ").split():
            if sub.strip():
                tokens.append(sub.strip().lower())
    if not tokens:
        return [int(seed)]
    if any(t in ("all", "3seeds", "default") for t in tokens):
        return list(DEFAULT_SEEDS)
    parsed: List[int] = []
    for t in tokens:
        val = int(t)
        if val not in parsed:
            parsed.append(val)
    return parsed


def parse_datasets(dataset_arg: str) -> List[str]:
    raw = str(dataset_arg or "synapse").strip().lower()
    if raw in ("all", "3datasets"):
        return list(CANONICAL_DATASETS)
    out: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            ds = normalize_dataset_name(part)
            if ds not in out:
                out.append(ds)
    return out or ["synapse"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--method", type=str, default="ours", choices=["ours", "iad", "dist_plus", "lskd", "sinkd"])
    parser.add_argument("--dataset", type=str, default="synapse")
    parser.add_argument("--data_dir", type=str, default="data/btcv/train_npz")
    parser.add_argument("--list_path", type=str, default="data/lists_Synapse/train.txt")
    parser.add_argument("--test_dir", type=str, default="data/btcv/test_vol_h5")
    parser.add_argument("--test_list", type=str, default="data/lists_Synapse/test_vol.txt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval_role", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, default="teacher_weights/epoch_299.pth")
    parser.add_argument("--teacher_base_ckpt", type=str, default="data/weights/sam_vit_l_0b3195.pth")
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs/ours")
    parser.add_argument("--drive_sync_dir", type=str, default=None)

    parser.set_defaults(
        img_size=0,
        num_classes=9,
        warmup_iters=250,
        weight_decay=0.1,
        smoke_test=False,
        skip_eval=False,
        sgs_weight=0.8,
        kis_weight=0.2,
        lambda_a=0.02,
        beta=0.5,
        m=0.2,
        tau=0.1,
        r=2,
        k_bg=512,
        k_fg=512,
        cache_momentum=0.9,
        curriculum_warmup=20,
        curriculum_ramp=30,
        kis_stage_epochs=200,
        kas_weight=0.1,
        kas_temp=1.0,
        lskd_tau=2.0,
        sinkd_lam=10.0,
        sinkd_iters=5,
        dist_beta=1.0,
        dist_gamma=1.0,
        dist_tau=1.0,
    )
    return parser


def _aggregate_seed_summaries(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [r["final_eval"] for r in seed_results if isinstance(r.get("final_eval"), dict)]
    if not summaries:
        return {"n_seeds": len(seed_results), "seeds": [r.get("seed") for r in seed_results], "metrics_mean": {}, "metrics_std": {}}
    scalar_keys = [k for k, v in summaries[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    mean_dict: Dict[str, float] = {}
    std_dict: Dict[str, float] = {}
    for k in scalar_keys:
        vals = [float(s[k]) for s in summaries if isinstance(s.get(k), (int, float))]
        if vals:
            mean_dict[k] = float(np.mean(vals))
            std_dict[k] = float(np.std(vals, ddof=0))
    return {"n_seeds": len(summaries), "seeds": [r.get("seed") for r in seed_results], "metrics_mean": mean_dict, "metrics_std": std_dict}


def run_train_mode(args: argparse.Namespace) -> Dict[str, Any]:
    method = normalize_method_name(args.method)
    datasets = parse_datasets(args.dataset)
    seeds = parse_seeds(args.seed, args.seeds)

    multi_run = len(datasets) > 1 or len(seeds) > 1
    planned_runs: List[Dict[str, Any]] = []
    for ds in datasets:
        for seed in seeds:
            run_args = copy.deepcopy(args)
            run_args.dataset = ds
            run_args.method = method
            run_args.seed = seed
            if len(datasets) > 1:
                run_args.output_dir = os.path.join(args.output_dir, ds)
                if run_args.drive_sync_dir:
                    run_args.drive_sync_dir = os.path.join(run_args.drive_sync_dir, ds)
            if len(seeds) > 1:
                run_args.output_dir = os.path.join(run_args.output_dir, f"seed_{seed}")
                if run_args.drive_sync_dir:
                    run_args.drive_sync_dir = f"{run_args.drive_sync_dir}_s{seed}"
            apply_dataset_defaults(run_args)
            planned_runs.append({"dataset": ds, "method": method, "seed": seed, "output_dir": run_args.output_dir, "args": run_args})

    if not multi_run:
        return run_training(planned_runs[0]["args"])

    results: List[Dict[str, Any]] = []
    for item in planned_runs:
        res = run_training(item["args"])
        res["seed"] = item["seed"]
        res["dataset"] = item["dataset"]
        res["output_dir"] = item["output_dir"]
        results.append(res)

    by_dataset = {ds: _aggregate_seed_summaries([r for r in results if r.get("dataset") == ds]) for ds in datasets}
    summary_path = os.path.join(args.output_dir, "multiseed_summary.json")
    atomic_json(summary_path, {"method": method, "datasets": datasets, "seeds": seeds, "by_dataset": by_dataset})
    return {"mode": "train", "method": method, "datasets": datasets, "seeds": seeds, "runs": results, "by_dataset": by_dataset, "summary_path": summary_path}


def run_eval_mode(args: argparse.Namespace) -> Dict[str, Any]:
    datasets = parse_datasets(args.dataset)
    if len(datasets) > 1:
        raise ValueError("--mode eval requires a single dataset.")

    eval_args = copy.deepcopy(args)
    eval_args.dataset = datasets[0]
    ds = apply_dataset_defaults(eval_args)
    set_seed(eval_args.seed)
    role = str(eval_args.eval_role or "student").strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() and not eval_args.cpu else "cpu")
    use_amp = (device.type == "cuda" and not eval_args.no_amp)

    ckpt_path = eval_args.ckpt or (eval_args.teacher_ckpt if role == "teacher" else (eval_args.student_ckpt or eval_args.resume))
    out_json = eval_args.eval_output or os.path.join(eval_args.output_dir, f"eval_{role}_{ds}.json")
    preds_dir = os.path.join(eval_args.output_dir, f"preds_{role}_{ds}")

    if getattr(eval_args, "smoke_test", False):
        model = SimpleConvStudent(in_channels=3, num_classes=eval_args.num_classes).to(device)
        resolved_ckpt = resolve_file(ckpt_path) if ckpt_path else None
        if resolved_ckpt and os.path.isfile(resolved_ckpt):
            state = torch.load(resolved_ckpt, map_location=device, weights_only=False)
            if isinstance(state, dict) and "student" in state:
                state = state["student"]
            model.load_state_dict(state, strict=False)
        model.eval()

        tmp_eval_root = tempfile.mkdtemp(prefix=f"eval_smoke_{ds}_")
        vol_dir = os.path.join(tmp_eval_root, "test_vol_h5")
        os.makedirs(vol_dir, exist_ok=True)
        sz = int(eval_args.img_size or 64)
        label = np.zeros((2, sz, sz), dtype=np.uint8)
        label[:, sz // 4 : sz // 2, sz // 4 : sz // 2] = 1
        with h5py.File(os.path.join(vol_dir, "case0001.npy.h5"), "w") as hf:
            hf.create_dataset("image", data=(label > 0).astype(np.float32))
            hf.create_dataset("label", data=label)
        test_list = os.path.join(tmp_eval_root, "test_vol.txt")
        with open(test_list, "w", encoding="utf-8") as fh:
            fh.write("case0001\n")
        try:
            eval_res = evaluate_volumes(
                student_net=model,
                test_dir=vol_dir,
                test_list=test_list,
                device=device,
                batch_size=eval_args.batch_size,
                use_amp=False,
                max_cases=1,
                iad_root=str(REPO_ROOT),
                save_preds_dir=preds_dir,
                tag=f"eval_{role}",
                dataset="synapse",
                img_size=sz,
            )
        finally:
            shutil.rmtree(tmp_eval_root, ignore_errors=True)
    else:
        if role == "teacher":
            model = build_frozen_teacher(iad_root=str(REPO_ROOT), teacher_ckpt=ckpt_path, base_ckpt=eval_args.teacher_base_ckpt, device=device, dataset=ds)
        else:
            if not ckpt_path:
                raise ValueError("--mode eval with --eval_role student requires --ckpt.")
            model = build_student(iad_root=str(REPO_ROOT), student_ckpt=ckpt_path, device=device, strict_load=True, dataset=ds)
            model.eval()

        eval_res = evaluate_volumes(
            student_net=model,
            test_dir=eval_args.test_dir,
            test_list=eval_args.test_list,
            device=device,
            batch_size=eval_args.batch_size,
            use_amp=use_amp,
            iad_root=str(REPO_ROOT),
            save_preds_dir=preds_dir,
            tag=f"eval_{role}",
            dataset=ds,
            img_size=eval_args.img_size,
        )

    summary = summarize_eval(eval_res)
    atomic_json(out_json, eval_res)
    return {"mode": "eval", "dataset": ds, "role": role, "ckpt": ckpt_path, "eval_output": out_json, "summary": summary, "eval": eval_res}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    mode = "eval" if args.eval_only else args.mode
    return run_train_mode(args) if mode == "train" else run_eval_mode(args)


if __name__ == "__main__":
    main()
