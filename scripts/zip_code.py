#!/usr/bin/env python3
"""
zip_code.py - Builds self-contained code packages for Colab execution.
Supports packaging:
  - iad_ours_300ep_code.zip (default: for Ours 300-epoch fresh start)
  - iad_cba_pilot_code.zip / iad_kis_cba_pilot_code.zip
Automatically syncs code archives and Colab notebooks to Google Drive (Drive I and Drive H).
"""

import argparse
import os
import sys
import shutil
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
if _HERE.name == "scripts":
    REPO_ROOT = _HERE.parent
elif (_HERE / "IAD" / "main.py").exists():
    REPO_ROOT = (_HERE / "IAD").resolve()
else:
    REPO_ROOT = _HERE


def resolve_bundle_target(target: str, root: Path = REPO_ROOT):
    """Resolve a bundle target relative to root, root/models, root/scripts, or archive fallbacks."""
    for base_dir in (
        root,
        root / "models",
        root / "scripts",
        root.parent / "archive" / "vendor_legacy",
        root.parent / "archive",
    ):
        cand = base_dir / target
        if cand.exists():
            return base_dir, cand
    return None, None

OURS_CURRICULUM_TARGETS = [
    "experiments",
    "segment_anything_Stu",
    "segment_anything_Tea",
    "Synapse_Tool_KD",
    "datasets",
    "lists_Synapse",
    "tools",
    "tests",
    "data/lists_Synapse",
    "data/weights/sam_vit_l_0b3195.pth",
    "teacher_weights/epoch_299.pth",
    "sam_lora_image_encoder_tea.py",
    "sam_lora_image_encoder_stu.py",
    "fast_volume_validator.py",
    "run_ours_curriculum_300ep.py",
    "run_ours_300ep.py",
    "run_ours_curriculum_colab.ipynb",
    "run_ours_300ep_colab.ipynb",
    "requirements.txt",
    "requirements_colab.txt",
    "__init__.py",
]

OURS_300EP_TARGETS = [
    "experiments",
    "segment_anything_Stu",
    "segment_anything_Tea",
    "Synapse_Tool_KD",
    "datasets",
    "lists_Synapse",
    "tools",
    "tests",
    "data/lists_Synapse",
    "data/weights/sam_vit_l_0b3195.pth",
    "teacher_weights/epoch_299.pth",
    "sam_lora_image_encoder_tea.py",
    "sam_lora_image_encoder_stu.py",
    "fast_volume_validator.py",
    "run_ours_300ep.py",
    "create_colab_notebook.py",
    "run_ours_300ep_colab.ipynb",
    "requirements.txt",
    "requirements_colab.txt",
    "__init__.py",
]

CBA_PILOT_TARGETS = [
    "experiments",
    "segment_anything_Stu",
    "segment_anything_Tea",
    "Synapse_Tool_KD",
    "datasets",
    "lists_Synapse",
    "tools",
    "tests",
    "data/lists_Synapse",
    "data/weights/kis_epoch_300.pth",
    "data/weights/sam_vit_l_0b3195.pth",
    "teacher_weights/epoch_299.pth",
    "sam_lora_image_encoder_tea.py",
    "sam_lora_image_encoder_stu.py",
    "fast_volume_validator.py",
    "run_cba_pilot.py",
    "run_benchmark_5configs.py",
    "requirements.txt",
    "requirements_colab.txt",
    "__init__.py",
]

# Code-only bundle for experiments/cba_v2 (weights and the BTCV data stay on Drive; see the notebook).
CBA_V2_TARGETS = [
    "experiments/__init__.py",
    "experiments/cba_v2",
    "segment_anything_Stu",
    "segment_anything_Tea",
    "Synapse_Tool_KD/__init__.py",
    "Synapse_Tool_KD/criterion/__init__.py",
    "Synapse_Tool_KD/criterion/SinKD.py",
    "Synapse_Tool_KD/criterion/lskd.py",
    "Synapse_Tool_KD/distillation_loss.py",
    "ACDC_Tool/__init__.py",
    "ACDC_Tool/distillation_loss.py",
    "datasets",
    "data/lists_Synapse",
    "data/ACDC/lists_ACDC",
    "scripts/make_val_split.py",
    "scripts/calibrate_lambda.py",
    "scripts/inspect_test_spacing.py",
    "scripts/score_tuning.py",
    "scripts/run_tuning.py",
    "scripts/main_common.py",
    "scripts/run_main_kis.py",
    "scripts/run_main_ours.py",
    "scripts/run_main_iad.py",
    "scripts/run_main_lskd.py",
    "scripts/run_main_sinkd.py",
    "scripts/run_main_dist_plus.py",
    "scripts/run_main_sgs_only.py",
    "scripts/make_main_notebook.py",
    "scripts/prepare_5methods_notebooks.py",
    "scripts/prepare_acdc_polyp_3notebooks.py",
    "scripts/zip_code.py",
    "scripts/tools",
    "notebooks/run_cba_v2_main_colab.ipynb",
    "notebooks/run_cba_v2_main_colab_ours_3seeds.ipynb",
    "notebooks/run_cba_v2_main_colab_iad_3seeds.ipynb",
    "notebooks/run_cba_v2_main_colab_lskd_3seeds.ipynb",
    "notebooks/run_cba_v2_main_colab_sinkd_3seeds.ipynb",
    "notebooks/run_cba_v2_main_colab_dist_plus_3seeds.ipynb",
    "notebooks/run_acdc_polyp_parallel_nb1_colab.ipynb",
    "notebooks/run_acdc_polyp_parallel_nb2_colab.ipynb",
    "notebooks/run_acdc_polyp_parallel_nb3_colab.ipynb",
    "tests/conftest.py",
    "tests/test_cba_v2_loss.py",
    "tests/test_cba_v2_train.py",
    "tests/test_cba_v2_recipe.py",
    "tests/test_cba_v2_phase4.py",
    "tests/test_cba_v2_tuning.py",
    "tests/test_cba_v2_tuning_scripts.py",
    "tests/test_cba_v2_run_tuning.py",
    "tests/test_cba_v2_main_scripts.py",
    "tests/test_cba_v2_main_notebook.py",
    "tests/test_cba_v2_ratio_cap.py",
    "tests/test_cba_v2_5methods_multiseed.py",
    "tests/test_acdc_polyp_5methods.py",
    "tests/test_main_cli.py",
    "sam_lora_image_encoder_tea.py",
    "sam_lora_image_encoder_stu.py",
    "main.py",
    "run_cba_v2.py",
    "README.md",
    "requirements.txt",
    "requirements_colab.txt",
]


CBA_V2_ARM5_TARGETS = [
    "experiments",
    "segment_anything_Stu",
    "segment_anything_Tea",
    "Synapse_Tool_KD",
    "datasets",
    "lists_Synapse",
    "tools",
    "tests",
    "data/lists_Synapse",
    "data/weights/kis_epoch_300.pth",
    "data/weights/sam_vit_l_0b3195.pth",
    "teacher_weights/epoch_299.pth",
    "sam_lora_image_encoder_tea.py",
    "sam_lora_image_encoder_stu.py",
    "fast_volume_validator.py",
    "run_cba_v2_arm5.py",
    "run_cba_v2_pilot.py",
    "run_cba_v2_arm5_colab.ipynb",
    "requirements.txt",
    "requirements_colab.txt",
    "__init__.py",
]


def build_bundle(
    zip_name: str = "iad_ours_curriculum_code.zip",
    targets: list = None,
    notebook_name: str = "run_ours_curriculum_colab.ipynb",
):
    if targets is None:
        targets = OURS_CURRICULUM_TARGETS

    out_dir = REPO_ROOT.parent / "colab_upload" if (REPO_ROOT.parent / "colab_upload").is_dir() else REPO_ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_zip = out_dir / zip_name
    print(f"Building {zip_name} from {REPO_ROOT}...")
    files_to_zip = []

    for target in targets:
        base_dir, target_path = resolve_bundle_target(target, REPO_ROOT)
        if target_path is None or not target_path.exists():
            print(f"  ⚠️ Warning: {target} not found under {REPO_ROOT}!")
            continue

        if target_path.is_file():
            arcname = Path(target).as_posix()
            files_to_zip.append((target_path, arcname))
            print(f"  + Added file: {arcname}")
        elif target_path.is_dir():
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                for f in files:
                    if f.endswith(".pyc") or f.endswith(".tmp"):
                        continue
                    full_p = Path(root) / f
                    arcname = full_p.relative_to(base_dir).as_posix()
                    files_to_zip.append((full_p, arcname))
            print(f"  + Added directory: {target}")

    # Deduplicate entries
    dedup = {}
    for fp, arc in files_to_zip:
        dedup[arc] = fp

    print(f"Writing {len(dedup)} entries to {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for arc, fp in sorted(dedup.items()):
            zipf.write(fp, arcname=arc)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] Successfully created {output_zip} ({size_mb:.2f} MB)")

    # Sync to Drive targets
    print("\n--- Synchronizing to Google Drive ---")
    drive_sync_targets = [
        Path(r"I:\My Drive\IAD") / zip_name,
        Path(r"I:\My Drive\boost") / zip_name,
        Path(r"J:\My Drive\IAD") / zip_name,
        Path(r"J:\My Drive\boost") / zip_name,
        Path(r"H:\My Drive\IAD") / zip_name,
        Path(r"H:\My Drive\boost") / zip_name,
    ]

    nb_file = REPO_ROOT / notebook_name
    if not nb_file.exists() and (REPO_ROOT / "notebooks" / notebook_name).exists():
        nb_file = REPO_ROOT / "notebooks" / notebook_name

    for dst in drive_sync_targets:
        try:
            drive_root = dst.drive
            if not Path(f"{drive_root}\\").exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_zip, dst)
            print(f"  [OK] Synced {zip_name} to {dst}")

            # Also sync notebook
            if nb_file.exists():
                dst_nb = dst.parent / notebook_name
                shutil.copy2(nb_file, dst_nb)
                print(f"  [OK] Synced {notebook_name} to {dst_nb}")

            # Also sync far_fp pilot notebook if present
            far_nb = REPO_ROOT / "run_far_fp_pilot_colab.ipynb"
            if not far_nb.exists() and (REPO_ROOT.parent / "run_far_fp_pilot_colab.ipynb").exists():
                far_nb = REPO_ROOT.parent / "run_far_fp_pilot_colab.ipynb"
            if far_nb.exists():
                dst_far_nb = dst.parent / "run_far_fp_pilot_colab.ipynb"
                shutil.copy2(far_nb, dst_far_nb)
                print(f"  [OK] Synced run_far_fp_pilot_colab.ipynb to {dst_far_nb}")

            # Sync weights to Drive data/weights if present
            base_src = REPO_ROOT / "data" / "weights" / "sam_vit_l_0b3195.pth"
            if base_src.exists():
                dst_base = dst.parent / "data" / "weights" / "sam_vit_l_0b3195.pth"
                dst_base.parent.mkdir(parents=True, exist_ok=True)
                if not dst_base.exists() or dst_base.stat().st_size != base_src.stat().st_size:
                    shutil.copy2(base_src, dst_base)
                    print(f"  [OK] Synced sam_vit_l_0b3195.pth to {dst_base}")

            tea_src = REPO_ROOT / "teacher_weights" / "epoch_299.pth"
            if tea_src.exists():
                dst_tea = dst.parent / "teacher_weights" / "epoch_299.pth"
                dst_tea.parent.mkdir(parents=True, exist_ok=True)
                if not dst_tea.exists() or dst_tea.stat().st_size != tea_src.stat().st_size:
                    shutil.copy2(tea_src, dst_tea)
                    print(f"  [OK] Synced epoch_299.pth to {dst_tea}")
        except Exception as e:
            print(f"  [WARN] Could not sync to {dst}: {e}")

    # Also sync to colab_upload directories if present
    for upload_dir in (
        REPO_ROOT.parent / "colab_upload",
        REPO_ROOT / "colab_upload",
        REPO_ROOT.parent / "colab_upload" / "main_5methods_3seeds",
        REPO_ROOT / "colab_upload" / "main_5methods_3seeds",
        REPO_ROOT.parent / "colab_upload" / "acdc_polyp_3notebooks",
        REPO_ROOT / "colab_upload" / "acdc_polyp_3notebooks",
        Path(r"I:\My Drive\IAD\main_5methods_3seeds"),
        Path(r"I:\My Drive\IAD\acdc_polyp_3notebooks"),
        Path(r"J:\My Drive\IAD\main_5methods_3seeds"),
        Path(r"J:\My Drive\IAD\acdc_polyp_3notebooks"),
        Path(r"H:\My Drive\IAD\main_5methods_3seeds"),
        Path(r"H:\My Drive\IAD\acdc_polyp_3notebooks"),
    ):
        try:
            if upload_dir.drive and not Path(f"{upload_dir.drive}\\").exists():
                continue
            if upload_dir.is_dir():
                dst_zip = upload_dir / zip_name
                if output_zip.resolve() != dst_zip.resolve():
                    shutil.copy2(output_zip, dst_zip)
                print(f"  [OK] Synced to colab_upload: {dst_zip}")
        except Exception:
            pass


    print(f"[SUCCESS] Packaging and synchronization completed for {zip_name}!")
    return output_zip


def main():
    parser = argparse.ArgumentParser(description="Package code bundle and sync to Google Drive")
    parser.add_argument(
        "--bundle",
        type=str,
        default="cba_v2",
        choices=["cba_v2", "cba_v2_arm5", "ours_curriculum", "ours_300ep", "cba_pilot"],
        help="Bundle type to package (default: cba_v2; the others are legacy)",
    )
    args = parser.parse_args()

    if args.bundle == "cba_v2":
        build_bundle(
            zip_name="iad_cba_v2_code.zip",
            targets=CBA_V2_TARGETS,
            notebook_name="run_cba_v2_colab.ipynb",
        )
    elif args.bundle == "cba_v2_arm5":
        build_bundle(
            zip_name="iad_cba_v2_arm5_code.zip",
            targets=CBA_V2_ARM5_TARGETS,
            notebook_name="run_cba_v2_arm5_colab.ipynb",
        )
    elif args.bundle == "ours_curriculum":
        build_bundle(
            zip_name="iad_ours_curriculum_code.zip",
            targets=OURS_CURRICULUM_TARGETS,
            notebook_name="run_ours_curriculum_colab.ipynb",
        )
    elif args.bundle == "ours_300ep":
        build_bundle(
            zip_name="iad_ours_300ep_code.zip",
            targets=OURS_300EP_TARGETS,
            notebook_name="run_ours_300ep_colab.ipynb",
        )
    elif args.bundle == "cba_pilot":
        build_bundle(
            zip_name="iad_cba_pilot_code.zip",
            targets=CBA_PILOT_TARGETS,
            notebook_name="run_cba_pilot_colab.ipynb",
        )


if __name__ == "__main__":
    main()
