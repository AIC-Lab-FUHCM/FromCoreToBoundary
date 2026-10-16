# GatedCoreToBoundary (CBA v2): Gated Core-to-Boundary Assimilation & Error-Driven Distillation for Medical Image Segmentation

Unified multi-dataset and multi-seed Knowledge Distillation benchmark (`CBA v2` / `Ours` vs. `IAD`, `DIST+`, `LSKD`, `SinKD`) across **Synapse (BTCV)**, **ACDC**, and **Polyp** via a single top-level [`main.py`](./main.py) entry point.

---

## 🛠️ Environment Setup

### 1. Local / Server Environment (Recommended: Python $\ge 3.10$, PyTorch $\ge 2.0$)

```bash
# Install all required dependencies (including PyTorch, timm, MedPy, SimpleITK, nibabel, h5py, Pillow, pytest)
pip install -r requirements.txt
```

### 2. Google Colab Environment (where PyTorch with CUDA is pre-installed)

```bash
pip install -r scripts/requirements_colab.txt
```

---

## 📂 Repository Structure

```text
GatedCoreToBoundary/
├── main.py                        # Unified CLI entry point (train, reproduce, eval, tune, package)
├── requirements.txt               # Full Python dependencies (Local/Server + Pytest)
├── README.md                      # Project documentation & CLI reference
├── LICENSE                        # Apache-2.0 License
├── .gitignore                     # Git ignore rules for datasets, checkpoints, and run artifacts
├── data/                          # Datasets (btcv/synapse, ACDC, dataset/Polyp) & split lists
├── datasets/                      # Dataset loaders (dataset_synapse.py, dataset_ACDC.py, dataset_polyp.py)
├── experiments/
│   └── cba_v2/                    # Main benchmark pipeline (5 KD methods × 3 datasets × multi-seed)
│       ├── train.py               # Controlled training & checkpoint/curriculum/EMA loop
│       ├── evaluation.py          # 3D volume (Synapse 12 vols / ACDC 40 vols) & 5-dataset (Polyp) evaluation
│       ├── models.py              # Teacher (SAM ViT-L LoRA) & Student (EfficientNet-B1 + SAM) builders
│       ├── data.py                # Dataset & list resolvers (Synapse, ACDC, Polyp)
│       └── losses/                # CBAv2Loss, KISLoss, BaselineKDLoss (IAD/KAS, DIST+, LSKD, SinKD), FarFPLoss
├── models/                        # Model architectures & reference KD criteria
│   ├── segment_anything_Tea/      # Teacher SAM ViT-L architecture
│   ├── segment_anything_Stu/      # Student EfficientNet-B1 + SAM Mask Decoder architecture
│   ├── sam_lora_image_encoder_tea.py  # Teacher LoRA adapter
│   ├── sam_lora_image_encoder_stu.py  # Student adapter wrapper
│   ├── Synapse_Tool_KD/           # Reference KD criteria (SinKD, LSKD, DIST+)
│   └── ACDC_Tool/                 # Reference ACDC distillation loss utilities
├── notebooks/                     # Generated Google Colab notebooks (Synapse 3-seed & ACDC/Polyp parallel)
├── scripts/                       # Reproducible benchmark launchers, tuning drivers, packaging & diagnostic tools
│   ├── main_common.py             # Shared 300-epoch launcher with epoch-20 warmup & Drive resume
│   ├── run_cba_v2.py              # Direct trainer entry point (forwards to experiments/cba_v2/train.py)
│   ├── run_main_ours.py           # Launcher for Ours (Arm_V5_Dual_CEHard / Arm_V2_Boundary)
│   ├── run_main_iad.py            # Launcher for 2-Stage IAD (200 KIS + 100 KAS)
│   ├── run_main_dist_plus.py      # Launcher for DIST+
│   ├── run_main_lskd.py           # Launcher for LSKD
│   ├── run_main_sinkd.py          # Launcher for SinKD
│   ├── run_main_kis.py            # Launcher for Stage-1 KIS baseline
│   ├── run_main_sgs_only.py       # Launcher for supervised-only (SGS) baseline
│   ├── prepare_5methods_notebooks.py      # Generates 5 Synapse 3-seed Colab notebooks
│   ├── prepare_acdc_polyp_3notebooks.py   # Generates 3 parallel ACDC+Polyp Colab notebooks
│   ├── run_tuning.py              # Hyperparameter tuning driver (Stages warmup, ref, A, B, C)
│   ├── zip_code.py                # Self-contained Colab bundle builder & Google Drive sync utility
│   ├── requirements_colab.txt     # Colab-optimized dependencies
│   └── tools/                     # Post-processing diagnostics, paired bootstrap & dataset packaging tools
├── teacher_weights/               # Pre-trained Teacher & Student checkpoints (gitignored)
└── tests/                         # Automated Pytest verification suite
```

---

## 🚀 Running the Code (Unified `main.py` Entry Point)

All training, benchmark reproduction, standalone checkpoint evaluation, hyperparameter tuning, and Colab bundle packaging are controlled through `main.py`.

### 1. Discover Available Parameters, Methods & Datasets

```bash
# View all 10 CLI parameter groups
python main.py --help

# List all supported KD methods (ours, iad, dist_plus, lskd, sinkd) and ablation arms (kis, v2..v4, sgs_only)
python main.py --list_methods

# List all supported datasets (synapse, acdc, polyp), resolutions, class counts, and default paths
python main.py --list_datasets
```

### 2. CLI Parameter Groups Reference

| Group | Key CLI Arguments | Description |
| :--- | :--- | :--- |
| **`[1]` Mode & Discovery** | `--mode {train,reproduce,eval,tune,package}`, `--eval_only`, `--dry_run`, `--list_methods`, `--list_datasets` | Select execution mode or inspect available configurations without running GPU jobs. |
| **`[2]` Method & Dataset** | `--method {ours,iad,dist_plus,lskd,sinkd,kis,v2,v3,v4,sgs_only,all}`, `--arm`, `--dataset {synapse,btcv,acdc,polyp,all}`, `--img_size`, `--num_classes`, `--data_dir`, `--list_path`, `--test_dir`, `--test_list` | Choose method(s) and dataset(s). Automatically configures `num_classes` (`9` Synapse, `4` ACDC, `1` Polyp binary channel), `img_size` (`512` Synapse, `256` ACDC/Polyp), and teacher checkpoints. |
| **`[3]` Reproducibility** | `--seed 2345`, `--seeds 2345 42 1234` (or `--seeds all`) | Deterministic seeding (`PYTHONHASHSEED`, `random`, `numpy`, `torch`, `cudnn.deterministic=True`, `DataLoader` worker seeds). |
| **`[4]` Schedule & Optim** | `--epochs`, `--batch_size`, `--grad_accum_steps`, `--lr`, `--lr_schedule`, `--sched_epochs`, `--warmup_iters`, `--weight_decay`, `--aug`, `--ema_buffer_mode`, `--no_amp`, `--cpu`, `--smoke_test` | Training epochs (`300`), microbatch size (`8`, `grad_accum_steps=2` -> effective batch `16`), AdamW LR (`1e-4`), AMP mixed precision (default enabled) or FP32 (`--no_amp`). |
| **`[5]` SGS & KIS** | `--sgs_weight`, `--kis_weight`, `--cpd_weight_mode {paper,flat}` | Base supervised loss ($\text{CE} + \text{Dice}$ or $\text{wbce} + \text{wiou}$ on Polyp) and Stage-1 KIS ($\text{KL} + \text{CPD}$). |
| **`[6]` Ours (`CBA v2`)** | `--lambda_a`, `--beta`, `--lambda_ratio_cap`, `--m`, `--tau`, `--r`, `--k_bg`, `--k_fg`, `--cache_momentum`, `--gating_type`, `--error_gating_type`, `--cba_feature`, `--gate_ref`, `--use_curriculum`, `--make_warmup` | Core-to-Boundary Assimilation ($\mathcal{L}_B$: $\lambda_A=0.02, r=2, \tau=0.1, m=0.2$) and Class-Balanced Hard-Error CE ($\mathcal{L}_E$: $\beta=0.5$). |
| **`[7]` Baseline KD** | `--kis_stage_epochs`, `--kas_weight`, `--kas_temp`, `--dist_beta`, `--dist_gamma`, `--dist_tau`, `--lskd_tau`, `--sinkd_lam`, `--sinkd_iters` | Hyperparameters for `IAD` (`200 KIS + 100 KAS`), `DIST+`, `LSKD`, and `SinKD`. |
| **`[8]` Far-FP & Experimental** | `--lambda_far`, `--far_threshold`, `--far_mode`, `--fp_by_pred_class`, `--fp_exclude_boundary`, `--empty_slice_suppression` | Optional Far-FP suppression and experimental error-set filters. |
| **`[9]` Checkpoints & Eval** | `--teacher_ckpt`, `--teacher_base_ckpt`, `--student_ckpt`, `--ckpt`, `--eval_role {student,teacher}`, `--eval_output`, `--resume`, `--output_dir`, `--drive_dir`, `--eval_freq`, `--save_final_preds` | Checkpoint paths, Google Drive sync, and full evaluation output JSON. |
| **`[10]` Tuning & Package** | `--tune_action {plan,run,score}`, `--tune_stage {warmup,ref,A,B,C}`, `--bundle`, `--regen_notebooks` | Multi-stage hyperparameter tuning and self-contained Colab bundle generation. |

### 3. Train Any Method on Any Dataset (`--mode train`)

```bash
# Train Ours (v5) on Synapse (300 epochs, seed 2345)
python main.py --mode train --method ours --dataset synapse --epochs 300 --seed 2345 \
    --lambda_a 0.02 --beta 0.5 --tau 0.1 --m 0.2 --r 2 --output_dir runs/synapse_ours

# Train 2-Stage IAD (200 KIS + 100 KAS) on ACDC
python main.py --mode train --method iad --dataset acdc --epochs 300 --kis_stage_epochs 200 --seed 2345 \
    --output_dir runs/acdc_iad

# Train DIST+, LSKD, SinKD, or SGS-Only on Polyp across 3 seeds sequentially
python main.py --mode train --method dist_plus --dataset polyp --epochs 300 --seeds 2345 42 1234 \
    --output_dir runs/polyp_dist_plus
```

### 4. Reproduce Official 300-Epoch Multi-Seed Benchmarks (`--mode reproduce`)

Uses `scripts/main_common.py` to manage the shared 20-epoch KIS warmup, automatic checkpoint resume, completion skipping, and Google Drive synchronization:

```bash
# Reproduce Ours (v5) across all 3 seeds (2345, 42, 1234) on Synapse
python main.py --mode reproduce --method ours --dataset synapse --seeds 2345 42 1234 \
    --runs_dir runs/main --drive_dir /content/drive/MyDrive/IAD/main --make_warmup

# Reproduce all 5 main methods (ours, iad, dist_plus, lskd, sinkd) on ACDC or Polyp
python main.py --mode reproduce --method all --dataset acdc --seed 2345
python main.py --mode reproduce --method all --dataset polyp --seed 2345

# Preview all commands across all 5 methods × 3 datasets × 3 seeds without executing
python main.py --mode reproduce --method all --dataset all --seeds all --dry_run
```

### 5. Standalone Checkpoint Evaluation (`--mode eval` / `--eval_only`)

```bash
# Evaluate a trained Student checkpoint on Synapse (12 3D test volumes: Raw/LCC Dice, HD95, NSD, ASD)
python main.py --mode eval --dataset synapse --eval_role student \
    --ckpt teacher_weights/synapse_stu_iad.pth

# Evaluate a trained Student or Teacher checkpoint on ACDC (40 3D test volumes) or Polyp (5 test sets)
python main.py --mode eval --dataset acdc --eval_role student --ckpt teacher_weights/acdc_stu_iad.pth
python main.py --mode eval --dataset polyp --eval_role teacher --ckpt teacher_weights/polyp_tea.pth
```

### 6. Package Self-Contained Colab Bundle (`--mode package`)

```bash
python main.py --mode package --bundle cba_v2 --regen_notebooks
```

---

## ✅ Automated Test Suite

The repository includes an automated `pytest` suite (`242` tests across `13` test modules in [`tests/`](./tests)) covering:
- Top-level `main.py` CLI parsing, method/dataset auto-discovery, dry-run execution, evaluation, and packaging ([`tests/test_main_cli.py`](./tests/test_main_cli.py))
- ACDC & Polyp 5-method loss computation, data loaders, checkpoint loading, and 3-notebook generation ([`tests/test_acdc_polyp_5methods.py`](./tests/test_acdc_polyp_5methods.py))
- Synapse 5-method × 3-seed protocol, warmup handoff, and loss parity ([`tests/test_cba_v2_5methods_multiseed.py`](./tests/test_cba_v2_5methods_multiseed.py))
- `CBAv2Loss`, `KISLoss`, `BaselineKDLoss`, and `FarFPLoss` unit tests ([`tests/test_cba_v2_loss.py`](./tests/test_cba_v2_loss.py))

```bash
# Run all 242 tests
python -m pytest tests -q

# Run focused CLI & multi-dataset benchmark tests
python -m pytest tests/test_main_cli.py tests/test_acdc_polyp_5methods.py tests/test_cba_v2_5methods_multiseed.py -q
```

---

## 🗂️ Dataset & Pre-trained Checkpoint Reference

| Dataset | Classes | Resolution | Train Samples | Test Evaluation | Teacher Checkpoint (`teacher_weights/`) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Synapse (BTCV)** | `9` (BG + 8 organs) | `512 × 512` | `2,211` 2D slices (`18` cases) | `12` 3D CT volumes (Raw & LCC DSC, NSD, HD95, ASD) | `epoch_299.pth` (`82.78%` DSC) |
| **ACDC** | `4` (BG, RV, Myo, LV) | `256 × 256` | `1,304` 2D slices (`70` cases) | `40` 3D MRI volumes (`20` patients × ED/ES) | `acdc_tea.pth` (`91.42%` DSC) |
| **Polyp** | `1` (binary channel) | `256 × 256` | `1,450` RGB images | `5` datasets (`798` imgs: `Kvasir`, `CVC-ClinicDB`, `CVC-ColonDB`, `ETIS`, `CVC-300`) | `polyp_tea.pth` (`80.34%` 4-Set / `81.49%` 5-Set DSC) |

---

## 📝 Citation

```bibtex
@inproceedings{hong2026infusion,
  title={From Infusion to Assimilation Distillation for Medical Image Segmentation},
  author={Hong, Jiankang and Luo, Ye and Liu, Yinan and Yuan, Junsong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20985--20995},
  year={2026}
}
```
