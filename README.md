# FromCoreToBoundary: Core-to-Boundary Assimilation & Hard-Error Distillation for Medical Image Segmentation

Official implementation of **FromCoreToBoundary (`Ours`)** and Knowledge Distillation benchmarks (`IAD`, `DIST+`, `LSKD`, `SinKD`) across **Synapse (BTCV)**, **ACDC**, and **Polyp**.

---

## 🛠️ Environment Setup

```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```text
FromCoreToBoundary/
├── main.py                        # Top-level CLI entry point (train & eval)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── LICENSE                        # Apache-2.0 License
├── data/                          # Official dataset split lists (Synapse / BTCV & ACDC)
├── datasets/                      # Dataset loaders & resolvers (data.py)
├── losses/                        # Loss formulations
│   ├── ours_loss.py               # Proposed Ours (Directionally Gated Boundary Stream L_B + Hard-Error CE Stream L_E)
│   ├── kis_loss.py                # Stage-1 Knowledge Infusion
│   └── kd_baselines.py            # Baseline KD criteria (IAD, DIST+, LSKD, SinKD)
├── models/                        # Teacher & Student architectures
│   ├── builder.py                 # Teacher (SAM ViT-L LoRA) & Student (EfficientNet-B1 + SAM) builders
│   ├── segment_anything_Tea/      # Teacher SAM ViT-L architecture
│   ├── segment_anything_Stu/      # Student EfficientNet-B1 + SAM Mask Decoder architecture
│   ├── sam_lora_image_encoder_tea.py  # Teacher LoRA adapter
│   └── sam_lora_image_encoder_stu.py  # Student adapter wrapper
└── scripts/                       # Core training & evaluation engine
    ├── train.py                   # Training loop
    └── evaluation.py              # 3D volume (Synapse / ACDC) & 5-dataset (Polyp) evaluation
```

---

## 🚀 Usage (`main.py`)

### 1. Training (`--mode train`)

```bash
# Train Ours on Synapse (BTCV), ACDC, or Polyp
python main.py --mode train --method ours --dataset synapse --epochs 300 --seed 2345 --output_dir runs/synapse_ours
python main.py --mode train --method ours --dataset acdc --epochs 300 --seed 2345 --output_dir runs/acdc_ours
python main.py --mode train --method ours --dataset polyp --epochs 300 --seed 2345 --output_dir runs/polyp_ours

# Train across multiple seeds sequentially
python main.py --mode train --method ours --dataset synapse --epochs 300 --seeds 2345 42 1234 --output_dir runs/synapse_ours

# Train baseline KD methods (iad, dist_plus, lskd, sinkd)
python main.py --mode train --method iad --dataset synapse --epochs 300 --seed 2345 --output_dir runs/synapse_iad
```

### 2. Evaluation (`--mode eval` / `--eval_only`)

```bash
# Evaluate a trained Student checkpoint on Synapse (12 3D test volumes), ACDC (40 3D test volumes), or Polyp (5 test sets)
python main.py --mode eval --dataset synapse --ckpt runs/synapse_ours/ours/final_student.pth
python main.py --mode eval --dataset acdc --ckpt teacher_weights/acdc_stu_iad.pth
python main.py --mode eval --dataset polyp --ckpt teacher_weights/polyp_stu_iad.pth

# Evaluate a pre-trained Teacher checkpoint
python main.py --mode eval --dataset acdc --eval_role teacher --ckpt teacher_weights/acdc_tea.pth
```

### 3. CLI Arguments

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--mode` | `train` | Execution mode: `train` or `eval` (or use `--eval_only`). |
| `--method` | `ours` | KD method: `ours`, `iad`, `dist_plus`, `lskd`, `sinkd`. |
| `--dataset` | `synapse` | Dataset: `synapse` (`btcv`), `acdc`, or `polyp`. |
| `--epochs` | `300` | Total training epochs. |
| `--batch_size` | `8` | Microbatch size (`grad_accum_steps=2` -> effective batch size `16`). |
| `--lr` | `1e-4` | Base learning rate. |
| `--seed` / `--seeds` | `2345` | Single seed (`--seed 2345`) or multi-seed list (`--seeds 2345 42 1234`). |
| `--ckpt` | `None` | Checkpoint path to evaluate in `--mode eval`. |
| `--eval_role` | `student` | Model role to evaluate in `--mode eval`: `student` or `teacher`. |
| `--teacher_ckpt` | *(auto)* | Teacher checkpoint path (auto-resolved per dataset). |
| `--student_ckpt` | `None` | Optional initial student checkpoint path. |
| `--resume` | `None` | Checkpoint path to resume training from, or `auto`. |
| `--output_dir` | `runs/ours` | Output directory for checkpoints and logs. |

---

## 🗂️ Datasets & Pre-trained Checkpoints

| Dataset | Classes | Resolution | Train Split | Test Split | Default Teacher Checkpoint |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Synapse (BTCV)** | `9` (BG + 8 organs) | `512 × 512` | `2,211` 2D slices (`18` cases) | `12` 3D CT volumes | `teacher_weights/epoch_299.pth` |
| **ACDC** | `4` (BG, RV, Myo, LV) | `256 × 256` | `1,304` 2D slices (`70` cases) | `40` 3D MRI volumes (`20` patients × ED/ES) | `teacher_weights/acdc_tea.pth` |
| **Polyp** | `1` (binary channel) | `256 × 256` | `1,450` RGB images | `5` benchmark sets (`798` images) | `teacher_weights/polyp_tea.pth` |
