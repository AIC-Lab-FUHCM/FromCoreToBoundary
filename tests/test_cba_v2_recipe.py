"""Phase 3 tests: KIS-300 baseline recipe defaults, EMA buffer handling, original augmentation port."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import dataset_synapse as ds
from experiments.cba_v2.data import full_augment
from experiments.cba_v2.train import build_parser, ema_update, recipe_summary


def test_recipe_defaults_match_plan():
    a = build_parser().parse_args([])
    assert a.weight_decay == 0.1
    assert a.warmup_iters == 250
    assert a.aug == "full"
    assert a.cpd_weight_mode == "flat"
    assert a.ema_buffer_mode == "average"
    assert a.student_ckpt is None
    assert a.teacher_ckpt == "teacher_weights/epoch_299.pth"
    assert (a.batch_size, a.grad_accum_steps, a.epochs) == (8, 2, 300)
    assert (a.curriculum_warmup, a.curriculum_ramp, a.lambda_a) == (20, 30, 0.02)


def test_recipe_summary_records_known_differences():
    a = build_parser().parse_args([])
    r = recipe_summary(a, use_amp=True, is_bf16=True)
    assert r["used"]["precision"] == "bf16 autocast" and r["used"]["weight_decay"] == 0.1
    joined = " ".join(r["known_differences_from_kis300_run"])
    assert "fp32" in joined and "microbatches of 8" in joined and "best_probe" in joined
    assert recipe_summary(a, use_amp=False, is_bf16=False)["used"]["precision"] == "fp32"


# ----------------------------------------------------------------------------- EMA

def _bn_net():
    return nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))


def test_ema_average_mode_averages_float_buffers_and_copies_int_buffers():
    torch.manual_seed(0)
    student, ema = _bn_net(), _bn_net()
    ema.load_state_dict(student.state_dict())
    student.train()
    student(torch.randn(8, 3, 4, 4) * 3 + 2)  # moves running stats, bumps num_batches_tracked
    old_mean = ema[1].running_mean.clone()
    old_p = ema[0].weight.clone()
    ema_update(student, ema, decay=0.99, buffer_mode="average")
    assert torch.allclose(ema[1].running_mean, 0.99 * old_mean + 0.01 * student[1].running_mean)
    assert torch.allclose(ema[1].running_var, 0.99 * torch.ones(4) + 0.01 * student[1].running_var)
    assert torch.allclose(ema[0].weight, 0.99 * old_p + 0.01 * student[0].weight)
    assert ema[1].num_batches_tracked.item() == student[1].num_batches_tracked.item() == 1
    assert ema[1].num_batches_tracked.dtype == torch.long


def test_ema_copy_mode_copies_all_buffers():
    torch.manual_seed(0)
    student, ema = _bn_net(), _bn_net()
    ema.load_state_dict(student.state_dict())
    student.train()
    student(torch.randn(8, 3, 4, 4) * 3 + 2)
    ema_update(student, ema, decay=0.99, buffer_mode="copy")
    assert torch.equal(ema[1].running_mean, student[1].running_mean)
    assert torch.equal(ema[1].running_var, student[1].running_var)


# ----------------------------------------------------------------------------- augmentation

def test_full_augment_selection_probabilities_match_random_generator(monkeypatch):
    """Branch frequencies of the ported chain must equal the original RandomGenerator's:
    rot_flip 0.5, rotate 0.5*0.45, scale 0.5*0.55*0.75, elastic 0.5*0.55*0.25*0.85, identity rest."""
    counts = {"rot_flip": 0, "rotate": 0, "scale": 0, "elastic": 0}

    def make(name):
        def fake(image, label, binary_label, *a, **k):
            counts[name] += 1
            return image, label, binary_label
        return fake

    for name, fn in (("rot_flip", "random_rot_flip"), ("rotate", "random_rotate"),
                     ("scale", "random_scale"), ("elastic", "random_elastic")):
        monkeypatch.setattr(ds, fn, make(name))

    random.seed(0)
    n = 6000
    img = np.zeros((8, 8), dtype=np.float32)
    lab = np.zeros((8, 8), dtype=np.int64)
    for _ in range(n):
        full_augment(img, lab, output_size=(8, 8))
    expected = {
        "rot_flip": 0.5,
        "rotate": 0.5 * 0.45,
        "scale": 0.5 * 0.55 * 0.75,
        "elastic": 0.5 * 0.55 * 0.25 * 0.85,
    }
    for name, p in expected.items():
        assert counts[name] / n == pytest.approx(p, abs=0.02), (name, counts[name] / n, p)


def test_full_augment_real_ops_keep_shapes_dtypes_and_label_range():
    random.seed(123)
    np.random.seed(123)
    h = w = 512
    image = np.random.rand(h, w).astype(np.float32)
    label = np.zeros((h, w), dtype=np.int64)
    label[100:220, 120:260] = 5
    label[300:360, 300:380] = 1
    changed = 0
    for _ in range(60):
        out_img, out_lab = full_augment(image, label)
        assert out_img.shape == (512, 512) and out_img.dtype == np.float32
        assert out_lab.shape == (512, 512) and out_lab.dtype == np.int64
        assert out_lab.min() >= 0 and out_lab.max() <= 8
        assert set(np.unique(out_lab)) <= {0, 1, 5}, "augmentation must not invent class ids"
        changed += int(not np.array_equal(out_lab, label))
    assert changed > 30, "most draws should alter the sample (identity branch is ~1%)"
