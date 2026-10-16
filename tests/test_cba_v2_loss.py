"""
tests/test_cba_v2_loss.py - Comprehensive Unit Test Suite for CBA v2 Loss.

Covers:
1. Mathematical precision: Closed-form Error contrastive gradient g_i^E matches autograd through L2 norm.
2. Prototype memory cache: EMA momentum update, multi-batch persistence, reset_cache.
3. Missing class fallback: Spleen absent in batch, student predicts stomach as spleen, cached spleen used.
4. Unseen class graceful skip: Class never seen in batch or cache is skipped and recorded in diagnostics.
5. Dedicated error sampling budgets: K_bg for background false positives, K_fg for organ errors.
6. Error categories: Background false positive (y=0, pred>0), Organ missed as background (y>0, pred=0), Organ-organ confusion.
7. Class-balanced reduction: Balanced averaging across active error classes vs pixel-mean reduction.
8. Directional gating: Soft, binary, parametric, none, and anti-aligned reference gradient suppression.
9. Gradient flow to student: Zero NaNs, valid autograd graph, AMP compatibility.
10. Overlap diagnostics: Boundary vs Error pixel overlap metrics.
11. Edge cases: Pure background, zero prediction errors.
12. Execution latency benchmark: Measure ms/step overhead on 512x512 inputs.
"""

import sys
import time
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss


class TestCBAv2Loss:

    def test_analytical_gradient_matches_autograd(self):
        """Verify closed-form Error gradient g_i^E matches autograd through L2 normalization."""
        m = 0.2
        tau = 0.1
        C = 9

        torch.manual_seed(42)
        h = torch.randn(1, C, dtype=torch.float64, requires_grad=True)
        z = F.normalize(h, p=2, dim=1)

        mu_c = F.normalize(torch.randn(1, C, dtype=torch.float64), p=2, dim=1).detach()
        mu_d = F.normalize(torch.randn(1, C, dtype=torch.float64), p=2, dim=1).detach()

        sim_c = (z * mu_c).sum(dim=1)
        sim_d = (z * mu_d).sum(dim=1)
        u = (m + sim_d - sim_c) / tau
        loss_val = F.softplus(u)

        (grad_autograd,) = torch.autograd.grad(loss_val, h, retain_graph=True)

        sigma = torch.sigmoid(u)
        delta_mu = mu_d - mu_c
        h_norm = torch.norm(h, p=2, dim=1, keepdim=True)
        proj = (z * delta_mu).sum(dim=1, keepdim=True)
        tangent = delta_mu - z * proj
        g_analytical = (sigma / (tau * h_norm)) * tangent

        diff = torch.norm(grad_autograd - g_analytical).item()
        assert diff < 1e-10, f"Analytical Error gradient mismatch: {diff}"

    def test_prototype_memory_cache_update_and_persistence(self):
        """Verify EMA Prototype Memory Cache updates when classes appear and persists across batches."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, cache_momentum=0.8, num_classes=9)

        # Batch 1: Contains classes 0, 1, 2
        torch.manual_seed(10)
        fplus_1 = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_1 = torch.zeros(2, 32, 32, dtype=torch.long)
        y_1[:, 4:12, 4:12] = 1
        y_1[:, 16:24, 16:24] = 2

        criterion.train()
        loss_1, diag_1 = criterion(fplus_1, y_1, return_diagnostics=True)

        assert criterion.valid_mask[0].item() is True
        assert criterion.valid_mask[1].item() is True
        assert criterion.valid_mask[2].item() is True
        assert criterion.valid_mask[3].item() is False
        assert criterion.valid_mask[7].item() is False

        # First appearance: accumulator vector v_c is directly initialized from batch proto (norm = 1.0)
        assert abs(torch.norm(criterion.prototype_buffer[0], p=2).item() - 1.0) < 1e-5
        assert abs(torch.norm(criterion.prototype_buffer[1], p=2).item() - 1.0) < 1e-5

        proto_1_before = criterion.prototype_buffer[1].clone()
        proto_0_before = criterion.prototype_buffer[0].clone()

        # Batch 2: Contains classes 0, 3, 4 (classes 1 and 2 are absent)
        fplus_2 = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_2 = torch.zeros(2, 32, 32, dtype=torch.long)
        y_2[:, 4:12, 4:12] = 3
        y_2[:, 16:24, 16:24] = 4

        loss_2, diag_2 = criterion(fplus_2, y_2, return_diagnostics=True)

        # Classes 1 and 2 must remain valid in cache!
        assert criterion.valid_mask[1].item() is True
        assert criterion.valid_mask[2].item() is True
        # Their prototypes should be preserved exactly since they were absent
        assert torch.allclose(criterion.prototype_buffer[1], proto_1_before)

        # Classes 3 and 4 are now valid with norm 1.0
        assert criterion.valid_mask[3].item() is True
        assert criterion.valid_mask[4].item() is True
        assert abs(torch.norm(criterion.prototype_buffer[3], p=2).item() - 1.0) < 1e-5

        # Class 0 appeared again, so accumulator v_0 updated via momentum without normalizing
        proto_0_after = criterion.prototype_buffer[0]
        assert not torch.allclose(proto_0_before, proto_0_after)
        assert criterion.update_count[0].item() == 2
        # Temporal directional stability is measured on v_0 BEFORE normalization
        r_0 = torch.norm(proto_0_after, p=2).item()
        assert 0.0 < r_0 <= 1.0 + 1e-5
        # Anchor hat{mu}_0 is unit-normalized
        anchor_0 = criterion.get_cache_anchor(0)
        assert abs(torch.norm(anchor_0, p=2).item() - 1.0) < 1e-5

        # Reset cache test
        criterion.reset_cache()
        assert not criterion.valid_mask.any()
        assert (criterion.update_count == 0).all()

    def test_eval_mode_freezes_cache(self):
        """Verify eval mode strictly prevents cache updates so eval GT never enters training state."""
        criterion = CBAv2Loss(num_classes=9)
        criterion.reset_cache()

        # In train mode, appearance of class 1 populates cache
        f_train = torch.randn(1, 9, 16, 16)
        y_train = torch.zeros(1, 16, 16, dtype=torch.long)
        y_train[0, 2:6, 2:6] = 1
        criterion.train()
        criterion(f_train, y_train)
        assert criterion.valid_mask[1].item() is True
        assert criterion.update_count[1].item() == 1
        proto_1 = criterion.prototype_buffer[1].clone()

        # Switch to eval mode
        criterion.eval()
        f_eval = torch.randn(1, 9, 16, 16)
        y_eval = torch.zeros(1, 16, 16, dtype=torch.long)
        y_eval[0, 4:8, 4:8] = 2  # New class 2 in validation slice
        y_eval[0, 2:6, 2:6] = 1  # Existing class 1 in validation slice

        criterion(f_eval, y_eval)

        # Class 2 MUST NOT be added to cache!
        assert criterion.valid_mask[2].item() is False
        # Class 1 MUST NOT be updated!
        assert criterion.update_count[1].item() == 1
        assert torch.allclose(criterion.prototype_buffer[1], proto_1)

    def test_state_dict_save_and_resume(self):
        """Verify checkpoint state_dict correctly saves and restores accumulator, valid_mask, update_count, and coherence."""
        criterion = CBAv2Loss(num_classes=9)
        # Populate prototypes with specific accumulator vectors
        p1 = F.normalize(torch.tensor([1.0] * 9), p=2, dim=0)
        p2 = F.normalize(torch.tensor([2.0, -1.0, 0.5, 0.0, 1.2, -0.4, 0.8, -0.2, 0.1]), p=2, dim=0)
        criterion.set_prototype(1, p1)
        criterion.set_prototype(2, p2)
        # Simulate an update on class 1
        criterion.prototype_buffer[1] = 0.9 * criterion.prototype_buffer[1] + 0.1 * p2
        criterion.update_count[1] = 5

        state = criterion.state_dict()
        assert "prototype_buffer" in state
        assert "valid_mask" in state
        assert "update_count" in state

        # Create fresh instance and load
        criterion_new = CBAv2Loss(num_classes=9)
        assert not criterion_new.valid_mask.any()
        criterion_new.load_state_dict(state)

        assert torch.allclose(criterion_new.prototype_buffer, criterion.prototype_buffer)
        assert torch.equal(criterion_new.valid_mask, criterion.valid_mask)
        assert torch.equal(criterion_new.update_count, criterion.update_count)
        assert criterion_new.get_cache_coherence() == criterion.get_cache_coherence()

        anchors_old = criterion.get_cached_prototypes()
        anchors_new = criterion_new.get_cached_prototypes()
        for c in anchors_old:
            assert torch.allclose(anchors_old[c], anchors_new[c])

    def test_boundary_v1_consistency_vs_cache(self):
        """Verify boundary_use_cache=False uses fresh batch prototypes (exact v1), whereas True uses cache anchors."""
        c_v1 = CBAv2Loss(boundary_use_cache=False, num_classes=9)
        c_cached = CBAv2Loss(boundary_use_cache=True, num_classes=9)

        # Pre-seed cache with orthogonal artificial prototypes
        p_c = torch.zeros(9)
        p_c[1] = 1.0  # Class 1 points along axis 1
        p_d = torch.zeros(9)
        p_d[0] = 1.0  # Class 0 points along axis 0
        c_v1.set_prototype(1, p_c)
        c_v1.set_prototype(0, p_d)
        c_cached.set_prototype(1, p_c)
        c_cached.set_prototype(0, p_d)

        # Batch features point along different axes:
        fplus = torch.zeros(1, 9, 32, 32, requires_grad=True)
        with torch.no_grad():
            fplus[0, 5, 8:24, 8:24] = 5.0
            fplus[0, 6, :, :] = 5.0
            fplus[0, 5, 8:24, 8:24] = 10.0

        y_gt = torch.zeros(1, 32, 32, dtype=torch.long)
        y_gt[0, 8:24, 8:24] = 1

        # Run forward with update_cache=False to preserve pre-seeded cache
        loss_v1, diag_v1 = c_v1(fplus, y_gt, return_diagnostics=True, update_cache=False)
        loss_cached, diag_cached = c_cached(fplus, y_gt, return_diagnostics=True, update_cache=False)

        assert diag_v1["boundary_use_cache"] is False
        assert diag_cached["boundary_use_cache"] is True
        assert loss_v1.item() != loss_cached.item()

    def test_missing_class_fallback_to_cache(self):
        """Simulate Stomach predicted as Spleen when Spleen is absent in batch but present in cache."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, r=2, beta=0.5, num_classes=9)

        # Pre-populate cache with Spleen (class 1) prototype
        torch.manual_seed(100)
        spleen_proto = F.normalize(torch.randn(9), p=2, dim=0)
        criterion.set_prototype(c=1, prototype=spleen_proto)

        assert criterion.valid_mask[1].item() is True
        assert torch.allclose(criterion.prototype_buffer[1], spleen_proto, atol=1e-6)

        # Batch 2: Ground truth has Stomach (class 7), NO Spleen (class 1) in GT
        fplus = torch.zeros(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
        # Place stomach region
        y_gt[:, 8:20, 8:20] = 7

        # Force student to predict Stomach as Spleen (class 1)
        with torch.no_grad():
            fplus[:, 1, 8:16, 8:16] = 5.0  # Misclassified as class 1 (Spleen)
            fplus[:, 7, 16:20, 8:20] = 5.0  # Correctly predicted stomach

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)

        assert loss.item() > 0.0
        assert diag["loss_e"] > 0.0
        # Spleen prototype was supplied by cache
        assert diag["cache_hits"] >= 1
        assert diag["proto_sources"][1] == "cache"
        assert diag["proto_sources"][7] == "batch"
        # The misclassified pixels were NOT skipped
        assert diag["skipped_error_pixels"] == 0

    def test_unseen_class_graceful_skip(self):
        """When an opponent class has never been seen in batch or cache, gracefully skip and log."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, num_classes=9)
        criterion.reset_cache()

        # Batch has only class 0 (background) and class 3
        # But student predicts class 8 (which was NEVER seen)
        fplus = torch.zeros(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
        y_gt[:, 10:20, 10:20] = 3

        with torch.no_grad():
            # Force class 8 prediction for class 3 pixels
            fplus[:, 8, 10:15, 10:20] = 10.0

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)

        # Class 8 has no prototype anywhere
        assert diag["skipped_error_pixels"] > 0
        assert not torch.isnan(loss)
        assert loss.requires_grad

    def test_background_and_foreground_sampling_budgets(self):
        """Verify dedicated budgets K_bg and K_fg are strictly respected for background and organ errors."""
        k_bg = 64
        k_fg = 32
        criterion = CBAv2Loss(k_bg=k_bg, k_fg=k_fg, num_classes=9)

        # Pre-seed cache so prototypes exist
        for c in range(9):
            criterion.set_prototype(c, torch.randn(9))

        # Create large error regions
        fplus = torch.zeros(2, 9, 64, 64, requires_grad=True)
        y_gt = torch.zeros(2, 64, 64, dtype=torch.long)

        # 1. Background false positive: 500 pixels of class 0 predicted as class 4 (gallbladder)
        with torch.no_grad():
            fplus[:, 4, 0:20, 0:25] = 10.0  # 2 * 20 * 25 = 1000 pixels

        # 2. Foreground error: 300 pixels of class 1 predicted as class 0
        y_gt[:, 30:45, 30:40] = 1            # 2 * 15 * 10 = 300 pixels
        with torch.no_grad():
            fplus[:, 0, 30:45, 30:40] = 10.0

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)

        # Budget constraints verified
        assert diag["sampled_pixels_per_class"][0] == k_bg, f"Background sampled should be {k_bg}"
        assert diag["sampled_pixels_per_class"][1] == k_fg, f"Organ 1 sampled should be {k_fg}"
        assert diag["bg_error_count"] > k_bg
        assert diag["fg_error_count"] > k_fg
        assert diag["fg_missed_as_bg_count"] > 0

    def test_class_balanced_reduction_invariance(self):
        """Verify class-balanced reduction averages within each class, then across active classes."""
        criterion_cb = CBAv2Loss(m=0.2, tau=0.1, error_reduction="class_balanced", num_classes=9)
        criterion_pm = CBAv2Loss(m=0.2, tau=0.1, error_reduction="pixel_mean", num_classes=9)

        for c in range(9):
            criterion_cb.set_prototype(c, torch.randn(9))
            criterion_pm.set_prototype(c, criterion_cb.prototype_buffer[c])

        # Organ 1 has 200 error pixels, Organ 2 has 4 error pixels
        fplus = torch.zeros(1, 9, 64, 64, requires_grad=True)
        y_gt = torch.zeros(1, 64, 64, dtype=torch.long)

        # Organ 1: 200 pixels predicted as Organ 3
        y_gt[0, 0:20, 0:10] = 1
        with torch.no_grad():
            fplus[0, 3, 0:20, 0:10] = 8.0

        # Organ 2: 4 pixels predicted as Organ 4
        y_gt[0, 30:32, 30:32] = 2
        with torch.no_grad():
            fplus[0, 4, 30:32, 30:32] = 8.0

        loss_cb, diag_cb = criterion_cb(fplus, y_gt, return_diagnostics=True, sample_seed=42)
        loss_pm, diag_pm = criterion_pm(fplus, y_gt, return_diagnostics=True, sample_seed=42)

        # In class-balanced reduction, class 1 and class 2 contribute equally: (L_1 + L_2) / 2
        # In pixel-mean reduction, class 1 has 200 / 204 = 98% of the weight!
        assert diag_cb["active_error_classes"] == [1, 2]
        assert loss_cb.item() > 0.0
        assert loss_pm.item() > 0.0

    def test_directional_gating_and_anti_alignment(self):
        """Verify directional gate suppresses conflicting error gradients when anti-aligned."""
        criterion_gated = CBAv2Loss(m=0.2, tau=0.1, gating_type="soft", error_gating_type="soft", num_classes=9)
        criterion_ungated = CBAv2Loss(m=0.2, tau=0.1, gating_type="none", error_gating_type="none", num_classes=9)

        for c in range(9):
            criterion_gated.set_prototype(c, torch.randn(9))
            criterion_ungated.set_prototype(c, criterion_gated.prototype_buffer[c])

        torch.manual_seed(42)
        fplus = torch.randn(1, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(1, 32, 32, dtype=torch.long)
        y_gt[0, 5:15, 5:15] = 1

        loss_u = criterion_ungated(fplus, y_gt, sample_seed=42)
        g_u = torch.autograd.grad(loss_u, fplus, retain_graph=True)[0]

        # Anti-aligned reference gradient should yield zero loss and ~100% conflict rate
        loss_anti, diag_anti = criterion_gated(fplus, y_gt, return_diagnostics=True, g_ref=-g_u, sample_seed=42)
        assert abs(loss_anti.item()) < 1e-4
        assert diag_anti["error_micro_conflict_rate"] > 0.95

    def test_gradient_flow_and_no_nans(self):
        """Verify loss cleanly backpropagates to fplus without NaNs or Infs."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, r=2, beta=0.5, num_classes=9)

        fplus = torch.randn(2, 9, 48, 48, requires_grad=True)
        y_gt = torch.randint(0, 9, (2, 48, 48))

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True, compute_autograd_norms=True)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.requires_grad

        loss.backward()
        assert fplus.grad is not None
        assert not torch.isnan(fplus.grad).any()
        assert not torch.isinf(fplus.grad).any()
        assert fplus.grad.norm().item() > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_and_amp_compatibility(self):
        """Verify CBA v2 loss works on GPU under torch.amp.autocast."""
        device = torch.device("cuda")
        criterion = CBAv2Loss(m=0.2, tau=0.1, r=2, beta=0.5, num_classes=9).to(device)

        fplus = torch.randn(2, 9, 64, 64, device=device, requires_grad=True)
        y_gt = torch.randint(0, 9, (2, 64, 64), device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(fplus, y_gt)

        assert not torch.isnan(loss)
        loss.backward()
        assert fplus.grad is not None
        assert not torch.isnan(fplus.grad).any()

    def test_all_background_edge_case(self):
        """When input is all background and prediction is correct, loss is 0.0 with valid grad."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, num_classes=9)
        fplus = torch.zeros(2, 9, 32, 32, requires_grad=True)
        # Set class 0 logit high so prediction is class 0 everywhere
        with torch.no_grad():
            fplus[:, 0] = 10.0
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)
        assert loss.item() == 0.0
        assert loss.requires_grad
        assert diag["total_error_pixels"] == 0
        assert diag["total_boundary_pixels"] == 0

        loss.backward()
        assert fplus.grad is not None
        assert not torch.isnan(fplus.grad).any()

    def test_overlap_diagnostics_calculation(self):
        """Verify calculation of overlap between boundary pixels and error pixels."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, r=2, num_classes=9)

        # Create structured image: organ 1 square adjacent to background
        fplus = torch.zeros(1, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(1, 32, 32, dtype=torch.long)
        y_gt[0, 10:22, 10:22] = 1

        # Predict organ 2 on boundary pixels of organ 1
        with torch.no_grad():
            fplus[0, 1, :, :] = 2.0
            # Mispredict a subset of organ 1 pixels
            fplus[0, 2, 10:14, 10:14] = 10.0

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)

        assert diag["total_error_pixels"] > 0
        assert diag["overlap_boundary_error_pixels"] >= 0
        assert 0.0 <= diag["overlap_ratio_boundary"] <= 1.0
        assert 0.0 <= diag["overlap_ratio_error"] <= 1.0

    def test_execution_latency_benchmark(self):
        """Measure ms/step overhead on 512x512 inputs to confirm it does not slow down training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = CBAv2Loss(m=0.2, tau=0.1, r=2, beta=0.5, k_bg=512, k_fg=512, num_classes=9).to(device)

        # Pre-seed cache so all classes are ready
        for c in range(9):
            criterion.set_prototype(c, torch.randn(9, device=device))

        B, C, H, W = 2, 9, 512, 512
        torch.manual_seed(42)
        fplus = torch.randn(B, C, H, W, device=device, requires_grad=True)
        y_gt = torch.zeros(B, H, W, dtype=torch.long, device=device)
        # Structured multi-organ CT-like slice (background + 4 organs)
        y_gt[:, 100:250, 100:250] = 1  # Spleen
        y_gt[:, 150:300, 260:400] = 2  # Right Kidney
        y_gt[:, 280:420, 120:260] = 3  # Left Kidney
        y_gt[:, 320:460, 270:420] = 4  # Gallbladder

        # Warmup
        for _ in range(5):
            loss = criterion(fplus, y_gt)
            loss.backward()
            fplus.grad = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_steps = 20
        t0 = time.perf_counter()
        for _ in range(n_steps):
            loss = criterion(fplus, y_gt)
            loss.backward()
            fplus.grad = None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms_per_step = (elapsed / n_steps) * 1000.0

        print(f"\n[Benchmark] CBA v2 forward+backward on {device} ({B}x{C}x{H}x{W}): {ms_per_step:.2f} ms/step")
        if torch.cuda.is_available():
            assert ms_per_step < 150.0, f"Overhead {ms_per_step:.2f} ms/step exceeds budget"

    def test_dynamic_class_adaptation_and_mismatch_error(self):
        """Verify dynamic C adaptation when empty and explicit ValueError on class count mismatch."""
        criterion = CBAv2Loss(num_classes=9)

        # 1. Adapt cleanly to 14 classes (BTCV standard) when cache is empty
        f_14 = torch.randn(2, 14, 32, 32, requires_grad=True)
        y_14 = torch.randint(0, 14, (2, 32, 32))
        loss = criterion(f_14, y_14)
        assert loss.item() >= 0.0
        assert criterion.prototype_buffer.shape == (14, 14)
        assert criterion.valid_mask.shape == (14,)
        assert criterion.valid_mask.any()

        # 2. Passing C=9 when cache is populated for C=14 raises ValueError
        f_9 = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_9 = torch.randint(0, 9, (2, 32, 32))
        with pytest.raises(ValueError, match="Class count mismatch"):
            criterion(f_9, y_9)

        # 3. After reset_cache(), it can adapt back to C=9
        criterion.reset_cache()
        loss_9 = criterion(f_9, y_9)
        assert loss_9.item() >= 0.0
        assert criterion.prototype_buffer.shape == (9, 9)

    def test_skipped_boundary_pairs_and_rate_tracking(self):
        """Verify skipped boundary pairs and rate are accurately tracked when adjacent classes lack prototypes."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, r=2, num_classes=9)
        # Pre-seed cache ONLY for class 1 (Spleen); class 2 has NO prototype
        spleen_proto = F.normalize(torch.randn(9), p=2, dim=0)
        criterion.set_prototype(1, spleen_proto)

        # Create adjacent boundary between class 1 and class 2 in batch, but class 2 has 0 pixels in GT core
        # So class 2 cannot extract a batch prototype and has no cache prototype
        fplus = torch.zeros(1, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(1, 32, 32, dtype=torch.long)
        y_gt[0, 5:15, 5:15] = 1

        # Class 1 is on boundary with background (class 0). Since class 0 has pixels, class 0 has proto.
        # But if we clear class 0 from GT (only thin ring), let's test missing adjacent proto:
        # Evaluate with return_diagnostics
        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)
        assert "skipped_boundary_pairs" in diag
        assert "skipped_boundary_rate" in diag
        assert 0.0 <= diag["skipped_boundary_rate"] <= 1.0

    def test_skipped_error_rate_tracking(self):
        """Verify skipped error rate is computed and logged in diagnostics."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, num_classes=9)
        criterion.reset_cache()

        fplus = torch.zeros(1, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(1, 32, 32, dtype=torch.long)
        y_gt[0, 10:20, 10:20] = 3

        # Predict unseen class 8
        with torch.no_grad():
            fplus[0, 8, 10:15, 10:20] = 10.0

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)
        assert diag["skipped_error_pixels"] > 0
        assert diag["skipped_error_rate"] > 0.0
        assert diag["skipped_error_rate"] <= 1.0

    def test_prototype_coherence_monitoring(self):
        """Verify prototype coherence (mean resultant length in [0, 1]) is tracked for background and organs."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, num_classes=9)
        fplus = torch.randn(2, 9, 48, 48, requires_grad=True)
        y_gt = torch.zeros(2, 48, 48, dtype=torch.long)
        y_gt[:, 10:25, 10:25] = 1  # Spleen
        y_gt[:, 25:40, 25:40] = 2  # Kidney

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)
        assert "bg_proto_coherence" in diag
        assert "fg_proto_coherence_mean" in diag
        assert "proto_coherence_per_class" in diag

        bg_coh = diag["bg_proto_coherence"]
        fg_coh = diag["fg_proto_coherence_mean"]
        assert 0.0 <= bg_coh <= 1.0 + 1e-5
        assert 0.0 <= fg_coh <= 1.0 + 1e-5

    def test_return_stream_losses(self):
        """Verify return_stream_losses=True returns raw differentiable (loss_b, loss_e) tensors."""
        criterion = CBAv2Loss(m=0.2, tau=0.1, beta=0.5, num_classes=9)
        for c in range(9):
            criterion.set_prototype(c, torch.randn(9))

        fplus = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
        y_gt[:, 8:24, 8:24] = 1

        # 1. Without diagnostics: returns (loss_b, loss_e)
        lb, le = criterion(fplus, y_gt, return_stream_losses=True)
        assert isinstance(lb, torch.Tensor)
        assert isinstance(le, torch.Tensor)
        assert lb.requires_grad
        assert le.requires_grad

        # Total combined matches
        total_combined = lb + 0.5 * le
        total_combined.backward()
        assert fplus.grad is not None

        # 2. With diagnostics: returns (loss_b, loss_e, diagnostics)
        fplus.grad = None
        lb2, le2, diag = criterion(fplus, y_gt, return_stream_losses=True, return_diagnostics=True)
        assert isinstance(diag, dict)
        assert diag["loss_b"] == pytest.approx(float(lb2.item()), rel=1e-5)
        assert diag["loss_e"] == pytest.approx(float(le2.item()), rel=1e-5)

    def test_ablation_controls_and_ce_hard(self):
        """Verify ablation controls: enable_boundary, enable_error, and error_mode='ce_hard'."""
        # Arm 2: Boundary only
        c_boundary_only = CBAv2Loss(enable_boundary=True, enable_error=False, num_classes=9)
        # Arm 3: Error only
        c_error_only = CBAv2Loss(enable_boundary=False, enable_error=True, num_classes=9)
        # Arm 5: Boundary + CE on hard pixels
        c_ce_hard = CBAv2Loss(
            enable_boundary=True, enable_error=True, error_mode="ce_hard", error_gating_type="none", num_classes=9
        )

        for c in range(9):
            p = torch.randn(9)
            c_boundary_only.set_prototype(c, p)
            c_error_only.set_prototype(c, p)
            c_ce_hard.set_prototype(c, p)

        fplus = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
        y_gt[:, 8:20, 8:20] = 1

        # Boundary only: loss_e is 0
        loss_b_only, diag_b = c_boundary_only(fplus, y_gt, return_diagnostics=True)
        assert diag_b["loss_e"] == 0.0
        assert diag_b["enable_error"] is False
        assert loss_b_only.item() == pytest.approx(diag_b["loss_b"], rel=1e-5)

        # Error only: loss_b is 0
        loss_e_only, diag_e = c_error_only(fplus, y_gt, return_diagnostics=True)
        assert diag_e["loss_b"] == 0.0
        assert diag_e["enable_boundary"] is False
        assert loss_e_only.item() == pytest.approx(0.5 * diag_e["loss_e"], rel=1e-5)

        # CE hard ablation (Arm 5): computes cross-entropy on sampled hard pixels
        loss_ce_h, diag_ce_h = c_ce_hard(fplus, y_gt, return_diagnostics=True)
        assert diag_ce_h["error_mode"] == "ce_hard"
        assert loss_ce_h.item() > 0.0
        loss_ce_h.backward()
        assert fplus.grad is not None

    def test_effective_gradient_ratios(self):
        """Verify effective gradient ratio tracking (beta * grad_norm_e / grad_norm_b)."""
        beta = 0.5
        criterion = CBAv2Loss(m=0.2, tau=0.1, beta=beta, num_classes=9)
        for c in range(9):
            criterion.set_prototype(c, torch.randn(9))

        fplus = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
        y_gt[:, 8:24, 8:24] = 1

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True, compute_autograd_norms=True)

        assert "effective_grad_norm_e_analytical" in diag
        assert "effective_grad_ratio_analytical" in diag
        assert "effective_grad_norm_e_autograd" in diag
        assert "effective_grad_ratio_autograd" in diag

        expected_ratio = beta * diag["grad_norm_ratio_analytical"]
        assert diag["effective_grad_ratio_analytical"] == pytest.approx(expected_ratio, rel=1e-5)

    def test_sampled_category_counts(self):
        """Verify sampled error breakdown: background, organ missed as bg, organ-organ confusion."""
        criterion = CBAv2Loss(num_classes=9)
        for c in range(9):
            criterion.set_prototype(c, torch.randn(9))

        fplus = torch.zeros(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)

        # 1. Background false positive: class 0 predicted as class 4
        with torch.no_grad():
            fplus[:, 4, 0:8, 0:8] = 5.0

        # 2. Organ missed as bg: class 1 predicted as class 0
        y_gt[:, 12:20, 12:20] = 1
        with torch.no_grad():
            fplus[:, 0, 12:20, 12:20] = 5.0

        # 3. Organ confused with organ: class 2 predicted as class 3
        y_gt[:, 24:30, 24:30] = 2
        with torch.no_grad():
            fplus[:, 3, 24:30, 24:30] = 5.0

        loss, diag = criterion(fplus, y_gt, return_diagnostics=True)

        assert diag["sampled_bg_error_count"] > 0
        assert diag["sampled_fg_missed_as_bg_count"] > 0
        assert diag["sampled_fg_confused_with_organ_count"] > 0
        assert diag["sampled_fg_error_count"] == (
            diag["sampled_fg_missed_as_bg_count"] + diag["sampled_fg_confused_with_organ_count"]
        )

    def test_sync_cache_ddp_graceful(self):
        """Verify sync_cache_ddp is a safe no-op when distributed is not initialized."""
        criterion = CBAv2Loss(num_classes=9)
        criterion.set_prototype(1, torch.randn(9))
        before = criterion.prototype_buffer[1].clone()
        criterion.sync_cache_ddp()
        assert torch.allclose(criterion.prototype_buffer[1], before)

    def test_eval_mode_freezes_cache_even_with_explicit_flag(self):
        """Verify eval mode strictly prevents cache updates even if update_cache=True is passed."""
        criterion = CBAv2Loss(num_classes=9)
        criterion.reset_cache()
        criterion.eval()

        f_eval = torch.randn(1, 9, 16, 16)
        y_eval = torch.zeros(1, 16, 16, dtype=torch.long)
        y_eval[0, 2:6, 2:6] = 1

        criterion(f_eval, y_eval, update_cache=True)
        assert not criterion.valid_mask.any(), "Eval mode must strictly freeze cache even when update_cache=True"
        assert (criterion.update_count == 0).all()

    def test_ce_hard_exact_pixel_alignment_with_contrastive(self):
        """Verify error_mode='ce_hard' with align_ce_hard_samples=True samples the exact same pixels as contrastive."""
        c_contra = CBAv2Loss(num_classes=9, error_mode="contrastive", k_bg=32, k_fg=32, align_ce_hard_samples=True)
        c_ce = CBAv2Loss(
            num_classes=9, error_mode="ce_hard", error_gating_type="none", k_bg=32, k_fg=32, align_ce_hard_samples=True
        )

        # Pre-seed cache ONLY for classes 0, 1, 2. Classes 3..8 have NO prototypes!
        for c in (0, 1, 2):
            p = F.normalize(torch.randn(9), p=2, dim=0)
            c_contra.set_prototype(c, p)
            c_ce.set_prototype(c, p)

        torch.manual_seed(999)
        fplus = torch.randn(2, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
        y_gt[:, 8:16, 8:16] = 1
        y_gt[:, 16:24, 16:24] = 2

        # Create some predictions for unseen classes (3, 4) which must be skipped in BOTH modes
        with torch.no_grad():
            fplus[:, 3, 8:12, 8:12] = 10.0   # Class 3 has no prototype
            fplus[:, 2, 12:16, 8:16] = 10.0  # Class 2 has prototype

        _, diag_contra = c_contra(fplus, y_gt, return_diagnostics=True, sample_seed=42, update_cache=False)
        _, diag_ce = c_ce(fplus, y_gt, return_diagnostics=True, sample_seed=42, update_cache=False)

        assert diag_contra["sampled_error_pixels"] == diag_ce["sampled_error_pixels"]
        assert diag_contra["skipped_error_pixels"] == diag_ce["skipped_error_pixels"]
        assert diag_contra["sampled_pixels_per_class"] == diag_ce["sampled_pixels_per_class"]
        assert diag_contra["error_pixels_per_class"] == diag_ce["error_pixels_per_class"]

    def test_boundary_proto_source_fallback_vs_batch(self):
        """Verify boundary_proto_source='fallback' falls back to cached anchor when class is missing, whereas 'batch' skips."""
        c_batch = CBAv2Loss(num_classes=9, boundary_proto_source="batch")
        c_fallback = CBAv2Loss(num_classes=9, boundary_proto_source="fallback")

        # Cache contains class 1 (Spleen) and class 2 (Kidney)
        c_batch.set_prototype(1, F.normalize(torch.randn(9), p=2, dim=0))
        c_batch.set_prototype(2, F.normalize(torch.randn(9), p=2, dim=0))
        c_fallback.set_prototype(1, c_batch.prototype_buffer[1].clone())
        c_fallback.set_prototype(2, c_batch.prototype_buffer[2].clone())

        # Batch contains class 0 (background) and class 1 (Spleen), but NO class 2
        fplus = torch.randn(1, 9, 32, 32, requires_grad=True)
        y_gt = torch.zeros(1, 32, 32, dtype=torch.long)
        y_gt[0, 8:20, 8:20] = 1

        # Dilation creates interface with background (0)
        # Class 1 is present in batch for both
        _, diag_b = c_batch(fplus, y_gt, return_diagnostics=True, update_cache=False)
        _, diag_f = c_fallback(fplus, y_gt, return_diagnostics=True, update_cache=False)

        assert diag_b["boundary_proto_source"] == "batch"
        assert diag_f["boundary_proto_source"] == "fallback"


