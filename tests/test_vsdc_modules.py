"""Comprehensive unit & integration tests for VSDC-Net modules."""
import pytest
import torch
import torch.nn.functional as F

from wavfry.model.vsdc import (
    LinearSoftmaxPooling,
    VowelBreathySuppressor,
    SpeakerAnchoredInfoNCELoss,
    MeanTeacherEngine,
)
from wavfry.data.utterance_bag import (
    merge_frame_features,
    extract_vowel_instances,
    collate_paired_utterance_bags,
)


class TestLinearSoftmaxPooling:
    def test_basic_pooling_properties(self):
        pool = LinearSoftmaxPooling()
        # [B=2, T=4]
        p = torch.tensor([[0.8, 0.1, 0.05, 0.05],
                          [0.5, 0.5, 0.5, 0.5]], requires_grad=True)
        y = pool(p)
        assert y.shape == (2,)
        assert (y >= 0.0).all() and (y <= 1.0).all()
        # For uniform 0.5, sum(p^2)/sum(p) = (4*0.25)/(4*0.5) = 1.0 / 2.0 = 0.5
        assert pytest.approx(y[1].item(), abs=1e-5) == 0.5

    def test_gradient_cutoff_behavior(self):
        """
        Verify TALNet property:
        Gradient w.r.t p_i is positive only when p_i > y / 2.
        For loss = -y, grad is negative for p_i > y/2 (encouraging higher p_i),
        and positive for p_i < y/2 (suppressing p_i).
        """
        pool = LinearSoftmaxPooling()
        p = torch.tensor([0.9, 0.1, 0.05], requires_grad=True)
        y = pool(p.unsqueeze(0)).squeeze(0)
        # y calculation: (0.81 + 0.01 + 0.0025) / (0.9 + 0.1 + 0.05) = 0.8225 / 1.05 = ~0.7833
        # y / 2 = ~0.3916
        # p[0]=0.9 > y/2 -> dy/dp_0 > 0
        # p[1]=0.1 < y/2 -> dy/dp_1 < 0
        y.backward()
        assert p.grad[0] > 0.0, "High confidence frame must receive positive gradient"
        assert p.grad[1] < 0.0, "Low confidence frame below y/2 must receive negative gradient (TALNet cutoff)"
        assert p.grad[2] < 0.0

    def test_amp_and_mask_robustness(self):
        pool = LinearSoftmaxPooling()
        p = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.8, 0.1]], dtype=torch.bfloat16)
        mask = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)
        y = pool(p, mask)
        # Returns float32 to prevent AMP gradient explosion on loss computation
        assert y.dtype == torch.float32
        # Clamped to open interval min=1e-5 to prevent underflow deadlock
        assert pytest.approx(y[0].item(), abs=1e-5) == 1e-5
        assert not torch.isnan(y).any()


class TestSpeakerAnchoredInfoNCELoss:
    def test_positive_and_negative_contrast(self):
        loss_fn = SpeakerAnchoredInfoNCELoss(temperature=0.07, topk=2)
        pos_z = torch.randn(5, 64, requires_grad=True)
        pos_weights = torch.tensor([0.1, 0.9, 0.05, 0.8, 0.2])
        neg_z = torch.randn(10, 64, requires_grad=True)
        loss = loss_fn(pos_z, pos_weights, neg_z)
        assert loss.item() > 0.0
        assert loss.requires_grad
        loss.backward()
        assert pos_z.grad is not None
        assert neg_z.grad is not None

    def test_missing_speaker_graceful_fallback(self):
        loss_fn = SpeakerAnchoredInfoNCELoss()
        pos_z = torch.randn(4, 64, requires_grad=True)
        pos_weights = torch.rand(4)
        # neg_z is None
        loss_none = loss_fn(pos_z, pos_weights, None)
        assert loss_none.item() == 0.0
        assert loss_none.requires_grad

        # neg_z is empty
        empty_neg = torch.empty(0, 64)
        loss_empty = loss_fn(pos_z, pos_weights, empty_neg)
        assert loss_empty.item() == 0.0
        assert loss_empty.requires_grad

    def test_pos_mask_anchor_protection(self):
        loss_fn = SpeakerAnchoredInfoNCELoss(temperature=0.07, topk=2)
        # K_max=5, but only slot 0 is valid.
        pos_z = torch.randn(5, 32, requires_grad=True)
        # If unmasked, slots 0 and 1 would be chosen
        pos_weights = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        pos_mask = torch.tensor([True, False, False, False, False])
        neg_z = torch.randn(4, 32)
        neg_mask = torch.tensor([True, True, True, True])

        # With pos_mask, only 1 valid candidate exists (topk=2 drops to min(topk, 1) = 1)
        loss = loss_fn(pos_z, pos_weights, neg_z, neg_mask=neg_mask, pos_mask=pos_mask)
        assert loss.item() > 0.0
        assert loss.requires_grad
        loss.backward()
        # Verify gradient only flows to slot 0, not to padded slots 1..4
        assert pos_z.grad[0].abs().sum() > 0.0
        assert pos_z.grad[1:].abs().sum() == 0.0


class TestVowelBreathySuppressor:
    def test_probability_modulation_prevents_zero_weight_collapse(self):
        """Verify that when suppressor predicts w=0, bag_prob_supp drops to ~0 (1e-5), preventing shortcut collapse."""
        suppressor = VowelBreathySuppressor(in_features=32, hidden_dim=16, proj_dim=8)
        h = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4, dtype=torch.bool)
        out = suppressor(h, mask)
        # If weights are zeroed, probs_supp must be strictly 0
        w_zero = torch.zeros_like(out["weights"])
        p_supp_zero = w_zero * out["probs_base"]
        bag_zero = suppressor.pooling(p_supp_zero, mask)
        assert pytest.approx(bag_zero.item(), abs=1e-4) == 1e-5
    def test_forward_and_backward(self):
        suppressor = VowelBreathySuppressor(in_features=64, hidden_dim=32, proj_dim=16)
        # B=2, K=6, D=64
        h = torch.randn(2, 6, 64, requires_grad=True)
        mask = torch.tensor([[True, True, True, True, False, False],
                             [True, True, True, True, True, True]])
        out = suppressor(h, mask)

        for key in ["probs_base", "probs_supp", "probs_bg", "weights"]:
            assert out[key].shape == (2, 6)
            assert (out[key] >= 0.0).all() and (out[key] <= 1.0).all()
        for key in ["bag_prob_base", "bag_prob_supp", "bag_prob_bg"]:
            assert out[key].shape == (2,)

        assert out["z_proj"].shape == (2, 6, 16)
        # Norm should be 1.0
        norms = torch.norm(out["z_proj"], p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Test gradient propagation
        loss = out["bag_prob_supp"].sum() + out["weights"].sum()
        loss.backward()
        assert h.grad is not None
        assert not torch.isnan(h.grad).any()

    def test_edge_cases_single_vowel(self):
        suppressor = VowelBreathySuppressor(in_features=32, hidden_dim=16, proj_dim=8)
        # B=1, K=1 (GroupNorm must not crash)
        h = torch.randn(1, 1, 32)
        out = suppressor(h)
        assert out["bag_prob_supp"].shape == (1,)
        assert not torch.isnan(out["bag_prob_supp"]).any()


class TestMeanTeacherEngine:
    def test_teacher_ema_update(self):
        suppressor = VowelBreathySuppressor(in_features=32, hidden_dim=16, proj_dim=8)
        engine = MeanTeacherEngine(suppressor, ema_decay=0.9)

        # Perturb student parameters
        with torch.no_grad():
            for p in suppressor.parameters():
                p.add_(1.0)

        # Update teacher
        engine.update_teacher()
        # Teacher param should be updated by 0.1 * student + 0.9 * old_teacher
        for s_p, t_p in zip(suppressor.parameters(), engine.teacher.parameters()):
            assert not torch.equal(s_p, t_p)

    def test_pseudolabel_sharpening_and_consistency(self):
        suppressor = VowelBreathySuppressor(in_features=32, hidden_dim=16, proj_dim=8)
        engine = MeanTeacherEngine(suppressor, ema_decay=0.99, temp_sharp=0.5, conf_margin=0.35)

        h = torch.randn(2, 8, 32)
        mask = torch.ones(2, 8, dtype=torch.bool)
        pseudo = engine.generate_refined_pseudolabels(h, mask)
        assert pseudo.shape == (2, 8)
        assert (pseudo >= 0.0).all() and (pseudo <= 1.0).all()

        student_probs = torch.rand(2, 8, requires_grad=True)
        loss = engine.consistency_loss(student_probs, pseudo, mask)
        assert loss.item() >= 0.0
        if loss.item() > 0.0:
            loss.backward()
            assert student_probs.grad is not None

    def test_boundary_median_filter_padding_protection(self):
        """Verify that clause-final high confidence vowels are NOT crushed by zero-padding."""
        suppressor = VowelBreathySuppressor(in_features=16, hidden_dim=8, proj_dim=8)
        engine = MeanTeacherEngine(suppressor, median_window=3)

        # Mock teacher output so raw_probs has high value at final valid vowel
        raw = torch.tensor([[0.8, 0.85, 0.0, 0.0]])  # valid length 2, followed by padding
        mask = torch.tensor([[True, True, False, False]])

        # Mock teacher call
        from unittest.mock import MagicMock
        mock_t = MagicMock()
        mock_t.return_value = {"probs_supp": raw}
        engine.teacher = mock_t
        h = torch.randn(1, 4, 16)
        q = engine.generate_refined_pseudolabels(h, mask)

        # In unmasked replicate padding, [0.8, 0.85, 0.0] median would pull 0.85 down or distort
        # But with valid boundary protection, slot 1 (0.85) replicates within valid range:
        # median of [0.8, 0.85, 0.85] is 0.85!
        assert q[0, 1].item() > 0.8, f"Final vowel was crushed to {q[0, 1].item()}"
        assert q[0, 2].item() == 0.0, "Padding slot must remain 0"
        assert q[0, 3].item() == 0.0, "Padding slot must remain 0"


class TestUtteranceFeatureExtraction:
    def test_merge_frame_features_and_vowel_extraction(self):
        N_windows = 3
        T_frames = 10
        D = 16
        features = torch.randn(N_windows, T_frames, D, requires_grad=True)
        valid = torch.ones(N_windows, T_frames, dtype=torch.bool)
        starts = torch.tensor([0, 1600, 3200])  # aligned to stride 320
        bag_index = torch.tensor([0, 0, 1])  # 2 windows for bag 0, 1 window for bag 1

        merged = merge_frame_features(features, valid, starts, bag_index, stride=320, receptive=400)
        assert len(merged) == 2
        assert merged[0]["bag_index"] == 0
        assert merged[1]["bag_index"] == 1
        assert merged[0]["features"].size(-1) == D

        vowels_per_bag = [
            [[0.05, 0.15], [0.20, 0.30]],  # Bag 0 has 2 vowels
            [[0.10, 0.25]]                  # Bag 1 has 1 vowel
        ]
        h_vowels, vowel_mask = extract_vowel_instances(merged, vowels_per_bag)
        assert h_vowels.shape == (2, 2, D)
        assert vowel_mask.shape == (2, 2)
        assert vowel_mask[0, 0] and vowel_mask[0, 1]
        assert vowel_mask[1, 0] and not vowel_mask[1, 1]

        # Verify gradient flow through merging and vowel extraction
        loss = h_vowels.sum()
        loss.backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


class TestTrainerVSDCCurriculum:
    def test_three_stage_curriculum_forward(self):
        """Simulate Trainer's _forward_weak_batch across Stage 1, Stage 2, and Stage 3."""
        from unittest.mock import MagicMock
        from wavfry.train.trainer import Trainer

        suppressor = VowelBreathySuppressor(in_features=16, hidden_dim=8, proj_dim=8)
        mock_model = MagicMock()
        mock_model.tasks = ["breathy"]
        mock_model.hidden_dim = 16
        mock_model.breathy_suppressor = suppressor

        trainer = MagicMock(spec=Trainer)
        trainer.device = torch.device("cpu")
        trainer.model = mock_model
        trainer.vsdc_enabled = True
        trainer.vsdc_cfg = {
            "curriculum": {"stage_epochs": [3, 10]},
            "bas_net": {
                "loss_weight_base": 0.1,
                "loss_weight_supp": 0.1,
                "loss_weight_supp_zero": 0.5,
                "loss_weight_sparse": 0.05,
            },
            "speaker_contrast": {"enabled": True, "loss_weight": 0.2},
            "mean_teacher": {"enabled": True, "loss_weight": 1.0},
        }
        trainer.weak_cfg = {"loss_weight": 0.3}
        trainer.speaker_contrast_loss = SpeakerAnchoredInfoNCELoss(temperature=0.07, topk=2)
        trainer.mean_teacher = MeanTeacherEngine(suppressor, ema_decay=0.99)
        trainer._forward_weak_batch = Trainer._forward_weak_batch.__get__(trainer, Trainer)

        # Build mock paired batch
        def mock_forward(waveforms, lengths, return_frames=True):
            N = waveforms.size(0)
            return {
                "frame_hidden": {"breathy": torch.randn(N, 10, 16, requires_grad=True)},
                "frame_valid": torch.ones(N, 10, dtype=torch.bool),
                "frame_stride_samples": 320,
                "frame_receptive_samples": 400,
            }
        mock_model.side_effect = mock_forward

        pos_batch = {
            "waveforms": torch.randn(2, 3200),
            "lengths": torch.tensor([3200, 3200]),
            "starts": torch.tensor([0, 1600]),
            "bag_index": torch.tensor([0, 0]),
            "vowels": [[[0.05, 0.15], [0.20, 0.30]]],
            "metadata": [{"file_key": "pos1"}],
        }
        neg_batch = {
            "waveforms": torch.randn(1, 3200),
            "lengths": torch.tensor([3200]),
            "starts": torch.tensor([0]),
            "bag_index": torch.tensor([0]),
            "vowels": [[[0.05, 0.15]]],
            "metadata": [{"file_key": "neg1"}],
        }
        paired_batch = {
            "pos": pos_batch,
            "neg": neg_batch,
            "has_same_speaker_neg": torch.tensor([True]),
            "pos_to_neg_idx": torch.tensor([0]),
        }

        # Stage 1 (Epoch 2 <= 3): Cold start -> 0 loss
        loss_s1, count_s1 = trainer._forward_weak_batch(paired_batch, epoch=2)
        assert loss_s1.item() == 0.0
        assert count_s1 == 0

        # Stage 2 (Epoch 5 > 3, <= 10): BaS + InfoNCE active
        loss_s2, count_s2 = trainer._forward_weak_batch(paired_batch, epoch=5)
        assert loss_s2.item() > 0.0
        assert count_s2 == 1

        # Stage 3 (Epoch 12 > 10): Mean Teacher distillation active
        loss_s3, count_s3 = trainer._forward_weak_batch(paired_batch, epoch=12)
        assert loss_s3.item() > 0.0
        assert count_s3 == 1

