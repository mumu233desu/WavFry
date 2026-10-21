"""Tests for Multi-Backbone Comparison: Table 5 formatting and CSV consistency."""

import pytest
import torch
import pandas as pd
from scripts.backbone_comparison.run_backbone_comparison import format_academic_summary_table
from wavfry.eval.unified import f1_scan_best_threshold


def test_f1_scan_best_threshold_synthetic():
    logits = torch.tensor([-2.0, -1.0, 0.5, 2.0, 3.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    best_t, best_f1 = f1_scan_best_threshold(logits, labels)
    assert 0.01 <= best_t <= 0.99
    assert best_f1 == 1.0


def test_table5_markdown_formatting_with_breathy():
    mock_results = [
        {
            "backbone": "wavlm",
            "display_name": "WavLM-Base",
            "conditions": {
                "clean": {
                    "auc_pr_creaky": 0.9250,
                    "f1_creaky_calibrated": 0.8400,
                    "auc_pr_breathy": 0.7820,
                    "f1_breathy_calibrated": 0.6950,
                    "f1_breathy_best": 0.7320,
                    "threshold_breathy_best": 0.35,
                },
                "cafe_15db": {
                    "auc_pr_creaky": 0.8920,
                },
            },
        },
        {
            "backbone": "hubert",
            "display_name": "HuBERT-Base",
            "conditions": {
                "clean": {
                    "auc_pr_creaky": 0.9110,
                    "f1_creaky_calibrated": 0.8250,
                    "auc_pr_breathy": 0.7540,
                    "f1_breathy_calibrated": 0.6710,
                    "f1_breathy_best": 0.7100,
                    "threshold_breathy_best": 0.38,
                },
                "cafe_15db": {
                    "auc_pr_creaky": 0.8710,
                },
            },
        },
    ]

    table_md = format_academic_summary_table(mock_results, conditions=["clean", "cafe_15db"])
    assert "Breathy PR" in table_md
    assert "Breathy F1 (Cal.)" in table_md
    assert "Breathy F1 (Best)" in table_md
    assert "**0.7320** (@0.35)" in table_md
    assert "Table 5: Multi-Backbone Comparison Ablation" in table_md


def test_table5_detail_csv_columns():
    expected_cols = [
        "Model", "Task", "Condition", "Scenario", "SNR_dB", "Tier", "Has_RIR",
        "PR-AUC", "Delta-PR-AUC", "ROC-AUC", "F1-Score", "F1-Calibrated",
        "Threshold-Calibrated", "F1-Best", "Threshold-Best", "Precision", "Recall"
    ]
    detail_rows = [
        {
            "Model": "WavLM-Base",
            "Task": "breathy",
            "Condition": "clean",
            "Scenario": "Clean",
            "SNR_dB": "inf",
            "Tier": "Clean",
            "Has_RIR": False,
            "PR-AUC": 0.7820,
            "Delta-PR-AUC": 0.0,
            "ROC-AUC": 0.8910,
            "F1-Score": 0.6500,
            "F1-Calibrated": 0.6950,
            "Threshold-Calibrated": 0.4200,
            "F1-Best": 0.7320,
            "Threshold-Best": 0.3500,
            "Precision": 0.7100,
            "Recall": 0.6800,
        }
    ]
    df = pd.DataFrame(detail_rows)
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"
