# WavFry: WavLM-based Creaky Voice Detection

WavFry is a framework for high-precision, noise-robust creaky voice (vocal fry) detection under complex acoustic degradations using fine-tuned WavLM models.

---

## 🌐 Interactive Demo

Interactive audio examples and model detections across 18 acoustic degradation conditions:  
👉 [https://mumu233desu.github.io/wavfry-demo/](https://mumu233desu.github.io/wavfry-demo/)

---

## 📦 Pretrained Checkpoints (Release v1.0.0)

Pretrained delta checkpoints and complete ablation suite weights from the paper *WavFry: Noise-Robust Phonation Detection and Functional Representation Auditing in Conversational Speech* are available on [GitHub Releases (v1.0.0)](https://github.com/mumu233desu/WavFry/releases/tag/v1.0.0):

| Model Checkpoint | Underlying SSL Backbone | Architecture Description | Creaky PR-AUC | Creaky ROC-AUC | Optimal Thresholds |
| :--- | :--- | :--- | :---: | :---: | :---: |
| [`wavfry_wavlm_baseplus.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_wavlm_baseplus.pt) | `microsoft/wavlm-base-plus` | **Primary Dual-Task Model** (VSDC-Net & Dynamic Masking) | **0.9133** | **0.9841** | `creaky: 0.560 / breathy: 0.440` |
| [`wavfry_wavlm_base.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_wavlm_base.pt) | `microsoft/wavlm-base` | Cross-Backbone Comparison | 0.9132 | 0.9850 | `creaky: 0.549` |
| [`wavfry_hubert_base.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_hubert_base.pt) | `facebook/hubert-base-ls960` | Cross-Backbone Comparison | 0.9135 | 0.9851 | `creaky: 0.570` |
| [`wavfry_wav2vec2_base.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_wav2vec2_base.pt) | `facebook/wav2vec2-base` | Cross-Backbone Comparison | 0.9143 | 0.9844 | `creaky: 0.505 / breathy: 0.470` |
| [`wavfry_weak_off.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_weak_off.pt) | `microsoft/wavlm-base-plus` | Ablation Baseline (Supervised Only) | 0.9122 | 0.9844 | `creaky: 0.580 / breathy: 0.500` |
| [`wavfry_weak_mil_positive_only.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_weak_mil_positive_only.pt) | `microsoft/wavlm-base-plus` | Ablation (Positive Bags Only MIL) | 0.9132 | 0.9839 | `creaky: 0.560 / breathy: 0.480` |
| [`wavfry_weak_mil_soft_weight.pt`](https://github.com/mumu233desu/WavFry/releases/download/v1.0.0/wavfry_weak_mil_soft_weight.pt) | `microsoft/wavlm-base-plus` | Ablation (Soft Instance Weighting MIL) | 0.9133 | 0.9842 | `creaky: 0.555 / breathy: 0.440` |

*Weights are stored as lightweight delta bundles (~1.5 MB to ~4.4 MB); base models are automatically downloaded from Hugging Face Hub on demand.*

### Direct Inference with Pretrained Checkpoints:
```bash
# Segment-level evaluation with TextGrid intervals
uv run python scripts/predict.py \
  --checkpoint deployments/checkpoints_v1.0.0/wavfry_wavlm_baseplus.pt \
  --audio-dir /path/to/audios \
  --textgrid-dir /path/to/textgrids \
  --output-csv predictions.csv \
  --task creaky
```

---

## Quick Start

1. **Build Manifest**:
```bash
uv run python scripts/build_manifest.py
```
2. **Train**:
```bash
# Standard training (using config's val_speakers)
uv run python scripts/train.py --config config/_defaults.yaml --resume

# Leave-One-Speaker-Out Cross Validation
# (Set 'val_speakers: loocv' in config, or use --loocv flag)
# uv run python scripts/train.py --override data.val_speakers=loocv
```
3. **Predict**:
```bash
uv run python scripts/predict.py --checkpoint checkpoints/best_model.pt --audio-dir <path> --textgrid-dir <path> --output-csv results.csv
```
4. **Archive Results**:
To package your experiment (runs, reports, and checkpoints) for backup:
```bash
uv run python scripts/package_experiment.py
```
This will create a `wavfry_exp_YYYYMMDD_{N}spks.zip` in the project root.

5. **Export to TextGrid** (Optional):
```bash
uv run python scripts/export_textgrid.py --csv results.csv --tg-dir <path> --output-dir <path>
```

## Weak Supervision & Annotation Review

- **Optional breathy supervision**: `scripts/build_weak_manifest.py` imports existing MLLM caches offline. A positive label means breathy occurs somewhere in the complete audio clip—not in every frame or block. MLLM timestamps are review hints; training pools over all vowels from the original TextGrid `phones` tier. Unmarked clips are not negatives. Enable `training.weak_supervision` explicitly; it is off by default.
- **Self-check after training (creaky or breathy)**: read original batch annotation shards directly—no training manifest is required. Use a checkpoint with the selected task's trained frame head and LSE pooling:

```bash
uv run python scripts/review_annotation_quality.py \
  --checkpoint <checkpoint.pt> --annotation-cache <_creaky_cache> \
  --audio-dir <audio_directory> --batches 0,1 --dataset-id cs_dialogue \
  --task creaky --device cpu --output-dir <new_report_directory>
# For breathy: select its _breathy_cache, audio directory and dataset ID; use --task breathy.
# One batch only: --annotation-shards <manual_creaky_annotations.json.d> instead of --annotation-cache.
# Legacy CSV input: --manual-manifest <segments_multi.csv>; optional MLLM clips: --weak-manifest <weak_breathy.csv>.
```

Original batch/shard/interval identities are retained; uncertain, machine-prefilled and automatic-silence annotations stay separate. `review_queues.csv` prioritizes low label agreement and unstable predictions, with random/high-agreement controls. `all_scores.csv` includes scores, audio paths and frame traces. Scores guide human review—not label correctness probabilities. This tool neither runs automatically after training nor changes annotations; predictions on training data may reflect memorization.

See [usage and training configuration](docs/weak_label_training_technical_specification.md).

## Monitoring

Visualize training progress, confusion matrices, and audio samples:
```bash
# Outputs (runs/ and checkpoints/) are kept inside the project root
uv run tensorboard --logdir runs/
```

## Key Features
- **WavLM Fine-tuning**: Selective layer fine-tuning (Layers 4–8, 0-indexed) with Scalar Mix.
- **Efficiency**: Delta checkpoints (~40MB), Mixed Precision (FP16), Resampler caching.
- **Robustness**: Focal Loss for class imbalance, speaker-based data splitting.
- **Observability**: TensorBoard integration with loss, metrics, and error analysis.
- **Modular Design**: Proper Python package structure (`wavfry/`) with clean imports.

## Structure
- `wavfry/`: Core package containing data loaders, model definitions, and trainer logic.
- `scripts/`: Ready-to-use entry point scripts for the pipeline.
- `config/`: YAML configurations.
- `manifests/`: Generated segment CSVs for training.
- `checkpoints/`: Model weights (Delta or Full).
- `runs/`: TensorBoard logs.
