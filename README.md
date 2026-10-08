# WavFry: WavLM-based Creaky Voice Detection

WavFry is a sub-project for high-precision creaky voice detection in the CS-Dialogue corpus using fine-tuned WavLM models.

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

See [usage and training configuration](docs/weak_breathy_supervision.md).

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
